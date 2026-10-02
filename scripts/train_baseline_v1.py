"""
Day1 실습을 시작하기 전에 1회만 실행하는 부트스트랩 스크립트입니다.

Day1에는 아직 MLflow가 등장하지 않으므로(Day2에서 도입), FastAPI 서버가
곧바로 로드할 수 있는 "사전 학습된" 로컬 LSTM 모델 파일을 만들어 둡니다.
이 모델은 Day2에서 train_and_register.py(MLflow 버전)로 대체됩니다.

여기서 fit한 스케일러(scaler.pkl)는 Day2/Day3에서도 계속 재사용됩니다 -
서빙 시점 정규화 기준이 Day1~3 내내 바뀌지 않아야 하기 때문입니다.

실행 순서:
    1) uvicorn serving_app.main:app --reload   (서버 먼저 기동 - lazy 모드라 데이터 없이도 뜹니다)
    2) 대시보드(http://localhost:8000/)에서 HAIC CSV를 업로드하세요
       (data/sample_haic_prices.csv를 예시로 업로드해볼 수 있습니다)
    3) python scripts/train_baseline_v1.py    (별도 터미널에서 - scaler.pkl 생성)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import load_rows, build_sequences, train_test_split, HAICScaler
from data.storage import latest_upload
from serving_app.lstm_model import build_model

MODEL_PATH = "serving_app/models/haic_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = 100  # 3층 LSTM + 3년치 데이터 기준, RMSE가 안정적으로 게이트 아래로 수렴하는 지점


def rmse(y_true, y_pred) -> float:
    return (sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)) ** 0.5


def main():
    import numpy as np

    rows = load_rows(latest_upload())

    scaler = HAICScaler().fit(rows)
    scaler.save(SCALER_PATH)
    print(f"scaler fit on {len(rows)}행 -> {SCALER_PATH}")

    X, y = build_sequences(rows, scaler)  # y: 스케일 안 된 실제 종가
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    X_train = np.array(X_train, dtype="float32")
    X_test = np.array(X_test, dtype="float32")
    # 입력 시퀀스와 같은 스케일로 학습해야 loss가 과도하게 커지지 않고 안정적으로 수렴한다.
    y_train_scaled = np.array([scaler.scale_close(v) for v in y_train], dtype="float32")

    model = build_model()
    model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, verbose=0)

    preds_scaled = model.predict(X_test, verbose=0).flatten()
    preds = [scaler.inverse_close(p) for p in preds_scaled]  # 실제 달러 단위로 복원
    score = rmse(y_test, preds)
    print(f"baseline v1 RMSE = {score:.2f}  (배포 게이트: $4.00)")

    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")
    if score > 4.00:
        print(
            "※ 참고: 이 RMSE는 Day1 로컬 모델이며 배포 게이트($4.00) 통과 여부는 "
            "Day2에서 MLflow로 다시 정식 검증합니다."
        )


if __name__ == "__main__":
    main()
