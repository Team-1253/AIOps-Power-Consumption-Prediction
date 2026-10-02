"""
최초 1회만 실행하는 부트스트랩 스크립트 (MLflow 없이 로컬 baseline 생성).

MLflow가 등장하기 전, FastAPI 서버가 곧바로 로드할 수 있는 "사전 학습된"
로컬 LSTM(.pt 번들)을 만들어 둔다. 학습 루프는 serving_app/train_and_register.py의
헬퍼를 재사용한다 (단일 소스).

스케일러는 여기서 fit하지 않는다 — 스케일링은 데이터셋 담당 영역이므로,
데이터셋 담당이 제공한 fit済 스케일러(data/features.py load_scaler)를 로드만 한다.

실행 (project/ 루트에서):
    python scripts/train_baseline_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serving_app.train_and_register import (
    BASE_EPOCHS,
    _fit,
    _load_tensors,
    _make_loaders,
    _save_local,
    build_model,
    evaluate,
    get_device,
    set_seed,
)

DEVICE = get_device()


def main():
    from data.features import FEATURE_COLS, TARGET_COL, load_scaler

    set_seed(42)
    scaler = load_scaler()
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = _load_tensors(scaler)
    train_loader, valid_loader, test_loader = _make_loaders(Xtr, ytr, Xva, yva, Xte, yte)

    model = build_model()
    _fit(model, train_loader, valid_loader, scaler, BASE_EPOCHS, lr=1e-3)

    test_rmse, test_mae = evaluate(model, test_loader, scaler)
    print(f"baseline v1 RMSE = {test_rmse:.4f}  MAE = {test_mae:.4f}  (게이트: 확정 전)")

    _save_local(model, scaler, FEATURE_COLS, TARGET_COL, Xtr.shape[1])


if __name__ == "__main__":
    main()
