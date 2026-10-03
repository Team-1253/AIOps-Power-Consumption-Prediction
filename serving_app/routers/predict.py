"""
예측 API  —  serving_app/routers/predict.py

외부 요청을 받아 모델에 전달하고 결과를 돌려주는 창구다. 계산은 모델(model_loader)과
감시 도구(retrain_trigger)가 맡는다.

■ 엔드포인트
   POST /predict             : 최근 INPUT_LEN시간 측정값 → 다음 1시간 전력 사용량
   POST /predict/batch-test  : 연속 측정값 목록 → 슬라이딩 윈도우로 여러 번 예측 → 드리프트 검사

■ 모델과의 약속 (model_loader.LoadedModel.predict_one)
   입력 : [{"energy_kwh": 2750.0, "humi_pct": 41.0, "temp_F": 58.0}, ... INPUT_LEN개]  (오래된 시간 → 최근 시간)
   출력 : 다음 1시간 energy_kwh 예측값 (kWh, float)
"""
from fastapi import APIRouter

from serving_app import model_loader
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import INPUT_LEN, BatchTestRequest, BatchTestResponse, PredictRequest, PredictResponse

router = APIRouter()

# 최근 예측 기록 [{"predicted": ..., "actual": ...}, ...].
# 드리프트 판단은 최근 WINDOW_SIZE건만 보므로 그만큼만 유지한다.
recent_predictions: list[dict] = []


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    받는 것  : {"sequence": [{"energy_kwh": ..., "humi_pct": ..., "temp_F": ...}, ... INPUT_LEN개]}
               개수가 다르면 schemas.py가 422를 돌려준다.
    돌려줄 것: {"predicted_energy_kwh": 2761.4, "model_version": "production"}
    """
    model = model_loader.get_model()
    sequence = [p.model_dump() for p in req.sequence]
    predicted = model.predict_one(sequence)
    return PredictResponse(predicted_energy_kwh=round(predicted, 2), model_version=model.version)


@router.post("/predict/batch-test", response_model=BatchTestResponse)
def batch_test(req: BatchTestRequest):
    """
    받는 것  : {"sequence": [{"energy_kwh": ..., "humi_pct": ..., "temp_F": ...}, ... INPUT_LEN + N개]}
               (대시보드 배치 주입이 GET /data/sample 의 실제 측정값을 보냄)
    돌려줄 것: {"predictions": [예측값 N개], "drift_check": {"status": "ok"} 또는 재학습 결과}

    ■ 슬라이딩 윈도우: INPUT_LEN칸 창문을 한 칸씩 밀며 바로 다음 1시간을 예측하고 실제 값과 비교한다.
        i=0 : [e0 ~ e(INPUT_LEN-1)] → 예측  vs  실제 e(INPUT_LEN)
        i=1 : [e1 ~ e(INPUT_LEN)]   → 예측  vs  실제 e(INPUT_LEN+1)
        → 총 len(sequence) - INPUT_LEN번 예측
    """
    model = model_loader.get_model()
    predictions: list[float] = []

    series = [p.model_dump() for p in req.sequence]
    for i in range(len(series) - INPUT_LEN):
        pred = model.predict_one(series[i : i + INPUT_LEN])
        actual = series[i + INPUT_LEN]["energy_kwh"]
        predictions.append(pred)
        recent_predictions.append({"predicted": pred, "actual": actual})

    # 최근 WINDOW_SIZE건만 남긴다. (recent_predictions = ... 로 쓰면 함수 안의 새 변수가 되므로 [:]로 내용을 바꾼다)
    recent_predictions[:] = recent_predictions[-WINDOW_SIZE:]

    drift_check = check_and_trigger(recent_predictions)
    return BatchTestResponse(predictions=predictions, drift_check=drift_check)
