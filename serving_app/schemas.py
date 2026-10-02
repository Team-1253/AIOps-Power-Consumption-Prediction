"""
FastAPI 요청/응답 Pydantic 스키마.

입력 피처는 정제 CSV(data/hourly_clean.csv)의 수치 컬럼을 그대로 쓴다.
    CSV 컬럼: date_utc, time_utc, energy_kwh, humi_pct, temp_F
    입력 1건(HourlyPoint): energy_kwh, humi_pct, temp_F  (필드명 = CSV 컬럼명)
    예측 대상: 다음 1시간 energy_kwh

LSTM은 최근 SEQ_LEN시간의 흐름을 입력받으므로 /predict는 SEQ_LEN개 시퀀스를 요청 본문으로 받는다.
길이(SEQ_LEN)와 값 범위를 스키마 단에서 검증해, 학습 시점 입력(data/features.py)과 어긋나지 않게 한다.
"""
from pydantic import BaseModel, Field, PositiveFloat

from data.features import SEQ_LEN

TIME_FIELDS = ("date_utc", "time_utc")  # CSV 시간 컬럼 (입력 피처 아님)
TARGET_FIELD = "energy_kwh"             # 예측 대상 컬럼


class HourlyPoint(BaseModel):
    energy_kwh: float = Field(..., gt=0, description="해당 1시간 전력 사용량 (kWh)")
    humi_pct: float = Field(..., ge=0, le=100, description="외기 상대습도 (%)")
    temp_F: float = Field(..., description="외기 온도 (°F)")


class PredictRequest(BaseModel):
    sequence: list[HourlyPoint] = Field(
        ...,
        min_length=SEQ_LEN,
        max_length=SEQ_LEN,
        description=f"가장 오래된 시간 -> 가장 최근 시간 순서의 최근 {SEQ_LEN}시간 측정값",
    )


class PredictResponse(BaseModel):
    predicted_energy_kwh: float
    model_version: str


class BatchTestRequest(BaseModel):
    # 드리프트 시뮬레이션용. SEQ_LEN + N개의 연속된 사용량을 보내면 서버가 슬라이딩 윈도우로
    # 잘라 N건을 연속 예측한다. (습도·온도는 시뮬레이션이므로 routers/predict.py의 고정값 사용)
    # 드리프트 판정이 실제값 대비 오차율(%)이라 0 이하 값은 받지 않는다.
    energy_series: list[PositiveFloat] = Field(..., min_length=SEQ_LEN + 1)


class BatchTestResponse(BaseModel):
    predictions: list[float]
    drift_check: dict
