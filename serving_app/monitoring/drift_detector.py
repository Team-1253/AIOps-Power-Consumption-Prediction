"""
드리프트 감지  —  serving_app/monitoring/drift_detector.py
"""

# RMSE는 kWh가 아니라 오차율(%)로 계산한다. 사용량 수준이 해마다 크게 달라도 같은 기준을 쓰기 위해서다.
# 대시보드 배치(시간당 변동성 1.2%)를 "직전 값 그대로" 예측하면 정상 배치 약 1.2%,
# 드리프트 배치(변동성 6배) 약 7.1%가 나온다.
RMSE_THRESHOLD = 5.0  # %. 이보다 많이 틀리면 드리프트
WINDOW_SIZE = 24  # 최근 24건(하루치)을 봅니다


def compute_rmse(recent_predictions: list[dict]) -> float:
    """
    받는 것  : [{"predicted": 100.0, "actual": 102.0}, {"predicted": 100.0, "actual": 98.0}, ...]
    돌려줄 것: 오차율 RMSE (숫자 1개, "실제값 대비 평균 몇 % 틀렸나").  빈 목록이면 0.0
    """
    import math

    # 기록이 하나도 없으면 0.0 (빈 목록이면 평균을 낼 때 0으로 나누기 에러가 나기 때문)
    if not recent_predictions:
        return 0.0

    errors_sq = [
        ((p["actual"] - p["predicted"]) / p["actual"] * 100) ** 2
        for p in recent_predictions
    ]
    return math.sqrt(sum(errors_sq) / len(errors_sq))


def is_drift(recent_predictions: list[dict]) -> bool:
    """
    드리프트인지 True/False 로 판단합니다.
    흐름: (데이터 충분한가?) → 최근 WINDOW_SIZE건만 골라서 → RMSE 계산 → 기준(RMSE_THRESHOLD)보다 크면 드리프트
    """

    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    window = recent_predictions[-WINDOW_SIZE:]
    rmse = compute_rmse(window)
    return rmse > RMSE_THRESHOLD
