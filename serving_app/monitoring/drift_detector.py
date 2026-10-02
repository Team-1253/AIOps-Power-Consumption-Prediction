"""
드리프트 감지 — serving_app/monitoring/drift_detector.py

■ 이 파일이 하는 일 (한 줄 요약)
   "최근 모델이 평균 얼마나 틀리고 있는지(RMSE)"를 계산해서,
   임계값을 넘게 틀리면 "데이터가 달라졌다(드리프트)"고 판단합니다.

■ 드리프트가 뭔가요?
   모델은 과거 데이터로 공부했습니다. 그런데 설비 운영 체제가 바뀌면(예: 사용량 급증)
   공부한 것과 다른 데이터가 들어와 예측이 크게 빗나가기 시작합니다. 이것이 드리프트입니다.
   그래서 최근 예측이 얼마나 틀렸는지 계속 지켜보다가, 너무 많이 틀리면 재학습을 시작합니다.

■ 판단 기준 (임계값 미확정 — PENDING)
   최근 WINDOW_SIZE(21)건의 RMSE > RMSE_THRESHOLD  →  드리프트!
   · 21건보다 짧으면 : 우연한 한두 번 실수에도 경보가 울립니다.
   · 21건보다 길면   : 상황이 바뀌어도 늦게 알아챕니다.
   · 임계값은 변화율 타깃 확정 후 팀 합의로 설정 (train_and_register.py RMSE_GATE와 함께).
"""
# NOTE(성능보류): 변화율 타깃 기준 임계값 미확정. None이면 드리프트 판정을 내리지 않는다.
RMSE_THRESHOLD = None
WINDOW_SIZE = 21       # 최근 21건을 봅니다


def compute_rmse(recent_predictions: list[dict]) -> float:
    """
    받는 것  : [{"predicted": 100.0, "actual": 102.0}, {"predicted": 100.0, "actual": 98.0}, ...]
    돌려줄 것: RMSE (숫자 1개, "평균 얼마나 틀렸나").  빈 목록이면 0.0

    ■ RMSE 계산 4단계 — 먼저 손으로 풀어 보세요
                             1건째            2건째
        ① 오차 (실제-예측)    102-100 = +2     98-100 = -2
        ② 제곱               2² = 4           (-2)² = 4
        ③ 평균               (4 + 4) / 2 = 4
        ④ 제곱근             √4 = 2.0         → "평균 2 틀렸다"

    확인 방법
      python -c "from serving_app.monitoring.drift_detector import compute_rmse; \
      print(compute_rmse([{'predicted':100,'actual':102},{'predicted':100,'actual':98}])); \
      print(compute_rmse([]))"
      → 2.0 과 0.0 이 나오면 성공
    """
    import math

    # 기록이 하나도 없으면 0.0 (빈 목록이면 ③에서 0으로 나누기 에러가 나기 때문)
    if not recent_predictions:
        return 0.0

    errors_sq = [(p["actual"] - p["predicted"]) ** 2 for p in recent_predictions]
    return math.sqrt(sum(errors_sq) / len(errors_sq))


def is_drift(recent_predictions: list[dict]) -> bool:
    """
    드리프트인지 True/False 로 판단합니다.
    흐름: (데이터 충분한가?) → 최근 WINDOW_SIZE건만 골라서 → RMSE 계산 → 임계값보다 크면 드리프트
    """
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    if RMSE_THRESHOLD is None:
        return False  # 임계값 미확정(PENDING) — 정상으로 간주
    window = recent_predictions[-WINDOW_SIZE:]
    rmse = compute_rmse(window)
    return rmse > RMSE_THRESHOLD
