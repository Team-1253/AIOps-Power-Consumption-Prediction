"""
[Day3] 드리프트 감지  —  serving_app/monitoring/drift_detector.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울 곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   "최근 모델이 평균 몇 달러씩 틀리고 있는지(RMSE)"를 계산해서,
   4달러보다 많이 틀리면 "데이터가 달라졌다(드리프트)"고 판단합니다.

■ 드리프트가 뭔가요?
   모델은 과거 데이터로 공부했습니다. 그런데 시장 상황이 갑자기 바뀌면(예: 변동성 폭증)
   공부한 것과 다른 데이터가 들어와 예측이 크게 빗나가기 시작합니다. 이것이 드리프트입니다.
   그래서 최근 예측이 얼마나 틀렸는지 계속 지켜보다가, 너무 많이 틀리면 재학습을 시작합니다.

■ 판단 기준
   최근 21건(약 한 달 거래일)의 RMSE > $4.00  →  드리프트!
   · 21건보다 짧으면 : 우연한 한두 번 실수에도 경보가 울립니다.
   · 21건보다 길면   : 상황이 바뀌어도 늦게 알아챕니다.

■ 이 파일의 빈칸 : [빈칸 7] compute_rmse   [빈칸 8] is_drift
"""
RMSE_THRESHOLD = 4.00  # 이보다 많이 틀리면 드리프트
WINDOW_SIZE = 21       # 최근 21건을 봅니다


def compute_rmse(recent_predictions: list[dict]) -> float:
    """
    받는 것  : [{"predicted": 100.0, "actual": 102.0}, {"predicted": 100.0, "actual": 98.0}, ...]
    돌려줄 것: RMSE (숫자 1개, "평균 몇 달러 틀렸나").  빈 목록이면 0.0

    ■ RMSE 계산 4단계 — 먼저 손으로 풀어 보세요
                             1건째            2건째
       ① 오차 (실제-예측)    102-100 = +2     98-100 = -2
       ② 제곱               2² = 4           (-2)² = 4
       ③ 평균               (4 + 4) / 2 = 4
       ④ 제곱근             √4 = 2.0         → "평균 2달러 틀렸다"

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

    # ════════════════════════════ [빈칸 7] ════════════════════════════
    # 위 4단계 중 ① 오차 와 ③ 평균 을 채우세요. (② 제곱 ** 2 와 ④ 제곱근 math.sqrt 는 이미 적혀 있습니다)
    #   · 첫 줄 ___ : 한 건(p)의 오차 = 실제 - 예측.   p 는 {"predicted": ..., "actual": ...}
    #   · 둘째 줄 ___ : errors_sq 의 평균 = 합계 ÷ 개수   (sum(목록), len(목록))
    #
    #   생각해 볼 질문
    #     · 이미 적힌 ** 2 를 빼고 오차를 그냥 평균 내면, 위 예시의 결과는 몇이 되나요? 그게 맞는 판단일까요?
    #     · 이미 적힌 math.sqrt 를 빼면 단위가 "달러"일까요, "달러²"일까요? 기준 $4.00 과 비교할 수 있을까요?
    errors_sq = [(p["actual"] - p["predicted"]) ** 2 for p in recent_predictions]
    return math.sqrt(sum(errors_sq) / len(errors_sq))


def is_drift(recent_predictions: list[dict]) -> bool:
    """
    드리프트인지 True/False 로 판단합니다.
    흐름: (데이터 충분한가?) → 최근 21건만 골라서 → RMSE 계산 → 기준($4)보다 크면 드리프트
    """
    # ════════════════════════════ [빈칸 8] ════════════════════════════
    # "아직 판단하지 않는다(False)"로 끝내야 하는 조건을 채우세요.
    #
    #   생각해 볼 질문
    #     · 서버를 켜고 처음 3건만 들어왔는데 그중 1건이 크게 빗나갔다면, 재학습을 돌려야 할까요?
    #     · 판단에 필요한 최소 건수는 위에 어떤 이름의 상수로 정해져 있나요?
    if len(recent_predictions) < WINDOW_SIZE:
        return False  # 아직 판단할 만큼 데이터가 쌓이지 않음
    window = recent_predictions[-WINDOW_SIZE:]
    rmse = compute_rmse(window)
    return rmse > RMSE_THRESHOLD
