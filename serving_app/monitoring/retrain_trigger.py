"""
[Day3] 드리프트 → 자동 재학습  —  serving_app/monitoring/retrain_trigger.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울 곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   드리프트가 감지되면 최근 데이터로 모델을 조금 더 학습(fine-tuning)시키고,
   시험을 통과하면 새 모델을 Production 으로 올립니다. 이 과정을 전부 로그로 남깁니다.

■ 전체 흐름
   드리프트 감지(RMSE > $4) → 경고 로그 → 최근 데이터 가져오기 → fine-tuning
     → 시험 통과(RMSE ≤ $4)? ─ 예   → 새 버전을 Production 으로 승격 + 성공 로그
                             └ 아니오 → 기존 Production 그대로 유지 (서비스는 멈추지 않음)

■ 확인 방법
   python scripts/simulate_drift.py 실행 후, logs/aiops.log (또는 대시보드 "재학습 로그")에
   아래 3줄이 순서대로 찍히면 성공입니다.
     [WARN] drift detected - triggering retrain
     [INFO] retrain triggered (window=last_21_days)
     [OK] new_rmse=1.47 - production promoted: HAIC_Predictor v2
   ※ 로그 문장은 대시보드가 읽으니 글자를 바꾸지 마세요.

■ 이 파일의 빈칸 : [빈칸 9] 데이터 범위   [빈칸 10] 학습 방식   [빈칸 11] 승격 여부
"""
import logging

from serving_app.monitoring.drift_detector import is_drift

# "aiops" 이름의 기록장. main.py 가 이 기록장을 logs/aiops.log 파일에 연결해 두었습니다.
logger = logging.getLogger("aiops")


def check_and_trigger(recent_predictions: list[dict]) -> dict:
    """
    받는 것  : 최근 예측 기록 [{"predicted": ..., "actual": ...}, ...]  (predict.py 가 넘겨줌)
    돌려줄 것:
      드리프트 없음 → {"status": "ok"}
      재학습 함     → {"status": "retrain_triggered", "promoted": True/False, "rmse": 1.47}
    """
    if not is_drift(recent_predictions):
        return {"status": "ok"}

    logger.warning("[WARN] drift detected - triggering retrain")

    # 함수 안에서 import 하는 이유: 파일끼리 서로를 import 하다 꼬이는 문제(순환 import)를 피하려고
    #   load_rows          : CSV → 행 목록
    #   latest_upload      : 가장 최근 업로드한 CSV 경로
    #   SEQ_LEN            : 창문 길이 (20)
    #   train_and_register : Day2 — 새 모델을 "처음부터" 학습 (scratch)
    #   fine_tune          : Day3 — Production 모델을 "이어받아" 짧게 추가 학습 (warm start)
    from data.features import load_rows, SEQ_LEN
    from data.storage import latest_upload
    from serving_app.train_and_register import fine_tune

    logger.info("[INFO] retrain triggered (window=last_21_days)")

    # ════════════════════════════ [빈칸 9] ════════════════════════════
    # 재학습에 쓸 "최근 데이터"만 잘라 오세요.  목표: 최근 21거래일(한 달)의 정답으로 학습
    #
    #   생각해 볼 질문
    #     · 21일치 정답으로 학습하려면, 문제(창문 20일)까지 포함해 몇 행이 필요할까요?
    #       (predict.py [빈칸 6]의 그림: 가격 41개 → 예측 21번)
    #     · 딱 21행만 자르면 build_sequences 가 만들 수 있는 문제는 몇 개일까요?
    #   형태 : 리스트[-(N):] 은 "뒤에서 N개"입니다. ___ 에 N 을 계산식으로 쓰세요. (숫자 41 대신 21 과 상수 이름으로)
    rows = load_rows(latest_upload())[-(21 + SEQ_LEN):]

    # ════════════════════════════ [빈칸 10] ════════════════════════════
    # 41행으로 재학습을 실행하세요.  (위 import 설명의 두 함수 중 하나)
    #
    #   생각해 볼 질문
    #     · train_and_register(rows=rows) 도 이 rows 로 실행은 됩니다. 41행으로 LSTM(파라미터 약 1.6만 개)을
    #       처음부터 학습하면 어떤 모델이 나올까요?
    #     · 3년치로 이미 잘 학습된 Production 모델을 활용하는 방법은 없을까요?
    #   결과 : {"run_id": "...", "rmse": 1.47, "promoted": True, "version": "2"}  (떨어지면 "version" 없음)
    result = fine_tune(rows)

    # ════════════════════════════ [빈칸 11] ════════════════════════════
    # 새 모델이 "실제로 Production 이 되었을 때만" 성공 로그를 남기도록 조건을 채우세요.
    #
    #   생각해 볼 질문
    #     · 재학습이 "실행됐다"와 "새 모델이 배포됐다"는 같은 뜻일까요?
    #     · 시험(RMSE ≤ $4)에 떨어진 새 모델은 어떻게 되고, 서비스는 어떤 모델이 계속 맡나요?
    #       (train_and_register.py 의 _register_if_gate_passed 참고)
    if result["promoted"]:
        logger.info(
            f"[OK] new_rmse={result['rmse']:.2f} - production promoted: HAIC_Predictor v{result['version']}"
        )
        return {"status": "retrain_triggered", "promoted": True, "rmse": result["rmse"]}
    return {"status": "retrain_triggered", "promoted": False, "rmse": result["rmse"]}
