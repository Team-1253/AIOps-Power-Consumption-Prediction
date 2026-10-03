"""
드리프트 → 자동 재학습  —  serving_app/monitoring/retrain_trigger.py

■ 이 파일이 하는 일 (한 줄 요약)
   드리프트가 감지되면 최근 데이터로 모델을 조금 더 학습(fine-tuning)시키고,
   시험을 통과하면 새 모델을 Production 으로 올립니다. 이 과정을 전부 로그로 남깁니다.

■ 전체 흐름
   드리프트 감지(RMSE > RMSE_THRESHOLD) → 경고 로그 → 최근 데이터 가져오기 → fine-tuning
     → 시험 통과(RMSE ≤ RMSE_GATE)? ─ 예   → 새 버전을 Production 으로 승격 + 성공 로그
                                   └ 아니오 → 기존 Production 그대로 유지 (서비스는 멈추지 않음)

■ 로그
   재학습이 일어나면 logs/aiops.log (대시보드 "재학습 로그")에 아래 3줄이 순서대로 남습니다.
   (python scripts/simulate_drift.py 로 드리프트 배치를 보내 재현할 수 있습니다)
     [WARN] drift detected - triggering retrain
     [INFO] retrain triggered (window=last_24_sequences)
     [OK] new_rmse=1.47 - production promoted: GIGA_Energy_LSTM v2
   ※ 대시보드가 [WARN]·[INFO]·[OK] 접두어로 알람 색을 구분하니 접두어는 바꾸지 마세요.
"""
import logging

from serving_app.monitoring.drift_detector import WINDOW_SIZE, is_drift

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
    #   read_complete_rows : CSV → 행 목록 (측정 누락 행 제외)
    #   latest_upload      : 가장 최근 업로드한 CSV 경로
    #   SEQ_LEN            : 모델 입력 창문 길이 (data/features.py)
    #   fine_tune          : Production 모델을 "이어받아" 짧게 추가 학습 (warm start)
    from data.features import SEQ_LEN
    from data.storage import latest_upload, read_complete_rows
    from serving_app.train_and_register import MODEL_NAME, fine_tune
    from serving_app import model_loader

    logger.info(f"[INFO] retrain triggered (window=last_{WINDOW_SIZE}_sequences)")

    # 재학습에 쓸 "최근 데이터": 드리프트 판정 윈도우와 같은 WINDOW_SIZE개 시퀀스(정답)가 나오는 구간
    #   WINDOW_SIZE + SEQ_LEN 행 → fine_tune(tensors_from_rows)이 WINDOW_SIZE개 시퀀스를 만든다
    _, upload_rows = read_complete_rows(
        latest_upload(), ("date_utc", "time_utc"), ("energy_kwh", "humi_pct", "temp_F")
    )
    rows = upload_rows[-(WINDOW_SIZE + SEQ_LEN):]

    result = fine_tune(rows)

    # 새 모델이 "실제로 Production 이 되었을 때만" 캐시를 비우고 성공 로그를 남긴다
    if result["promoted"]:
        model_loader.reset_cache()
        logger.info(
            f"[OK] new_rmse={result['rmse']:.2f}% - production promoted: {MODEL_NAME} v{result['version']}"
        )
        return {"status": "retrain_triggered", "promoted": True, "rmse": result["rmse"]}
    return {"status": "retrain_triggered", "promoted": False, "rmse": result["rmse"]}
