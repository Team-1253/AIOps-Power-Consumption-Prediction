"""
요청 로그 수집 미들웨어.

예측 요청(/predict, /predict/batch-test)마다 경로·상태코드·응답시간을 logs/requests.log에
JSON 한 줄씩 기록한다. 대시보드가 15초마다 호출하는 /system/* · /models/* 조회 요청까지
세면 요청 수가 부풀려지므로, 서비스 요청인 /predict 경로만 기록한다.

/system/timeseries, /metrics/summary 가 이 파일을 읽어 요청 수·평균 응답시간·오류율을 집계한다.
"""
import json
import os
import time

from fastapi import Request

LOG_DIR = "logs"
REQUEST_LOG = os.path.join(LOG_DIR, "requests.log")
TRACKED_PREFIX = "/predict"


def _write(record: dict) -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(REQUEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def log_requests(request: Request, call_next):
    path = request.url.path
    if not path.startswith(TRACKED_PREFIX):
        return await call_next(request)

    start = time.time()
    status = 500  # call_next에서 예외가 나면 500으로 기록하고 예외는 그대로 올린다
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        _write({
            "ts": round(start, 3),
            "method": request.method,
            "path": path,
            "status": status,
            "latency_ms": round((time.time() - start) * 1000, 1),
        })


def read_requests(since: float | None = None) -> list[dict]:
    """requests.log의 기록을 시간순으로 반환한다. since(epoch 초)를 주면 그 이후 기록만."""
    if not os.path.isfile(REQUEST_LOG):
        return []
    records = []
    with open(REQUEST_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since is None or r["ts"] >= since:
                records.append(r)
    return records


def summarize(records: list[dict]) -> dict:
    """요청 수, 평균 응답시간(ms), 오류율(0~1, 상태코드 400 이상 비율)."""
    n = len(records)
    if n == 0:
        return {"request_count": 0, "avg_latency_ms": 0.0, "error_rate": 0.0}
    errors = sum(1 for r in records if r["status"] >= 400)
    return {
        "request_count": n,
        "avg_latency_ms": round(sum(r["latency_ms"] for r in records) / n, 1),
        "error_rate": round(errors / n, 4),
    }
