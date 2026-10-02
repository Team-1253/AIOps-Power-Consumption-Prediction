"""
대시보드용 시스템 조회 API. 기존 파이프라인 동작은 바꾸지 않고, 설정값·로그·업로드 데이터를 읽어서 보여주기만 한다.

GET /system/info        서버 설정값 (입력 길이, 게이트, 판정 윈도우, 임계값, epoch, 학습률, 환경 변수)
GET /system/timeseries  requests.log를 시간 구간별로 집계 (운영 지표 KPI)
GET /system/alerts      aiops.log의 [WARN]/[INFO]/[OK] 기록 (최근 알람)
GET /system/dataset     가장 최근 업로드 CSV 통계

입력 피처·예측 대상·시간 컬럼은 schemas.py(HourlyPoint, TARGET_FIELD, TIME_FIELDS)를 따른다.
필드명은 CSV 컬럼명과 같다.
"""
import ast
import os
import re
import time
from datetime import datetime

from fastapi import APIRouter, Query

from data.storage import latest_upload, read_complete_rows
from serving_app.monitoring import logger as request_logger
from serving_app.monitoring.drift_detector import RMSE_THRESHOLD, WINDOW_SIZE
from serving_app.schemas import INPUT_LEN, TARGET_FIELD, TIME_FIELDS, HourlyPoint

router = APIRouter(prefix="/system")

AIOPS_LOG = os.path.join("logs", "aiops.log")
TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train_and_register.py")
PEAK_THRESHOLD = None  # 피크 경보 기준값. 정해지면 숫자로 바꾼다 (None이면 화면에서 경보 표시 안 함)

_ALERT_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[(\w+)\] (.*)$")


def _train_constants() -> dict:
    """train_and_register.py의 학습 상수를 읽는다.

    모듈을 import하면 torch·mlflow가 같이 올라와 조회 API가 느려지므로,
    소스 파일의 최상위 상수 대입문만 파싱한다.
    """
    names = {"RMSE_GATE", "MODEL_NAME", "BASE_EPOCHS", "FINE_TUNE_EPOCHS", "FINE_TUNE_LR"}
    with open(TRAIN_SCRIPT, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                found[name] = ast.literal_eval(node.value)
    return found


FEATURES = list(HourlyPoint.model_fields)


def _load_latest() -> tuple[str, int, list[dict]] | None:
    """가장 최근 업로드 CSV의 (경로, 전체 행 수, 측정값이 모두 있는 행). 없으면 None."""
    try:
        path = latest_upload()
    except FileNotFoundError:
        return None
    total, rows = read_complete_rows(path, TIME_FIELDS, tuple(FEATURES))
    return (path, total, rows) if rows else None


def _row_time(row: dict) -> str:
    return " ".join(row[c] for c in TIME_FIELDS)


@router.get("/info")
def info():
    c = _train_constants()
    return {
        "seq_len": INPUT_LEN,  # 대시보드가 /predict 입력·배치 길이로 쓰는 값 (API 입력 길이)
        "features": FEATURES,
        "target": TARGET_FIELD,
        "rmse_gate": c.get("RMSE_GATE"),
        "drift_window_size": WINDOW_SIZE,
        "drift_rmse_threshold": RMSE_THRESHOLD,
        "peak_threshold": PEAK_THRESHOLD,
        "model_name": c.get("MODEL_NAME"),
        "base_epochs": c.get("BASE_EPOCHS"),
        "fine_tune_epochs": c.get("FINE_TUNE_EPOCHS"),
        "fine_tune_lr": c.get("FINE_TUNE_LR"),
        "model_source": os.getenv("MODEL_SOURCE", "local"),
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
    }


@router.get("/timeseries")
def timeseries(
    window_sec: int = Query(300, ge=60, le=7 * 86400),
    buckets: int = Query(12, ge=1, le=120),
):
    """최근 window_sec초를 buckets개 구간으로 나눠 구간별 요청 수·평균 응답시간·오류율을 반환한다 (오래된 구간 → 최근 구간)."""
    now = time.time()
    start = now - window_sec
    size = window_sec / buckets
    groups = [[] for _ in range(buckets)]
    for r in request_logger.read_requests(since=start):
        idx = min(int((r["ts"] - start) / size), buckets - 1)
        groups[idx].append(r)

    return [
        {
            "start_ts": round(start + i * size, 3),
            "end_ts": round(start + (i + 1) * size, 3),
            **request_logger.summarize(g),
        }
        for i, g in enumerate(groups)
    ]


@router.get("/alerts")
def alerts(limit: int = Query(20, ge=1, le=200)):
    """aiops.log를 최신순으로 반환한다. 여러 줄짜리 기록(traceback 등)은 첫 줄만 사용한다."""
    if not os.path.isfile(AIOPS_LOG):
        return []
    items = []
    with open(AIOPS_LOG, encoding="utf-8") as f:
        for line in f:
            m = _ALERT_LINE.match(line.rstrip("\n"))
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            items.append({"ts": ts, "level": m.group(2), "message": m.group(3)})
    return items[::-1][:limit]


@router.get("/dataset")
def dataset():
    latest = _load_latest()
    if latest is None:
        return {"exists": False}
    path, total, rows = latest

    columns = {}
    for col in FEATURES:
        values = [r[col] for r in rows]
        columns[col] = {
            "min": round(min(values), 3),
            "max": round(max(values), 3),
            "mean": round(sum(values) / len(values), 3),
        }

    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": total,
        "complete_rows": len(rows),
        "start_date": _row_time(rows[0]),
        "end_date": _row_time(rows[-1]),
        "target": TARGET_FIELD,
        "columns": columns,
    }
