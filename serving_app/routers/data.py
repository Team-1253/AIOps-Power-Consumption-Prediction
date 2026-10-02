"""
전력 데이터 CSV 업로드·상태 조회.

업로드된 CSV는 data/uploads/에 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상
가장 최근 파일 하나를 사용한다(data/storage.py의 latest_upload()).
CSV 형식은 정제 데이터(data/hourly_clean.csv)와 같다: date_utc, time_utc, energy_kwh, humi_pct, temp_F

GET /data/sample 은 대시보드 예시 입력·배치 주입과 scripts/simulate_drift.py 가 쓸 실제 측정값을 돌려준다.
"""
import csv
import io
import math
import os
import random
import tempfile
import time

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from data.features import SEQ_LEN
from data.storage import UPLOAD_DIR, latest_upload, read_complete_rows
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.schemas import TARGET_FIELD, TIME_FIELDS, HourlyPoint

router = APIRouter(prefix="/data")

VALUE_COLUMNS = tuple(HourlyPoint.model_fields)
REQUIRED_COLUMNS = set(TIME_FIELDS) | set(VALUE_COLUMNS)
MIN_ROWS = SEQ_LEN + WINDOW_SIZE  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수 (측정값이 모두 있는 행 기준)
# 드리프트 배치: 실제 사용량에 시간마다 독립적인 곱셈 노이즈(log 표준편차)를 준다.
# 최근 업로드 마지막 구간 기준 오차율 RMSE가 정상 약 1.5% → 노이즈 적용 시 약 6~15% (임계값 5% 초과)
DRIFT_NOISE_SIGMA = 0.07


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV 파일만 업로드할 수 있습니다.")

    reader = csv.DictReader(io.StringIO(text))
    if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
        raise HTTPException(400, f"CSV에 {sorted(REQUIRED_COLUMNS)} 컬럼이 모두 있어야 합니다.")

    # 값 형식 검사와 완전한 행 수 계산은 저장 전에 임시 파일로 한다.
    with tempfile.NamedTemporaryFile("w", suffix=".csv", encoding="utf-8", delete=False) as tmp:
        tmp.write(text)
    try:
        total, rows = read_complete_rows(tmp.name, TIME_FIELDS, VALUE_COLUMNS)
    except ValueError:
        raise HTTPException(400, f"{list(VALUE_COLUMNS)} 컬럼에 숫자가 아닌 값이 있습니다.")
    finally:
        os.unlink(tmp.name)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"측정값이 모두 있는 행이 최소 {MIN_ROWS}행 이상 필요합니다.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"power_{int(time.time())}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write(text)

    return {"filename": os.path.basename(dest), "rows": total, "complete_rows": len(rows)}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}

    total, rows = read_complete_rows(path, TIME_FIELDS, VALUE_COLUMNS)
    if not rows:
        return {"exists": True, "filename": os.path.basename(path), "rows": total, "complete_rows": 0}
    values = [r[TARGET_FIELD] for r in rows]
    return {
        "exists": True,
        "filename": os.path.basename(path),
        "rows": total,
        "complete_rows": len(rows),
        "start": f"{rows[0]['date_utc']} {rows[0]['time_utc']}",
        "end": f"{rows[-1]['date_utc']} {rows[-1]['time_utc']}",
        "min_energy_kwh": min(values),
        "max_energy_kwh": max(values),
    }


@router.get("/sample")
def sample(rows: int = Query(..., ge=1, le=1000), drift: bool = False):
    """
    가장 최근 업로드 CSV의 마지막 rows행(측정값이 모두 있는 행)을 /predict 입력 형식으로 돌려준다.
    재학습(retrain_trigger)도 같은 파일의 마지막 구간을 쓰므로, 배치와 재학습 데이터가 같은 시점을 본다.
    drift=true 이면 energy_kwh 에 노이즈를 섞어 드리프트를 일으킨다 (습도·온도는 실제 값 그대로).
    """
    try:
        path = latest_upload()
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    _, complete = read_complete_rows(path, TIME_FIELDS, VALUE_COLUMNS)
    if len(complete) < rows:
        raise HTTPException(400, f"측정값이 모두 있는 행이 {len(complete)}행뿐이라 {rows}행을 줄 수 없습니다.")

    sequence = [{c: r[c] for c in VALUE_COLUMNS} for r in complete[-rows:]]
    if drift:
        for p in sequence:
            p[TARGET_FIELD] = round(p[TARGET_FIELD] * math.exp(random.gauss(0, DRIFT_NOISE_SIGMA)), 2)
    return {
        "sequence": sequence,
        "start": " ".join(complete[-rows][c] for c in TIME_FIELDS),
        "end": " ".join(complete[-1][c] for c in TIME_FIELDS),
    }
