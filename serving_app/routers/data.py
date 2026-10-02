"""
전력 데이터 CSV 업로드·상태 조회.

업로드된 CSV는 data/uploads/에 쌓이고, 학습(train_and_register.py, fine_tune 등)은 항상
가장 최근 파일 하나를 사용한다(data/storage.py의 latest_upload()).
CSV 형식은 정제 데이터(data/hourly_clean.csv)와 같다: date_utc, time_utc, energy_kwh, humi_pct, temp_F
"""
import csv
import io
import os
import tempfile
import time

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN
from data.storage import UPLOAD_DIR, latest_upload, read_complete_rows
from serving_app.monitoring.drift_detector import WINDOW_SIZE
from serving_app.schemas import TARGET_FIELD, TIME_FIELDS, HourlyPoint

router = APIRouter(prefix="/data")

VALUE_COLUMNS = tuple(HourlyPoint.model_fields)
REQUIRED_COLUMNS = set(TIME_FIELDS) | set(VALUE_COLUMNS)
MIN_ROWS = SEQ_LEN + WINDOW_SIZE  # 시퀀스 구성 + 드리프트 판정 윈도우에 필요한 최소 행 수 (측정값이 모두 있는 행 기준)


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
