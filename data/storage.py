"""
업로드된 전력 데이터 파일 관리.

대시보드에서 올린 CSV는 data/uploads/에 타임스탬프가 붙은 이름으로 계속 쌓이고
(과거 파일을 덮어쓰지 않는다), 학습(train_and_register.py 등)은 항상 가장 최근 파일 하나를 사용한다.
"""
import csv
import glob
import os

UPLOAD_DIR = "data/uploads"


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    """data/uploads/ 에 쌓인 CSV 중 가장 최근에 업로드된 파일의 경로를 반환한다."""
    files = sorted(glob.glob(os.path.join(upload_dir, "*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 전력 데이터가 없습니다. 대시보드에서 CSV 파일을 먼저 업로드하세요 "
            f"(data/hourly_clean.csv를 업로드할 수 있습니다 -> {upload_dir}/)."
        )
    return files[-1]


def read_complete_rows(path: str, time_columns: tuple, value_columns: tuple) -> tuple[int, list[dict]]:
    """
    CSV를 읽어 (전체 행 수, 값이 모두 있는 행 목록)을 반환한다. 대시보드 통계·업로드 검증용.
    값 컬럼은 float로 바꾸고, 하나라도 비어 있는 행(측정 누락)은 제외한다.
    """
    total = 0
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            total += 1
            if any(not (r.get(c) or "").strip() for c in value_columns):
                continue
            row = {c: r.get(c, "") for c in time_columns}
            row.update({c: float(r[c]) for c in value_columns})
            rows.append(row)
    return total, rows
