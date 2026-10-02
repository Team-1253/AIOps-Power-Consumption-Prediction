"""
대시보드의 "재학습 로그" 패널용 - MLflow Registry를 조회하는 별도 이력 API 대신,
monitoring/retrain_trigger.py의 "aiops" 로거가 그대로 기록하는 logs/aiops.log
파일을 읽기 전용으로 노출한다. 새 학습/승격 로직은 없다 (logging 설정은
serving_app/main.py에서 앱 시작 시 한 번만 구성한다).

드리프트 감지("[WARN] drift detected") -> 재학습 트리거("[INFO] retrain triggered") ->
게이트 통과("[OK] new_rmse=...")가 이 파일에 순서대로 쌓인다.
"""
import os

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/logs")

LOG_DIR = "logs"


@router.get("")
def list_logs():
    if not os.path.isdir(LOG_DIR):
        return []
    files = []
    for name in sorted(os.listdir(LOG_DIR)):
        path = os.path.join(LOG_DIR, name)
        if os.path.isfile(path):
            files.append({"name": name, "size": os.path.getsize(path)})
    return files


@router.get("/{filename}")
def read_log(filename: str):
    # 경로 조작(디렉토리 탈출) 방지: 순수 파일명만 허용
    if filename != os.path.basename(filename):
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다")

    path = os.path.join(LOG_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="로그 파일을 찾을 수 없습니다")

    with open(path, encoding="utf-8") as f:
        content = f.read()
    return {"name": filename, "content": content}
