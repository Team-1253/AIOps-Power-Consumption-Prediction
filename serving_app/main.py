"""
FastAPI 앱 진입점.

Day1: app 생성, 라우터(predict, health) 등록, startup 이벤트에서 로딩 모드에 따라 모델 준비
Day2: data 라우터 등록 (전력 데이터 CSV 업로드)
Day3: "aiops" 로거를 logs/aiops.log 파일로 연결(로깅 설정) + logs 라우터(로그 파일 조회) 등록
대시보드: 요청 로그 미들웨어(logs/requests.log) + system · models · metrics 조회 라우터 등록

정적 대시보드: serving_app/static/index.html 이 /health · /predict · /predict/batch-test ·
/data/upload · /logs · /system/* · /models/* · /metrics/summary 를 호출하는 운영 화면입니다.
API 라우터를 먼저 등록한 뒤 StaticFiles를 "/"에 마지막으로 mount해야, /predict 같은 API 경로가 정적 파일보다
먼저 매칭됩니다(Starlette는 등록 순서대로 라우트를 검사합니다).
"""
import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader
from serving_app.monitoring.logger import log_requests
from serving_app.routers import data, health, logs, metrics, models, predict, system

# monitoring/retrain_trigger.py가 쓰는 "aiops" 로거를 logs/aiops.log 파일에 연결한다.
# (routers/logs.py가 같은 디렉토리를 읽기 전용으로 노출한다.) 여기서 이 로거 하나만
# 직접 설정하므로, uvicorn 자체 로깅 설정과 충돌하지 않는다.
_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_aiops_logger = logging.getLogger("aiops")
_aiops_logger.setLevel(logging.INFO)
if not _aiops_logger.handlers:
    _handler = logging.FileHandler(os.path.join(_LOG_DIR, "aiops.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _aiops_logger.addHandler(_handler)
    _aiops_logger.addHandler(logging.StreamHandler())  # 터미널에서도 동일하게 확인 가능

app = FastAPI(title="AI Datacenter Power Forecast Serving & AIOps")
app.middleware("http")(log_requests)  # 예측 요청을 logs/requests.log에 기록

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(data.router)  # 전력 데이터 CSV 업로드
app.include_router(logs.router)  # 대시보드: 재학습 로그 파일 조회
app.include_router(system.router)  # 대시보드: 설정값 · 운영 지표 · 알람 · 데이터셋
app.include_router(models.router)  # 대시보드: 현재 운영 모델 · 재학습 이력
app.include_router(metrics.router)  # 대시보드: 요청 지표 요약

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")  # 대시보드 UI


@app.on_event("startup")
def startup():
    # Day1 실습 포인트: LOADING_MODE=eager 로 켜고 서버 시작 시간을 lazy와 비교해보세요.
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        model_loader.load_eager()
    else:
        print("[lazy] 모델은 첫 /predict 요청이 들어올 때 로드됩니다.")
