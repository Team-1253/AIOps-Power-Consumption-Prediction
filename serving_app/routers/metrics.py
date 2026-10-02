"""
GET /metrics/summary  예측 요청 전체 기간 요약 (요청 수, 평균 응답시간, 오류율).

집계 원본은 monitoring/logger.py 미들웨어가 기록하는 logs/requests.log다.
"""
from fastapi import APIRouter

from serving_app.monitoring import logger as request_logger

router = APIRouter(prefix="/metrics")


@router.get("/summary")
def summary():
    return request_logger.summarize(request_logger.read_requests())
