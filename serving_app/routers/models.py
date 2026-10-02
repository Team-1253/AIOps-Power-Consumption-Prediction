"""
MLflow Model Registry 조회 API (현재 운영 모델, 재학습 이력).

GET /models/current   현재 서빙 중인 모델 정보
GET /models/history   등록 버전 이력 (최신 버전부터)

train_and_register.py는 승격할 때 이전 Production 버전을 보관(Archived) 처리하지 않으므로,
Production 버전이 여러 개 남는다. model_loader가 "models:/<이름>/Production"으로 불러오는 것은
그중 가장 최근 버전이므로, 화면도 가장 최근 Production 버전을 운영 모델로 표시한다.
MLflow 조회 실패(레지스트리 없음 등) 시 빈 결과를 반환한다.
"""
import logging
import os

from fastapi import APIRouter, Query

from serving_app.routers.system import _train_constants

router = APIRouter(prefix="/models")
_log = logging.getLogger("uvicorn.error")


def _client():
    from mlflow.tracking import MlflowClient  # mlflow import가 무거워 호출 시점에 불러온다

    return MlflowClient()


def _versions(client, model_name: str) -> list:
    versions = client.search_model_versions(f"name='{model_name}'")
    return sorted(versions, key=lambda v: int(v.version), reverse=True)


def _serving_version(versions: list) -> str | None:
    return next((v.version for v in versions if v.current_stage == "Production"), None)


def _run_info(client, run_id: str | None) -> tuple[str | None, float | None]:
    if not run_id:
        return None, None
    try:
        run = client.get_run(run_id)
    except Exception:
        return None, None
    return run.data.params.get("mode"), run.data.metrics.get("rmse")


def _describe(client, v, serving: str | None, model_name: str) -> dict:
    mode, rmse = _run_info(client, v.run_id)
    return {
        "name": model_name,
        "version": v.version,
        "stage": v.current_stage,
        "serving": v.version == serving,
        "mode": mode,
        "rmse": rmse,
        "created_at": v.creation_timestamp / 1000 if v.creation_timestamp else None,
    }


@router.get("/current")
def current():
    c = _train_constants()
    source = os.getenv("MODEL_SOURCE", "local")
    if source != "mlflow":
        return {"source": "local", "name": None, "version": "v1-local", "rmse_gate": c.get("RMSE_GATE")}

    model_name = c.get("MODEL_NAME")
    try:
        client = _client()
        versions = _versions(client, model_name)
    except Exception as e:
        _log.warning("MLflow registry 조회 실패: %s", e)
        versions = []

    serving = _serving_version(versions)
    if serving is None:
        return {"source": "mlflow", "name": model_name, "version": None, "rmse_gate": c.get("RMSE_GATE")}

    v = next(v for v in versions if v.version == serving)
    return {"source": "mlflow", "rmse_gate": c.get("RMSE_GATE"), **_describe(client, v, serving, model_name)}


@router.get("/history")
def history(limit: int = Query(12, ge=1, le=100)):
    model_name = _train_constants().get("MODEL_NAME")
    try:
        client = _client()
        versions = _versions(client, model_name)
    except Exception as e:
        _log.warning("MLflow registry 조회 실패: %s", e)
        return []

    serving = _serving_version(versions)
    return [_describe(client, v, serving, model_name) for v in versions[:limit]]
