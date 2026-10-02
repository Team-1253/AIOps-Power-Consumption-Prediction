"""
로컬 .pt 번들 / MLflow Production 모델 로드 (Lazy vs Eager 선택).

템플릿의 Keras 로더를 PyTorch 규격으로 교체. 동작 계약은 동일하다:
- MODEL_SOURCE=local(기본) → serving_app/models/energy_lstm.pt 번들 로드
- MODEL_SOURCE=mlflow → Registry Production 버전 로드, 스케일러는 번들 내장분 사용
  (템플릿은 scaler.pkl 별도 파일이었으나, torch판은 번들에 스케일러를 동봉해
  가중치-스케일러 불일치를 원천 차단한다)

입력 형식: [{"energy_relative_pct":.., "humi_pct":.., "temp_F":..}, ...] SEQ_LEN개
(오래된 시각 → 최근 시각), raw 스케일. 서빙 요청 dict → 이 형식 변환은
schemas.py / routers (서빙 담당) 영역.
출력: 모델 raw 스칼라 1개 (변화율. kWh 역변환은 서빙팀 영역).

환경변수
    LOADING_MODE = lazy(기본값) | eager
    MODEL_SOURCE = local(기본값) | mlflow
    MLFLOW_TRACKING_URI = MODEL_SOURCE=mlflow 일 때 필요
"""

import os
import time

import numpy as np
import torch

from serving_app.lstm_model import N_FEATURES, SEQ_LEN, build_model, get_device

LOCAL_MODEL_PATH = "serving_app/models/energy_lstm.pt"
MLFLOW_MODEL_URI = "models:/GIGA_Energy_LSTM/Production"

DEVICE = get_device()
_model_cache = None  # Lazy Loading 캐시


class LoadedModel:
    """local .pt 번들과 mlflow 두 소스를 동일한 인터페이스로 감싸는 래퍼."""

    def __init__(self, model, scaler, version: str, seq_len: int = SEQ_LEN, n_features: int = N_FEATURES):
        self._model = model.to(DEVICE).eval()
        self.scaler = scaler
        self.version = version
        self.seq_len = seq_len
        self.n_features = n_features

    def predict_one(self, sequence) -> float:
        """
        sequence: 서빙이 넘기는 SEQ_LEN개 행.dict 형식
            [{"energy_relative_pct":.., "humi_pct":.., "temp_F":..}, ...] 또는
            [[temp, humi, energy], ...] — 오래된 시각 -> 최근 시각 순서.
            (seq_len은 번들 input_shape 기준. 윈도우 변경 시 번들 재생성만으로 대응)
        반환: 모델 raw 출력 1개 (변화율. kWh 역변환은 서빙팀 영역).
        """
        triples = []
        for p in sequence:
            if isinstance(p, dict):
                triples.append([p["temp_F"], p["humi_pct"], p["energy_relative_pct"]])
            else:
                triples.append(list(p))
        x = np.array(triples, dtype=np.float32).reshape(1, self.seq_len, self.n_features)
        xs = self.scaler.transform_X(x)
        with torch.no_grad():
            out = self._model(
                torch.from_numpy(np.ascontiguousarray(xs)).to(DEVICE)
            )
        return float(out.cpu().numpy().ravel()[0])


class _BundleScaler:
    """노트북 번들(feature_scaler/target_scaler 2개)을 학습 파이프라인의
    단일 스케일러 인터페이스(transform_X/transform_y/inverse_y)로 감싼 어댑터.
    데이터셋 담당의 정식 스케일러가 오면 교체한다."""

    def __init__(self, feature_scaler, target_scaler):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    def transform_X(self, a):
        shape = a.shape
        return self.feature_scaler.transform(a.reshape(-1, 3)).reshape(shape).astype(a.dtype)

    def transform_y(self, y):
        import numpy as _np

        return self.target_scaler.transform(_np.asarray(y).reshape(-1, 1)).ravel()

    def inverse_y(self, ys):
        import numpy as _np

        return self.target_scaler.inverse_transform(_np.asarray(ys).reshape(-1, 1)).ravel()


def _scaler_from_bundle(bundle):
    if "scaler" in bundle:
        return bundle["scaler"]
    return _BundleScaler(bundle["feature_scaler"], bundle["target_scaler"])


def _load_bundle(path: str, version: str) -> LoadedModel:
    bundle = torch.load(path, map_location=DEVICE, weights_only=False)
    seq_len, n_features = bundle["input_shape"]
    model = build_model(
        input_size=n_features,
        device=DEVICE,
    )
    model.load_state_dict(bundle["model_state_dict"])
    return LoadedModel(
        model=model, scaler=_scaler_from_bundle(bundle), version=version,
        seq_len=seq_len, n_features=n_features,
    )


def _load_from_local() -> LoadedModel:
    return _load_bundle(LOCAL_MODEL_PATH, version="v1-local")


def _load_from_mlflow() -> LoadedModel:
    import mlflow.pytorch as mlflow_pytorch

    model = mlflow_pytorch.load_model(MLFLOW_MODEL_URI)
    # 스케일러는 Registry가 아니라 로컬 번들에서 (가중치와 한 쌍으로 관리)
    bundle = torch.load(LOCAL_MODEL_PATH, map_location=DEVICE, weights_only=False)
    return LoadedModel(model=model, scaler=_scaler_from_bundle(bundle), version="production")


def _load_model() -> LoadedModel:
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


def load_eager() -> LoadedModel:
    """Eager Loading: 서버 시작 시점에 즉시 모델을 로드한다."""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 로드하고, 이후에는 캐시를 재사용한다."""
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache
