"""
로컬 .pt 번들 / MLflow Production 모델 로드 (Lazy vs Eager 선택).

■ 이 파일이 하는 일 (한 줄 요약)
   서버가 예측에 쓸 모델을 "어디서, 언제" 불러올지 정하고, 예측 한 건을 수행합니다.
   다른 파일(predict.py, health.py)은 get_model() 만 부르면 되고, 모델이 어디서 왔는지 몰라도 됩니다.

- MODEL_SOURCE=local(기본) → serving_app/models/energy_lstm.pt 번들 로드
- MODEL_SOURCE=mlflow → Registry Production 버전 로드, 스케일러는 번들 내장분 사용
  (번들에 스케일러를 동봉해 가중치-스케일러 불일치를 원천 차단한다)

■ API(kWh) ↔ 모델(변화율) 변환은 predict_one 이 맡는다
   입력 : [{"energy_kwh":.., "humi_pct":.., "temp_F":..}, ...] seq_len+1개 (오래된 시각 → 최근 시각)
   모델 : 시점별 [temp_F, humi_pct, energy_relative_pct] seq_len개 → 다음 1시간 변화율(%)
   출력 : 다음 1시간 energy_kwh 예측값 (kWh)
   변화율 seq_len개를 만들려면 직전 값이 하나 더 필요해서 입력이 seq_len+1개다.

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

    def predict_one(self, sequence: list[dict]) -> float:
        """
        최근 seq_len+1시간 데이터로 다음 1시간 전력 사용량 1개를 예측합니다.
        받는 것  : sequence = [{"energy_kwh": 2750.0, "humi_pct": 41.0, "temp_F": 58.0}, ... 25개]  (오래된 시간 → 최근 시간)
        돌려줄 것: 다음 1시간 예상 사용량 (kWh 단위, 예: 2761.4)
        """
        # ① kWh → 직전 시간 대비 변화율(%) — data/features.py 의 _relative_series 와 같은 식
        energy = np.array([p["energy_kwh"] for p in sequence], dtype=float)
        relative = 100.0 * (energy[1:] - energy[:-1]) / energy[:-1]

        # ② 모델 입력 (1, seq_len, 3) — 시점별 [temp_F, humi_pct, energy_relative_pct]. 맨 앞 1시간은 변화율 계산에만 쓴다
        x = np.array(
            [[p["temp_F"], p["humi_pct"], r] for p, r in zip(sequence[1:], relative)], dtype=np.float32
        ).reshape(1, self.seq_len, self.n_features)
        xs = self.scaler.transform_X(x)

        # ③ 예측 → 표준화된 값을 변화율(%)로 되돌린다
        with torch.no_grad():
            out = self._model(torch.from_numpy(np.ascontiguousarray(xs)).to(DEVICE))
        pred_relative = float(self.scaler.inverse_y(out.cpu().numpy())[0])

        # ④ 변화율 → kWh: 마지막 시간 사용량에 예측 변화율을 적용한다
        return float(energy[-1] * (1 + pred_relative / 100.0))


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
    """Eager Loading: 서버가 켜질 때(main.py 의 startup) 바로 불러와 상자에 넣어 둡니다."""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def reset_cache():
    """재학습으로 Production 이 바뀌었을 때 호출 — 다음 요청에서 새 모델을 다시 불러온다."""
    global _model_cache
    _model_cache = None


def get_model() -> LoadedModel:
    """Lazy Loading: 첫 요청이 들어올 때만 불러오고, 이후에는 상자(_model_cache)에 있는 것을 재사용합니다."""
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache
