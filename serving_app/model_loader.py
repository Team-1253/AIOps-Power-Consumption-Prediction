"""
[Day1 → Day2] 모델 불러오기  —  serving_app/model_loader.py
【실습용】 ___ (밑줄 3개)만 채우세요. 채울 곳은 [빈칸 N] 으로 표시되어 있습니다.
   ___ 가 남은 채 실행하면 "name '___' is not defined" 에러가 나며, 그 줄이 채울 곳입니다.

■ 이 파일이 하는 일 (한 줄 요약)
   서버가 예측에 쓸 모델을 "어디서, 언제" 불러올지 정하고, 예측 한 건을 수행합니다.
   다른 파일(predict.py, health.py)은 get_model() 만 부르면 되고, 모델이 어디서 왔는지 몰라도 됩니다.

■ 핵심 개념
   1) 학습 때와 똑같이 전처리해야 한다
        모델은 0~1 값으로 학습했습니다. 서빙할 때도 입력을 0~1로 바꾸고, 출력은 달러로 되돌려야 합니다.
   2) Lazy vs Eager
        Eager : 서버가 켜질 때 모델을 바로 불러옴 → 서버 시작은 느리지만 첫 요청이 빠름
        Lazy  : 첫 /predict 요청이 올 때 불러옴  → 서버 시작은 빠르지만 첫 요청이 느림
   3) 모델은 바뀌어도 스케일러는 안 바뀐다
        MODEL_SOURCE 가 local 이든 mlflow 든, 스케일러는 항상 로컬 scaler.pkl 을 씁니다.

■ 환경변수
   LOADING_MODE = lazy(기본) | eager
   MODEL_SOURCE = local(기본, Day1) | mlflow(Day2~)

■ 이 파일의 빈칸
   [빈칸 2] [빈칸 3]  Day1 : predict_one() — 입력 변환 / 출력 복원
   [빈칸 4]           Day1 : get_model()   — Lazy Loading
   [빈칸 5]           Day2 : _load_from_mlflow() — MLflow 에서 불러오기

■ self.scaler(HAICScaler)가 가진 도구 — [빈칸 2]·[빈칸 3]은 이 중에서 고르는 문제입니다
   transform_point(종가, 거래량) → [0~1, 0~1]   하루치 입력 2개를 0~1로
   scale_close(종가)             → 0~1          종가 하나를 0~1로 (학습 정답용)
   inverse_close(0~1 값)         → 종가(달러)    0~1 값을 달러로 되돌리기
"""
import os
import time

from data.features import HAICScaler

LOCAL_MODEL_PATH = "serving_app/models/haic_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
MLFLOW_MODEL_URI = "models:/HAIC_Predictor/Production"  # "models:/<모델 이름>/<단계>" 형식

_model_cache = None  # 한 번 불러온 모델을 담아 두는 상자 (처음엔 비어 있음 = None)


class LoadedModel:
    """
    모델 + 스케일러 + 버전을 한 묶음으로 포장한 상자.
    local 모델이든 MLflow 모델이든 이 상자에 담으면 똑같은 방법(predict_one)으로 쓸 수 있습니다.
    """

    def __init__(self, keras_model, scaler: HAICScaler, version: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version

    def predict_one(self, sequence: list[dict]) -> float:
        """
        20일치 데이터로 다음날 종가 1개를 예측합니다.
        받는 것  : sequence = [{"close": 160.0, "volume": 1200000}, ... 20개]  (오래된 날 → 최근 날)
        돌려줄 것: 다음날 예상 종가 (달러 단위, 예: 161.37)

        흐름:  [달러 값 20개] → ① 0~1로 변환 → ② 입력 모양 맞추기 → ③ 예측(0~1) → ④ 달러로 복원
        확인:  /predict 응답의 predicted_close 가 입력 종가와 비슷한 "달러" 값이면 성공
        """
        import numpy as np

        # ════════════════════════ [빈칸 2]  ① 0~1로 변환 ════════════════════════
        # 하루치(p)의 종가·거래량을 0~1로 바꾸는 스케일러 도구 이름을 채우세요. (파일 위 "도구" 목록에서 고르기)
        #   예) [{"close": 160.0, "volume": 1200000}, ...]  →  [[0.42, 0.31], ...]
        #
        #   생각해 볼 질문
        #     · 이 모델은 학습할 때 어떤 도구로 입력을 0~1로 바꿨을까요? (data/features.py 의 build_sequences 참고)
        #     · 서버에서 다른 방법으로 바꾸거나, 아예 안 바꾸고 넣으면 어떻게 될까요?
        scaled = [self.scaler.transform_point(p["close"], p["volume"]) for p in sequence]

        # ② 입력 모양 맞추기 — 모델은 "문제 여러 개"를 받으므로 1개라도 [ ]로 감쌉니다. (1, 20, 2)
        x = np.array([scaled], dtype="float32")  # (1, SEQ_LEN, 2)

        # ③ 예측 — 결과가 [[0.47]] 처럼 2겹이라 [0][0] 으로 숫자만 꺼냅니다. (아직 0~1 범위)
        pred_scaled = float(self._keras_model.predict(x, verbose=0)[0][0])

        # ════════════════════════ [빈칸 3]  ④ 달러로 복원 ════════════════════════
        # 사용자에게 돌려줄 값을 만드는 스케일러 도구 이름을 채우세요. (파일 위 "도구" 목록에서 고르기)
        #
        #   생각해 볼 질문
        #     · pred_scaled 는 0.47 같은 값입니다. 이대로 응답하면 사용자는 무엇을 보게 될까요?
        #     · train_baseline_v1.py 의 STEP 7(시험 보기)에서는 예측값을 어떻게 처리했나요?
        return self.scaler.inverse_close(pred_scaled)


# ═══════════════════════════════ 어디서 불러올까? ═══════════════════════════════

def _load_from_local() -> LoadedModel:
    """Day1: 로컬 파일에서 모델과 스케일러를 불러와 상자에 담습니다. ([빈칸 5]의 참고 예시)"""
    from tensorflow import keras

    keras_model = keras.models.load_model(LOCAL_MODEL_PATH)
    scaler = HAICScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="v1-local")


def _load_from_mlflow() -> LoadedModel:
    """
    Day2: train_and_register.py 가 "HAIC_Predictor" 이름으로 등록하고 Production 으로 올려 둔 모델을
    MLflow Model Registry 에서 불러옵니다.

    확인 방법
      1) python serving_app/train_and_register.py   → "[GATE PASSED] ... promoted to Production"
      2) MODEL_SOURCE=mlflow uvicorn serving_app.main:app --host 0.0.0.0 --port 8077
      3) /predict 응답의 model_version 이 "production" 이고 predicted_close 가 달러 값이면 성공
    """
    import mlflow.tensorflow

    # 모델 : MLflow 레지스트리에서 "Production" 단계 모델을 불러옵니다. (완성)
    #   버전 번호 대신 단계(Production)로 불러오므로, 재배포 때 서버 코드를 고칠 필요가 없습니다.
    keras_model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)

    # ════════════════════════════ [빈칸 5] ════════════════════════════
    # 스케일러를 알맞은 곳에서 불러오세요. (바로 위 _load_from_local 과 비교해 보세요)
    #
    #   생각해 볼 질문
    #     · 모델은 MLflow 에서 가져왔습니다. 스케일러도 MLflow 에서 가져와야 할까요, 로컬 scaler.pkl 을 써야 할까요?
    #     · Day2 모델은 어떤 스케일러로 0~1 변환한 데이터로 학습했나요? (train_and_register.py 의 SCALER_PATH 참고)
    #     · 스케일러를 여기서 새로 fit 하면 어떤 일이 생길까요?
    scaler = HAICScaler.load(SCALER_PATH)
    return LoadedModel(keras_model=keras_model, scaler=scaler, version="production")


def _load_model() -> LoadedModel:
    """MODEL_SOURCE 값에 따라 로컬/MLflow 중 어디서 불러올지 고릅니다."""
    source = os.getenv("MODEL_SOURCE", "local")
    if source == "mlflow":
        return _load_from_mlflow()
    return _load_from_local()


# ═══════════════════════════ 언제 불러올까? (Eager / Lazy) ═══════════════════════════

def load_eager() -> LoadedModel:
    """Eager Loading: 서버가 켜질 때(main.py 의 startup) 바로 불러와 상자에 넣어 둡니다. ([빈칸 4]의 참고 예시)"""
    start = time.time()
    model = _load_model()
    print(f"[eager] model loaded in {time.time() - start:.3f}s at startup")
    global _model_cache
    _model_cache = model
    return model


def get_model() -> LoadedModel:
    """
    Lazy Loading: 첫 요청이 들어올 때만 불러오고, 이후에는 상자(_model_cache)에 있는 것을 재사용합니다.

    확인 방법
      · 서버를 켜고 /predict 를 두 번 호출 → "[lazy] model loaded in ..." 이 첫 번째에만 한 번 찍히면 성공
      · /health 의 model_loaded 가 첫 호출 전 false → 호출 후 true
      · 두 번째 호출이 첫 번째보다 훨씬 빠른지도 비교해 보세요
    """
    global _model_cache

    # ════════════════════════════ [빈칸 4] ════════════════════════════
    # "필요할 때 한 번만" 불러오도록 조건과 불러오는 코드를 채우세요.
    #
    #   생각해 볼 질문
    #     · 상자가 비어 있다는 것은 코드로 어떻게 확인할까요? (파일 위쪽 _model_cache 의 처음 값)
    #     · 이 조건문 없이 매번 불러오면, 동작은 할까요? 요청이 초당 100건이면 어떻게 될까요?
    #     · 불러오는 함수는 load_eager() 가 무엇을 호출하는지 보면 알 수 있습니다.
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model loaded in {time.time() - start:.3f}s on first request")
    return _model_cache
