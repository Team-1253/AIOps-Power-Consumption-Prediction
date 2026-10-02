"""
HAIC 데이터를 LSTM 입력용 시퀀스로 변환하는 공용 유틸리티.

Day1 baseline 학습(scripts/train_baseline_v1.py), Day2 MLflow 학습
(serving_app/train_and_register.py), Day3 fine-tuning 재학습
(monitoring/retrain_trigger.py)이 모두 이 모듈을 재사용합니다. 시퀀스 정의를
한 곳에서만 관리해야 "서빙 시점 입력"과 "학습 시점 입력"이 어긋나는 실무 사고를
방지할 수 있습니다.

입력 시퀀스: 최근 SEQ_LEN(20)거래일의 (close, volume)
타깃: 그다음 거래일의 close
"""
import csv
import pickle

SEQ_LEN = 20  # LSTM 입력 윈도우 길이 (거래일 수) - 약 1개월치 거래일


def load_rows(csv_path: str = "data/haic_prices.csv") -> list[dict]:
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [
            {
                "Date": r["Date"],
                "Close": float(r["Close"]),
                "Volume": float(r["Volume"]),
            }
            for r in reader
        ]
    return rows


class HAICScaler:
    """
    close/volume을 각각 [0, 1] 범위로 정규화하는 min-max 스케일러.

    LSTM은 스케일에 민감하기 때문에(트리 기반 모델과 달리) 반드시 정규화가 필요합니다.
    Day1에서 base 데이터로 한 번 fit한 뒤 serving_app/models/scaler.pkl로 저장해두고,
    Day2 MLflow 학습과 Day3 fine-tuning 모두 같은 스케일러를 재사용합니다.
    (fine-tuning 시 스케일러를 다시 fit하지 않는 이유: 이미 이 스케일로 학습된 모델
     가중치와 어긋나면 fine-tuning 자체가 무의미해지기 때문입니다.)
    """

    def __init__(self):
        self.close_min = self.close_max = None
        self.volume_min = self.volume_max = None

    def fit(self, rows: list[dict]) -> "HAICScaler":
        closes = [r["Close"] for r in rows]
        volumes = [r["Volume"] for r in rows]
        self.close_min, self.close_max = min(closes), max(closes)
        self.volume_min, self.volume_max = min(volumes), max(volumes)
        return self

    def _scale(self, value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return (value - lo) / (hi - lo)

    def _unscale(self, value: float, lo: float, hi: float) -> float:
        return value * (hi - lo) + lo

    def transform_point(self, close: float, volume: float) -> list[float]:
        return [
            self._scale(close, self.close_min, self.close_max),
            self._scale(volume, self.volume_min, self.volume_max),
        ]

    def scale_close(self, close: float) -> float:
        """타깃(다음날 종가)을 학습용으로 정규화. 입력 시퀀스와 같은 스케일을 써야
        손실(loss)이 과도하게 커지지 않고 학습이 안정적으로 수렴한다."""
        return self._scale(close, self.close_min, self.close_max)

    def inverse_close(self, scaled_close: float) -> float:
        """모델이 뱉은 정규화된 예측값을 실제 달러 단위 종가로 되돌린다."""
        return self._unscale(scaled_close, self.close_min, self.close_max)

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "HAICScaler":
        scaler = cls()
        with open(path, "rb") as f:
            scaler.__dict__.update(pickle.load(f))
        return scaler


def build_sequences(rows: list[dict], scaler: HAICScaler, seq_len: int = SEQ_LEN):
    """
    rows(시간순 OHLCV)에서 (SEQ_LEN, 2) 크기의 정규화된 입력 시퀀스와
    다음날 종가(정규화 전 실값) 타깃을 만든다.

    반환: X (n_samples, seq_len, 2), y (n_samples,) - y는 스케일 안 된 실제 종가
    """
    scaled_points = [scaler.transform_point(r["Close"], r["Volume"]) for r in rows]
    closes = [r["Close"] for r in rows]

    X, y = [], []
    for i in range(len(rows) - seq_len):
        X.append(scaled_points[i : i + seq_len])
        y.append(closes[i + seq_len])
    return X, y


def train_test_split(X: list, y: list, test_ratio: float = 0.2):
    """시간 순서를 유지한 채 앞부분을 train, 뒷부분을 test로 나눈다 (미래 데이터 누수 방지)."""
    split_idx = int(len(X) * (1 - test_ratio))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
