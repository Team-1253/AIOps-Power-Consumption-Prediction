"""
에너지 lag 데이터셋 공용 유틸리티 (학습/서빙이 공유하는 단일 소스).

serving_app/train_and_register.py(학습), serving_app/model_loader.py(서빙),
monitoring/retrain_trigger.py(재학습)가 모두 이 모듈을 재사용한다. 시퀀스 정의와
컬럼명을 한 곳에서만 관리해야 "서빙 시점 입력"과 "학습 시점 입력"이 어긋나는
실무 사고를 방지할 수 있다.

입력 시퀀스: 과거 SEQ_LEN(24)시간 x [temp_F, humi_pct, energy_relative_pct]
타깃: 다음 1시간의 energy_relative_pct (변화율, % 단위)

값 미확정(PENDING): SEQ_LEN/FEATURE_COLS/TARGET_COL 최종값은 데이터셋 담당 확정 후 반영.
현재 lag CSV(train.csv/valid.csv/test.csv) 규격 기준으로 작성됨.
"""

import os
import pickle

import numpy as np
import pandas as pd

SEQ_LEN = 24  # LSTM 입력 윈도우 길이 (시간 수)
N_FEATURES = 3  # timestep당 [temp_F, humi_pct, energy_relative_pct]

TARGET_COL = "target_energy_relative_pct"
TIME_COL = "target_time_utc"

FEATURE_COLS: list[str] = []
for _lag in range(SEQ_LEN, 0, -1):
    FEATURE_COLS += [
        f"temp_F_lag_{_lag}h",
        f"humi_pct_lag_{_lag}h",
        f"energy_relative_pct_lag_{_lag}h",
    ]

# 원시 시간별 CSV 컬럼 (uploads/hourly_clean 규격, schemas.py HourlyPoint와 동일)
RAW_ENERGY_COL = "energy_kwh"
RAW_HUMI_COL = "humi_pct"
RAW_TEMP_COL = "temp_F"


def _data_dir(data_dir: str | None) -> str:
    return data_dir or "data"


def load_splits(data_dir: str | None = None):
    """미리 분리된 train/valid/test lag CSV를 시간순으로 로드."""
    d = _data_dir(data_dir)
    train_df = pd.read_csv(os.path.join(d, "train.csv"), parse_dates=[TIME_COL])
    valid_df = pd.read_csv(os.path.join(d, "valid.csv"), parse_dates=[TIME_COL])
    test_df = pd.read_csv(os.path.join(d, "test.csv"), parse_dates=[TIME_COL])
    for _df in (train_df, valid_df, test_df):
        _df.sort_values(TIME_COL, inplace=True)
        _df.reset_index(drop=True, inplace=True)
    return train_df, valid_df, test_df


class EnergyScaler:
    """피처(X: StandardScaler over (N,3)) + 타깃(y) 분리 스케일러.

    train split에만 fit하고 valid/test는 transform만 한다.
    pickle 저장/로드 지원 (serving_app/models/scaler.pkl).
    """

    def __init__(self):
        from sklearn.preprocessing import StandardScaler

        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

    def fit(self, X_train, y_train) -> "EnergyScaler":
        self.feature_scaler.fit(np.asarray(X_train).reshape(-1, N_FEATURES))
        self.target_scaler.fit(np.asarray(y_train).reshape(-1, 1))
        return self

    def transform_X(self, a: np.ndarray):
        arr = np.asarray(a, dtype=np.float32)
        shape = arr.shape
        out = self.feature_scaler.transform(arr.reshape(-1, N_FEATURES))
        return out.reshape(shape).astype(np.float32)

    def transform_y(self, y: np.ndarray):
        arr = np.asarray(y, dtype=float).reshape(-1, 1)
        return self.target_scaler.transform(arr).ravel().astype(np.float32)

    def inverse_y(self, ys: np.ndarray):
        arr = np.asarray(ys, dtype=float).reshape(-1, 1)
        return self.target_scaler.inverse_transform(arr).ravel()

    def save(self, path: str = "serving_app/models/scaler.pkl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = "serving_app/models/scaler.pkl") -> "EnergyScaler":
        with open(path, "rb") as f:
            return pickle.load(f)


def load_scaler(path: str = "serving_app/models/scaler.pkl") -> EnergyScaler:
    return EnergyScaler.load(path)


def _relative_series(energy: np.ndarray) -> np.ndarray:
    """kWh 시계열 → 시간별 변화율(%). 첫 행은 0.0 (이전값 없음)."""
    energy = np.asarray(energy, dtype=float)
    out = np.zeros_like(energy)
    prev = energy[:-1]
    cur = energy[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(prev != 0, 100.0 * (cur - prev) / prev, 0.0)
    out[1:] = pct
    return out


def tensors_from_rows(rows: list[dict], scaler: EnergyScaler):
    """원시 시간별 행들 → 스케일된 (X, y).

    rows: [{"energy_kwh":.., "humi_pct":.., "temp_F":..}, ...] 시간순.
    (retrain_trigger가 최근 업로드에서 잘라 넘기는 형식)
    energy축은 파일 내 직전값 대비 변화율(%)로 변환한다.
    반환: (X_scaled (n, SEQ_LEN, 3), y_scaled (n,)) — n = len(rows) - SEQ_LEN.
    """
    e = np.array([float(r[RAW_ENERGY_COL]) for r in rows], dtype=float)
    h = np.array([float(r[RAW_HUMI_COL]) for r in rows], dtype=float)
    t = np.array([float(r[RAW_TEMP_COL]) for r in rows], dtype=float)
    er = _relative_series(e)
    feats = np.stack([t, h, er], axis=1).astype(np.float32)  # (n, 3)
    X, y = [], []
    for i in range(len(feats) - SEQ_LEN):
        X.append(feats[i : i + SEQ_LEN])
        y.append(er[i + SEQ_LEN])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return scaler.transform_X(X), scaler.transform_y(y)
