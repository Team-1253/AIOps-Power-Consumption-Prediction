"""
공장 전력사용량 1시간 후 예측용 LSTM 아키텍처 (baseline 학습과 MLflow 학습이 공유).

노트북 `gigatime_LSTM_ML.ipynb` Cell 14 (LSTMRegressor)를 그대로 이식한 PyTorch 버전.
입력: 과거 24시간 x [temp, humidity, energy] (24, 3), 출력: 다음 1시간 타깃 스칼라 1개.
※ 타깃이 수치 → 변화율로 변경 중(데이터셋 담당). 회귀 구조는 동일하므로 head/loss는
그대로 두고, 변화율 컬럼명·역변환식은 데이터 확정 후 반영한다.

장치 우선순위: cuda → xpu(Intel Arc) → cpu. (노트북 Cell 4 로직)
XPU 학습 시 OOM 회피: Adam(foreach=False) + batch 64 + empty_cache는
train_and_register.py 쪽에서 처리한다.
"""

import random

import numpy as np
import torch
import torch.nn as nn

SEED = 42

# 입력 shape 기본값. 정식 값(SEQ_LEN/N_FEATURES)은 데이터셋 담당이 확정한다.
# (현재 lag 데이터셋 기준: lag_24h..lag_1h x [temp_F, humi_pct, energy_relative_pct])
SEQ_LEN = 24
N_FEATURES = 3


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int = N_FEATURES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(
    input_size: int = N_FEATURES,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    device: torch.device | None = None,
) -> LSTMRegressor:
    """학습/서빙이 공유하는 모델 팩토리. 가중치는 호출자가 로드한다."""
    set_seed(SEED)
    model = LSTMRegressor(input_size, hidden_size, num_layers, dropout)
    return model.to(device or get_device())
