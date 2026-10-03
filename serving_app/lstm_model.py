"""
공장 전력사용량 1시간 후 예측용 LSTM 아키텍처 (baseline 학습과 MLflow 학습이 공유).

노트북 `gigatime_LSTM_ML.ipynb`의 LSTMRegressor와 동일 규격.
입력: 과거 24시간 x [temp_F, humi_pct, energy_relative_pct] (24, 3),
출력: 다음 1시간 변화율 스칼라 1개. 하이퍼파라미터(hidden 64, 2층,
dropout 0.2, head 64→32→1)도 노트북과 같다.

기준 실행 환경은 CPU다.
"""

import random

import numpy as np
import torch
import torch.nn as nn

SEED = 42

# 입력 shape (노트북 baseline과 동일: lag_24h..lag_1h x [temp_F, humi_pct, energy_relative_pct]).
SEQ_LEN = 24
N_FEATURES = 3


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
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
