"""
MLflow로 에너지 LSTM을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

노트북 `gigatime_LSTM_ML.ipynb` Cell 16(학습 루프+EarlyStopping) / Cell 18(평가) /
Cell 12(DataLoader) / Cell 21(저장)을 PyTorch + MLflow 형태로 이식.

데이터 인터페이스 (데이터셋 담당 제공, `data/features.py`):
    load_splits() -> (train_df, valid_df, test_df)  # pandas DataFrame
    FEATURE_COLS: list[str]  # 입력 피처 컬럼 (현재 lag 72개: temp/humi/energy x 24h)
    TARGET_COL: str          # 타깃 컬럼 (수치 → 변화율로 변경 중, 확정 후 반영)
    TIME_COL: str            # 시간 컬럼
    load_scaler(path)        # fit済 스케일러 (transform/inverse_transform 제공)
이 파일은 스케일러를 fit하지 않는다 (스케일링은 데이터셋 담당 영역).

실행 (project/ 루트에서):
    python serving_app/train_and_register.py
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from serving_app.lstm_model import N_FEATURES, build_model, get_device, set_seed

SEED = 42
DEVICE = get_device()

BATCH_SIZE = 64  # Arc 140V 공유메모리 OOM 방지 (노트북 256 → 64)
LR = 1e-3
BASE_EPOCHS = 50  # 노트북 EPOCHS
PATIENCE = 7  # 노트북 EarlyStopping patience
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4  # base(1e-3)보다 낮게 살짝만 갱신

MODEL_NAME = "GIGA_Energy_LSTM"
LOCAL_MODEL_PATH = "serving_app/models/energy_lstm.pt"

# NOTE(성능보류): 타깃이 수치 → 변화율로 변경 중이라 게이트 값 미확정.
# 데이터 확정 후 팀 합의로 설정. None이면 등록은 하되 promoted=False로 둔다.
# (plain 대입 유지: routers/system.py가 AST 파싱으로 이 상수를 읽는다)
RMSE_GATE = None


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _make_loaders(X_train, y_train, X_valid, y_valid, X_test, y_test):
    def _loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
            ),
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    return (
        _loader(X_train, y_train, True),
        _loader(X_valid, y_valid, False),
        _loader(X_test, y_test, False),
    )


def _load_tensors(scaler):
    """데이터셋 담당의 load_splits() + 스케일러로 학습 텐서 준비. (노트북 Cell 7/11)"""
    from data.features import FEATURE_COLS, TARGET_COL, TIME_COL, load_splits

    train_df, valid_df, test_df = load_splits()
    n_feat = len(FEATURE_COLS)
    seq_len = n_feat // N_FEATURES  # timestep당 [temp, humi, energy] N_FEATURES개

    def _xy(df):
        X = df[FEATURE_COLS].to_numpy(dtype=np.float32).reshape(-1, seq_len, N_FEATURES)
        y = df[TARGET_COL].to_numpy(dtype=np.float32)
        ts = df[TIME_COL].to_numpy()
        return X, y, ts

    X_train, y_train, ts_train = _xy(train_df)
    X_valid, y_valid, ts_valid = _xy(valid_df)
    X_test, y_test, ts_test = _xy(test_df)

    X_train_s = scaler.transform_X(X_train)
    X_valid_s = scaler.transform_X(X_valid)
    X_test_s = scaler.transform_X(X_test)
    y_train_s = scaler.transform_y(y_train)
    y_valid_s = scaler.transform_y(y_valid)
    y_test_s = scaler.transform_y(y_test)
    return (X_train_s, y_train_s, ts_train), (X_valid_s, y_valid_s, ts_valid), (
        X_test_s,
        y_test_s,
        ts_test,
    )


def evaluate(model, loader, scaler) -> tuple[float, float]:
    """노트북 Cell 16 evaluate_loader. 역스케일 후 RMSE/MAE."""
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            actuals.append(yb.numpy())
    pred = scaler.inverse_y(np.concatenate(preds))
    actual = scaler.inverse_y(np.concatenate(actuals))
    return rmse(actual, pred), mae(actual, pred)


def _fit(model, train_loader, valid_loader, scaler, epochs, lr) -> dict:
    """노트북 Cell 16 학습 루프 + EarlyStopping + best 복원."""
    criterion = nn.MSELoss()
    # XPU OOM fix: foreach Adam이 Level Zero에서 OOM → single-tensor 경로 강제
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    best_rmse = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=False), yb.to(DEVICE, non_blocking=False)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            del loss
        if DEVICE.type == "xpu":
            torch.xpu.empty_cache()
        train_rmse, _ = evaluate(model, train_loader, scaler)
        val_rmse, val_mae = evaluate(model, valid_loader, scaler)
        if DEVICE.type == "xpu":
            torch.xpu.empty_cache()
        history.append((epoch, train_rmse, val_rmse, val_mae))
        print(
            f"Epoch {epoch:02d} | Train RMSE {train_rmse:.4f} "
            f"| Val RMSE {val_rmse:.4f} | Val MAE {val_mae:.4f}"
        )
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    return {"best_val_rmse": best_rmse, "history": history}


def _register_if_gate_passed(model, run_id: str, score: float, X_example) -> dict:
    import mlflow
    import mlflow.pytorch as mlflow_pytorch
    from mlflow.tracking import MlflowClient

    # mlflow 3.x 기본 pt2 직렬화는 TensorSpec 서명을 요구하므로 pickle 사용
    # (Tensor 변환은 올바른 dtype 추론용으로 유지)
    ex = X_example
    if isinstance(ex, np.ndarray):
        ex = torch.from_numpy(np.ascontiguousarray(ex, dtype=np.float32))
    mlflow_pytorch.log_model(model, name="model", input_example=ex, serialization_format="pickle")
    result = {"run_id": run_id, "rmse": score, "promoted": False}
    if RMSE_GATE is None:
        print(f"[GATE PENDING] rmse={score:.4f} -> 게이트 미확정(변화율 타깃 확정 후 설정)")
        return result
    if score <= RMSE_GATE:
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Production"
        )
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] rmse={score:.4f} -> {MODEL_NAME} v{v.version} Production")
    else:
        print(f"[GATE FAILED] rmse={score:.4f} > {RMSE_GATE} -> 배포 차단")
    return result


def _save_local(model, scaler, feature_cols, target_col, seq_len) -> None:
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "input_shape": (seq_len, N_FEATURES),
            "feature_order": list(feature_cols),
            "target": target_col,
        },
        LOCAL_MODEL_PATH,
    )
    print(f"saved -> {LOCAL_MODEL_PATH}")


def train_and_register() -> dict:
    """처음부터(scratch) 학습. base 학습에서만 사용."""
    import mlflow

    from data.features import FEATURE_COLS, TARGET_COL, load_scaler

    set_seed(SEED)
    scaler = load_scaler()
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = _load_tensors(scaler)
    train_loader, valid_loader, test_loader = _make_loaders(Xtr, ytr, Xva, yva, Xte, yte)

    with mlflow.start_run(run_name="base-train"):
        model = build_model()
        stats = _fit(model, train_loader, valid_loader, scaler, BASE_EPOCHS, LR)
        test_rmse, test_mae = evaluate(model, test_loader, scaler)
        print(f"=== Final Test ===\nRMSE: {test_rmse:.4f}\nMAE : {test_mae:.4f}")

        mlflow.log_param("mode", "scratch")
        mlflow.log_param("epochs", BASE_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_metric("best_val_rmse", stats["best_val_rmse"])
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        _save_local(model, scaler, FEATURE_COLS, TARGET_COL, Xtr.shape[1])

        return _register_if_gate_passed(
            model, mlflow.active_run().info.run_id, test_rmse, Xtr[:1]
        )


def fine_tune(rows=None, recent_frac: float | None = None) -> dict:
    """현재 Production 가중치에서 이어서(warm start) 짧게 fine-tuning. Day3 재학습용.

    rows: 최근 원시 행 목록 (retrain_trigger가 넘겨주는 방식). 데이터셋 담당이
        data.features.tensors_from_rows(rows, scaler)를 제공하면 그걸 쓰고,
        없으면 split tail로 대체한다 (아래 recent_frac 동작).
    recent_frac: train split의 시간순 뒷부분만 사용 (예: 0.1 = 최근 10%).
        Day3 드리프트 대응 시 최근 데이터로만 갱신할 때 지정한다.
    둘 다 None이면 전체 split 사용 (기본 동작).
    """
    import mlflow
    import mlflow.pytorch as mlflow_pytorch

    from data.features import FEATURE_COLS, TARGET_COL, load_scaler

    set_seed(SEED)
    scaler = load_scaler()
    if rows is not None:
        # 최근 행만으로 재학습: split 파일 없이 rows 자체를 train/valid로 나눈다.
        # (서버 환경에 train.csv가 없어도 동작해야 하므로 _load_tensors를 쓰지 않는다)
        from data.features import tensors_from_rows

        X_all, y_all = tensors_from_rows(rows, scaler)
        cut = max(1, int(len(X_all) * 0.7))
        Xtr, ytr = X_all[:cut], y_all[:cut]
        Xva, yva = X_all[cut:], y_all[cut:]
        Xte, yte = Xva, yva  # rows 모드의 test 평가는 valid로 대체 (리포트용)
    if rows is None:
        (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = _load_tensors(scaler)
        if recent_frac is not None:
            cut = max(1, int(len(Xtr) * recent_frac))
            Xtr, ytr = Xtr[-cut:], ytr[-cut:]
    train_loader, valid_loader, test_loader = _make_loaders(Xtr, ytr, Xva, yva, Xte, yte)

    model = mlflow_pytorch.load_model(f"models:/{MODEL_NAME}/Production")
    model.to(DEVICE)

    with mlflow.start_run(run_name="fine-tune"):
        stats = _fit(model, train_loader, valid_loader, scaler, FINE_TUNE_EPOCHS, FINE_TUNE_LR)
        test_rmse, test_mae = evaluate(model, test_loader, scaler)
        print(f"=== Fine-tune Test ===\nRMSE: {test_rmse:.4f}\nMAE : {test_mae:.4f}")

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param("recent_frac", recent_frac if recent_frac is not None else "full")
        mlflow.log_param("n_train", len(Xtr))
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)
        _save_local(model, scaler, FEATURE_COLS, TARGET_COL, Xtr.shape[1])

        return _register_if_gate_passed(
            model, mlflow.active_run().info.run_id, test_rmse, Xtr[:1]
        )


if __name__ == "__main__":
    train_and_register()
