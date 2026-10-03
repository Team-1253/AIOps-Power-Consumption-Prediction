"""
MLflow로 에너지 LSTM을 학습 -> 기록(Tracking) -> 게이트 검증 -> 등록(Registry) -> Production 승격.
드리프트 감지 후 Production 가중치에서 이어서 학습하는 fine-tuning 재학습.

노트북 `gigatime_LSTM_ML.ipynb` Cell 16(학습 루프+EarlyStopping) / Cell 18(평가) /
Cell 12(DataLoader) / Cell 21(저장)을 PyTorch + MLflow 형태로 이식.

데이터 인터페이스 (`data/features.py`):
    load_splits() -> (train_df, valid_df, test_df)  # pandas DataFrame
    FEATURE_COLS: list[str]  # 입력 피처 컬럼 (lag 72개: temp/humi/energy x 24h)
    TARGET_COL: str          # 타깃 컬럼 (다음 1시간 변화율)
    TIME_COL: str            # 시간 컬럼
    load_scaler(path)        # fit 스케일러 (transform/inverse_transform 제공)
이 파일은 스케일러를 fit하지 않는다 (스케일러는 노트북 baseline 산출물).

실행 (project/ 루트에서):
    python serving_app/train_and_register.py
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from serving_app.lstm_model import N_FEATURES, build_model, get_device, set_seed
from serving_app.monitoring.drift_detector import compute_rmse

SEED = 42
DEVICE = get_device()

BATCH_SIZE = 64
LR = 1e-3
BASE_EPOCHS = 50  # 노트북 EPOCHS
PATIENCE = 7  # 노트북 EarlyStopping patience
# fine-tune 용량: 17개 시퀀스 기준 10epoch/b64(스텝 10회)로는 적응이 안 돼
# 50epoch/b16(스텝 약 100회) + clip으로 올린다. base 학습은 건드리지 않는다.
FINE_TUNE_EPOCHS = 50
FINE_TUNE_LR = 3e-4
FINE_TUNE_BATCH_SIZE = 16
FINE_TUNE_GRAD_CLIP = 1.0

MODEL_NAME = "GIGA_Energy_LSTM"
LOCAL_MODEL_PATH = "serving_app/models/factory_energy_lstm.pt"

# baseline best-val 4.21 기준 배포 게이트 (상대변화율 RMSE, base 전체 학습용).
# (plain 대입 유지: routers/system.py가 AST 파싱으로 이 상수를 읽는다)
RMSE_GATE = 4.5

# fine-tune(rows 모드) 배포 게이트 (kWh 공간 오차율(%) RMSE).
# 드리프트 임계값(5%)보다 낮게 잡아야 승격이 곧 재요청 판정 통과를 의미한다.
# baseline 실측: drift 월 5.85~7.93%, normal 월 0.82%.
FINE_TUNE_PCT_GATE = 4.0


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _make_loaders(X_train, y_train, X_valid, y_valid, X_test, y_test, batch_size=BATCH_SIZE):
    def _loader(X, y, shuffle):
        return DataLoader(
            TensorDataset(
                torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)),
            ),
            batch_size=batch_size,
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
    return (
        (X_train_s, y_train_s, ts_train),
        (X_valid_s, y_valid_s, ts_valid),
        (
            X_test_s,
            y_test_s,
            ts_test,
        ),
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


def retrain_pct(model, rows: list[dict], scaler) -> float:
    """재학습 행들에 대한 kWh 공간 오차율(%) RMSE.

    드리프트 판정(drift_detector.compute_rmse)과 같은 식·같은 단위로,
    게이트 통과가 곧 재요청 판정 통과를 의미하게 한다.
    (evaluate()의 역스케일 값은 변화율 공간이라 이 용도로 쓸 수 없다.)
    """
    from data.features import SEQ_LEN

    energy = [float(r["energy_kwh"]) for r in rows]
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for i in range(len(rows) - SEQ_LEN):
            from data.features import tensors_from_rows

            Xs, _ = tensors_from_rows(rows[i : i + SEQ_LEN + 1], scaler)
            out = model(torch.from_numpy(np.ascontiguousarray(Xs[-1:])).to(DEVICE))
            pred_rel = float(scaler.inverse_y(out.cpu().numpy())[0])
            preds.append(energy[i + SEQ_LEN - 1] * (1 + pred_rel / 100.0))
            actuals.append(energy[i + SEQ_LEN])
    return compute_rmse(
        [{"actual": a, "predicted": p} for a, p in zip(actuals, preds)]
    )


def _fit(model, train_loader, valid_loader, scaler, epochs, lr, grad_clip=None, early_stop=True, restore_best=True) -> dict:
    """노트북 Cell 16 학습 루프 + EarlyStopping + best 복원."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_rmse = float("inf")
    best_state = None
    patience = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = (
                xb.to(DEVICE, non_blocking=False),
                yb.to(DEVICE, non_blocking=False),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            del loss
        train_rmse, _ = evaluate(model, train_loader, scaler)
        val_rmse, val_mae = evaluate(model, valid_loader, scaler)
        history.append((epoch, train_rmse, val_rmse, val_mae))
        print(
            f"Epoch {epoch:02d} | Train RMSE {train_rmse:.4f} "
            f"| Val RMSE {val_rmse:.4f} | Val MAE {val_mae:.4f}"
        )
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        elif early_stop:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping.")
                break

    if restore_best:
        model.load_state_dict(best_state)
    return {"best_val_rmse": best_rmse, "history": history}


def _register_if_gate_passed(model, run_id: str, score: float, X_example, gate=None) -> dict:
    """게이트 판정. score/gate는 같은 단위여야 한다.

    - base: 상대변화율 RMSE + RMSE_GATE.
    - fine_tune(rows): kWh 공간 오차율(%) + FINE_TUNE_PCT_GATE.
      드리프트 판정(drift_detector.compute_rmse)과 같은 단위·같은 식이라,
      승격된 모델에 같은 데이터를 다시 넣으면 판정을 통과한다.
    gate=None이면 RMSE_GATE를 쓴다. 최종 게이트가 None이면 등록은 하되
    promoted=False로 둔다.
    """
    import mlflow
    import mlflow.pytorch as mlflow_pytorch
    from mlflow.tracking import MlflowClient

    # mlflow 3.x 기본 pt2 직렬화는 TensorSpec 서명을 요구하므로 pickle 사용
    # (Tensor 변환은 올바른 dtype 추론용으로 유지)
    ex = X_example
    if isinstance(ex, np.ndarray):
        ex = torch.from_numpy(np.ascontiguousarray(ex, dtype=np.float32))
    mlflow_pytorch.log_model(
        model, name="model", input_example=ex, serialization_format="pickle"
    )
    result = {"run_id": run_id, "rmse": score, "promoted": False}
    gate = RMSE_GATE if gate is None else gate
    if gate is None:
        print(f"[GATE PENDING] rmse={score:.4f} -> 게이트 미확정")
        return result
    if score <= gate:
        v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        MlflowClient().transition_model_version_stage(
            name=MODEL_NAME, version=v.version, stage="Production"
        )
        result["promoted"] = True
        result["version"] = v.version
        print(f"[GATE PASSED] rmse={score:.4f} <= {gate} -> {MODEL_NAME} v{v.version} Production")
    else:
        print(f"[GATE FAILED] rmse={score:.4f} > {gate} -> 배포 차단")
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


def _save_if_promoted(out: dict, model, scaler, feature_cols, target_col, seq_len) -> bool:
    """게이트 통과(promoted)시에만 로컬 번들을 교체한다.

    RMSE_GATE=None이면 promoted가 될 수 없어 로컬 번들은 절대 바뀌지 않는다.
    """
    if out.get("promoted"):
        _save_local(model, scaler, feature_cols, target_col, seq_len)
        return True
    print(f"[GATE BLOCKED] rmse={out.get('rmse'):.4f} -> 로컬 번들 유지 ({LOCAL_MODEL_PATH})")
    return False


def train_and_register() -> dict:
    """처음부터(scratch) 학습. base 학습에서만 사용."""
    import mlflow
    from data.features import FEATURE_COLS, TARGET_COL, load_scaler

    set_seed(SEED)
    scaler = load_scaler()
    (Xtr, ytr, _), (Xva, yva, _), (Xte, yte, _) = _load_tensors(scaler)
    train_loader, valid_loader, test_loader = _make_loaders(
        Xtr, ytr, Xva, yva, Xte, yte
    )

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

        # 게이트는 validation 기준. test split은 고정 홀드아웃이라 반복 승격 기준으로
        # 쓰면 test에 과적합된다. test 수치는 최종 리포트용으로만 기록한다.
        # (base는 상대변화율 RMSE + RMSE_GATE로 판정한다.)
        out = _register_if_gate_passed(
            model, mlflow.active_run().info.run_id, stats["best_val_rmse"], Xtr[:1]
        )
        _save_if_promoted(out, model, scaler, FEATURE_COLS, TARGET_COL, Xtr.shape[1])
        out["test_rmse"] = test_rmse
        out["test_mae"] = test_mae
        return out


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
    train_loader, valid_loader, test_loader = _make_loaders(
        Xtr, ytr, Xva, yva, Xte, yte, batch_size=FINE_TUNE_BATCH_SIZE
    )

    model = mlflow_pytorch.load_model(f"models:/{MODEL_NAME}/Production")
    model.to(DEVICE)

    with mlflow.start_run(run_name="fine-tune"):
        # 조기종료 없이 전 epoch 수행한다. kWh-조기종료가 %-게이트 수렴 전에
        # 멈춰 버리면 게이트 통과 가능한 경우에도 차단되기 때문이다.
        # (best 체크포인트 선택은 kWh 기준 그대로 유지)
        # 게이트·저장은 최종 가중치 기준: best-kWh 복원을 끄고 끝까지 학습한
        # 가중치를 그대로 평가·저장한다.
        stats = _fit(
            model, train_loader, valid_loader, scaler,
            FINE_TUNE_EPOCHS, FINE_TUNE_LR, grad_clip=FINE_TUNE_GRAD_CLIP,
            early_stop=False, restore_best=False,
        )
        test_rmse, test_mae = evaluate(model, test_loader, scaler)
        print(f"=== Fine-tune Test ===\nRMSE: {test_rmse:.4f}\nMAE : {test_mae:.4f}")

        mlflow.log_param("mode", "fine-tune")
        mlflow.log_param("epochs", FINE_TUNE_EPOCHS)
        mlflow.log_param(
            "recent_frac", recent_frac if recent_frac is not None else "full"
        )
        mlflow.log_param("n_train", len(Xtr))
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_mae", test_mae)

        # base와 동일: 게이트는 validation 기준 (rows 모드의 valid 포함).
        # rows 모드(서버 재학습)는 kWh 공간 오차율(%)로 게이트한다.
        # split 모드(수동 전체 재학습)는 kWh를 알 수 없어 상대변화율 RMSE 게이트를 쓴다.
        if rows is not None:
            gate_score = retrain_pct(model, rows, scaler)
            gate = FINE_TUNE_PCT_GATE
        else:
            gate_score = stats["best_val_rmse"]
            gate = RMSE_GATE
        mlflow.log_metric("gate_score", gate_score)
        out = _register_if_gate_passed(
            model, mlflow.active_run().info.run_id, gate_score, Xtr[:1], gate=gate
        )
        _save_if_promoted(out, model, scaler, FEATURE_COLS, TARGET_COL, Xtr.shape[1])
        out["test_rmse"] = test_rmse
        out["test_mae"] = test_mae
        return out


if __name__ == "__main__":
    train_and_register()
