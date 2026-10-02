import os
from pathlib import Path

import numpy as np
import pandas as pd


SPLIT_NAMES = ("train_pos", "val_pos", "test_pos", "train_neg", "val_neg", "test_neg")
PRECISION_LABELS = (
    "10pct", "20pct", "30pct", "40pct", "50pct",
    "60pct", "70pct", "80pct", "90pct", "100pct",
)
HITS_LABELS = (25, 50, 100, 200, 400, 800, 1600, 3200)


def _to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _to_2d_array(value):
    array = _to_numpy(value)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.astype(np.int64, copy=False)


def make_split_dir(base_dir, setting, seed, fold_idx):
    return str(Path(base_dir) / "splits" / setting / f"seed_{seed}" / f"fold_{fold_idx}")


def save_split(split_dir, **splits):
    split_path = Path(split_dir)
    split_path.mkdir(parents=True, exist_ok=True)
    for name, values in splits.items():
        if values is None:
            continue
        array = _to_2d_array(values)
        df = pd.DataFrame(array, columns=["drug_id", "target_id"])
        df.to_csv(split_path / f"{name}.csv", index=False)


def load_split(split_dir, as_torch=True, device=None):
    split_path = Path(split_dir)
    loaded = {}
    for csv_path in sorted(split_path.glob("*.csv")):
        df = pd.read_csv(csv_path)
        array = df[["drug_id", "target_id"]].to_numpy(dtype=np.int64)
        if as_torch:
            import torch
            tensor = torch.tensor(array, dtype=torch.long)
            if device is not None:
                tensor = tensor.to(device)
            loaded[csv_path.stem] = tensor
        else:
            loaded[csv_path.stem] = array
    return loaded


def load_saved_folds(base_dir, setting, seed, fold_count=5):
    """Load a complete saved split, or return None when any required file is absent."""
    required = SPLIT_NAMES + ("test_neg_all",)
    columns = {name: [] for name in required}
    for fold_idx in range(fold_count):
        split_dir = make_split_dir(base_dir, setting, seed, fold_idx)
        loaded = load_split(split_dir)
        if any(name not in loaded for name in required):
            return None
        for name in required:
            columns[name].append(loaded[name])
    return tuple(columns[name] for name in required)


def save_metrics_txt(path, metrics):
    metric_path = Path(path)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metric_path, sep="\t", index=False)


def metrics_to_dict(setting, seed, fold_idx, results, suffix=None):
    metrics = {
        "setting": setting,
        "seed": seed,
        "fold": fold_idx,
    }
    name_prefix = "" if suffix is None else f"{suffix}_"
    if len(results) >= 2:
        metrics[f"{name_prefix}auc"] = float(results[0])
        metrics[f"{name_prefix}aupr"] = float(results[1])
    if len(results) >= 4 and isinstance(results[2], (list, tuple)) and isinstance(results[3], (list, tuple)):
        for label, value in zip(PRECISION_LABELS, results[2]):
            metrics[f"{name_prefix}precision_at_{label}"] = float(value)
        for label, value in zip(HITS_LABELS, results[3]):
            metrics[f"{name_prefix}hits_at_{label}"] = float(value)
    elif len(results) >= 7:
        metrics[f"{name_prefix}accuracy"] = float(results[2])
        metrics[f"{name_prefix}precision"] = float(results[3])
        metrics[f"{name_prefix}recall"] = float(results[4])
        metrics[f"{name_prefix}f1"] = float(results[5])
        metrics[f"{name_prefix}best_threshold"] = float(results[6])
    return metrics


def _mean_weights(weights):
    array = _to_numpy(weights).astype(float)
    if array.ndim == 0:
        array = array.reshape(1)
    array = np.squeeze(array)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        array = array.reshape(array.shape[0], -1).mean(axis=1)
    return array.reshape(-1)


def _attention_rows(seed, fold_idx, beta_user, beta_item, epoch=None):
    rows = []
    for side, weights in (("drug", beta_user), ("target", beta_item)):
        if weights is None:
            continue
        for idx, weight in enumerate(_mean_weights(weights)):
            row = {
                "seed": seed,
                "fold": fold_idx,
                "side": side,
                "metapath_index": idx,
                "weight": float(weight),
            }
            if epoch is not None:
                row["epoch"] = epoch
            rows.append(row)
    return rows


def save_attention_weights(path, seed, fold_idx, beta_user, beta_item):
    rows = _attention_rows(seed, fold_idx, beta_user, beta_item)
    attention_path = Path(path)
    attention_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["seed", "fold", "side", "metapath_index", "weight"]).to_csv(
        attention_path, sep="\t", index=False)


def append_attention_weights_by_epoch(path, seed, fold_idx, epoch, beta_user, beta_item):
    rows = _attention_rows(seed, fold_idx, beta_user, beta_item, epoch=epoch)
    attention_path = Path(path)
    attention_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["seed", "fold", "epoch", "side", "metapath_index", "weight"]
    pd.DataFrame(rows, columns=columns).to_csv(
        attention_path,
        sep="\t",
        index=False,
        mode="a",
        header=not attention_path.exists(),
    )


def save_attention_weights_history_mean(path, seed, fold_idx, beta_user_history, beta_item_history):
    mean_beta_user = _mean_history_weights(beta_user_history)
    mean_beta_item = _mean_history_weights(beta_item_history)
    save_attention_weights(path, seed, fold_idx, mean_beta_user, mean_beta_item)


def _mean_history_weights(history):
    if not history:
        return None
    epoch_weights = [_mean_weights(weights) for weights in history]
    return np.stack(epoch_weights, axis=0).mean(axis=0)
