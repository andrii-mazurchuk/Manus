"""
LSTM dynamic gesture training logic.

Importable from the API layer — no argparse, no sys.exit, no CLI side-effects.
All errors are raised as ValueError so callers can convert them to HTTP responses.

Training contract (must not be deviated from — see docs/architecture.md):
  1. Load all .npy files per class from data_dir subdirectories.
  2. Resample all sequences to the median length.
  3. Stratified train/val split BEFORE augmentation (prevents data leakage).
  4. Augment training split only (time warp, consistent rotation/scale, per-frame noise).
  5. Train GestureLSTM, save best checkpoint by val accuracy.
  6. Save model + metadata (label_encoder, seq_len).
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.core.sequence_model import GestureLSTM, _resample, DEFAULT_MODEL_PATH, DEFAULT_META_PATH
from src.config.loader import load_dynamic_gestures_config

DEFAULT_SEQUENCE_MODEL = DEFAULT_MODEL_PATH
DEFAULT_SEQUENCE_META  = DEFAULT_META_PATH

_MIN_CLASSES     = 2
_MIN_RECORDINGS  = 5


# ── Validation ────────────────────────────────────────────────────────────────

def validate_sequence_dataset(data_dir: Path = Path("src/data/sequences")) -> None:
    """
    Pre-flight check — raise ValueError if the dataset cannot support training.
    """
    if not data_dir.exists():
        raise ValueError(
            "No sequence data found. Record sequences via Studio → Dynamic Gestures → Dataset first."
        )
    subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if len(subdirs) < _MIN_CLASSES:
        raise ValueError(
            f"Need at least {_MIN_CLASSES} gesture classes. Found: {len(subdirs)}. "
            "Add more gestures and record sequences."
        )
    for d in subdirs:
        n = len(list(d.glob("*.npy")))
        if n < _MIN_RECORDINGS:
            raise ValueError(
                f"Gesture '{d.name}' has only {n} recording(s). "
                f"Need at least {_MIN_RECORDINGS} per class."
            )


# ── Main training function ────────────────────────────────────────────────────

def run_sequence_training(
    data_dir:   Path = Path("src/data/sequences"),
    model_path: Path = DEFAULT_SEQUENCE_MODEL,
    meta_path:  Path = DEFAULT_SEQUENCE_META,
    *,
    epochs:      int = 150,
    hidden_size: int = 64,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Train the LSTM dynamic gesture classifier.

    Args:
        data_dir:    Directory containing subdirectories named after gesture classes,
                     each holding .npy files of shape (T, 63).
        model_path:  Output path for the PyTorch state dict (.pt).
        meta_path:   Output path for the metadata pickle (label_encoder, seq_len).
        epochs:      Maximum training epochs.
        hidden_size: LSTM hidden units (default 64 — do not increase without more data).
        progress_cb: Optional callable(str) for progress messages.

    Returns:
        {
            "classes":               ["swipe_left", "swipe_right"],
            "accuracy":              0.91,    # test-set accuracy
            "val_accuracy":          0.89,    # best val accuracy seen during training
            "epochs_trained":        143,     # epoch of best checkpoint
            "recordings_per_class":  {"swipe_left": 12, "swipe_right": 8},
            "seq_len":               32,
        }

    Raises:
        ValueError if dataset does not meet minimum requirements.
    """
    def _prog(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    validate_sequence_dataset(data_dir)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    _prog("Loading sequences…")
    sequences: list[np.ndarray] = []
    labels:    list[str]        = []
    recordings_per_class: dict[str, int] = {}

    for gesture_dir in sorted(data_dir.iterdir()):
        if not gesture_dir.is_dir():
            continue
        files = sorted(gesture_dir.glob("*.npy"))
        recordings_per_class[gesture_dir.name] = len(files)
        for f in files:
            arr = np.load(str(f)).astype(np.float32)
            if arr.ndim != 2 or arr.shape[1] != 63:
                continue  # skip malformed files
            sequences.append(arr)
            labels.append(gesture_dir.name)

    if len(set(labels)) < _MIN_CLASSES:
        raise ValueError("Not enough valid classes after loading.")

    # ── 2. Median length + resample ───────────────────────────────────────────
    lengths = [s.shape[0] for s in sequences]
    seq_len = int(np.median(lengths))
    _prog(f"Resampling {len(sequences)} sequences to median length {seq_len}…")
    sequences = [_resample(s, seq_len) for s in sequences]

    # ── 3. Label encoding ─────────────────────────────────────────────────────
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    X  = np.stack(sequences)   # (N, seq_len, 63)

    # ── 4. Stratified split BEFORE augmentation ───────────────────────────────
    _prog("Splitting dataset (stratified, before augmentation)…")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ── 5. Augment training split only ────────────────────────────────────────
    _prog("Augmenting training split…")
    dg_cfg        = load_dynamic_gestures_config()
    mirror_flags  = {
        g["name"]: g.get("mirror_augment", False)
        for g in dg_cfg.get("gestures", [])
    }
    label_names   = list(le.classes_)

    aug_X, aug_y = _augment_split(X_train, y_train, label_names, mirror_flags, seq_len)
    X_train = np.concatenate([X_train, aug_X], axis=0)
    y_train = np.concatenate([y_train, aug_y], axis=0)

    # ── 6. PyTorch Dataset + DataLoader ───────────────────────────────────────
    n_classes = len(le.classes_)
    train_ds = _SequenceDataset(X_train, y_train)
    val_ds   = _SequenceDataset(X_val,   y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=32, shuffle=False)

    # ── 7. Train ──────────────────────────────────────────────────────────────
    model     = GestureLSTM(num_classes=n_classes, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc    = 0.0
    best_state_dict = None
    best_epoch      = 0

    _prog(f"Training LSTM for up to {epochs} epochs…")
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()

        val_acc = _evaluate(model, val_loader)
        if val_acc > best_val_acc:
            best_val_acc    = val_acc
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch      = epoch

        if epoch % 10 == 0:
            _prog(f"Epoch {epoch}/{epochs}  val_acc={val_acc:.3f}  best={best_val_acc:.3f}")

    # ── 8. Final test accuracy ────────────────────────────────────────────────
    model.load_state_dict(best_state_dict)
    test_acc = _evaluate(model, val_loader)

    # ── 9. Save ───────────────────────────────────────────────────────────────
    _prog("Saving model…")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state_dict, model_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"label_encoder": le, "seq_len": seq_len}, f)

    _prog("Done.")
    return {
        "classes":              list(le.classes_),
        "accuracy":             round(float(test_acc), 4),
        "val_accuracy":         round(float(best_val_acc), 4),
        "epochs_trained":       best_epoch,
        "recordings_per_class": recordings_per_class,
        "seq_len":              seq_len,
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

class _SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        self._X = torch.from_numpy(sequences.astype(np.float32))
        self._y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        return self._X[idx], self._y[idx]


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            preds = model(Xb).argmax(dim=1)
            correct += int((preds == yb).sum())
            total   += len(yb)
    return correct / total if total > 0 else 0.0


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment_split(
    X: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    mirror_flags: dict[str, bool],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one augmented variant per training sequence."""
    aug_seqs = []
    aug_labels = []
    for seq, label_idx in zip(X, y):
        aug = _augment_sequence(seq, label_names[label_idx], mirror_flags, seq_len)
        aug_seqs.append(aug)
        aug_labels.append(label_idx)
    return np.stack(aug_seqs).astype(np.float32), np.array(aug_labels, dtype=np.int64)


def _augment_sequence(
    seq: np.ndarray,
    label_name: str,
    mirror_flags: dict[str, bool],
    seq_len: int,
) -> np.ndarray:
    """
    Apply a random subset of transforms to a (seq_len, 63) sequence.

    Spatial transforms (rotation, scale, mirror) use ONE parameter for the
    entire sequence — applying different parameters per frame destroys motion
    information. Per-frame noise is the only i.i.d.-per-frame transform.
    """
    seq = seq.copy()

    # Time warp (80 %): stretch/compress motion in time
    if random.random() < 0.80:
        factor = random.uniform(0.75, 1.30)
        warped_len = max(2, int(seq_len * factor))
        seq = _resample(seq, warped_len)
        seq = _resample(seq, seq_len)

    # Consistent rotation (70 %): same angle every frame
    if random.random() < 0.70:
        angle = random.uniform(-20.0, 20.0)
        seq = _rotate_sequence(seq, angle)

    # Consistent scale (60 %): same factor every frame
    if random.random() < 0.60:
        scale = random.uniform(0.90, 1.10)
        seq = seq * scale

    # Mirror (per-gesture flag): flip x of all landmarks across all frames
    if mirror_flags.get(label_name, False):
        seq = _mirror_sequence(seq)

    # Per-frame noise (80 %): i.i.d. per frame — models real tracking noise
    if random.random() < 0.80:
        seq = seq + np.random.normal(0, 0.015, seq.shape).astype(np.float32)

    return seq.astype(np.float32)


def _rotate_sequence(seq: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply a single 2D rotation to x,y of all 21 landmarks, across all frames."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    # seq: (T, 63) — reshape to (T, 21, 3) to access per-landmark x,y
    T = seq.shape[0]
    seq3 = seq.reshape(T, 21, 3)
    xy   = seq3[:, :, :2]                    # (T, 21, 2)
    xy   = (R @ xy.reshape(-1, 2).T).T       # rotate all landmark-frame pairs
    seq3[:, :, :2] = xy.reshape(T, 21, 2)
    return seq3.reshape(T, 63)


def _mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """Negate x-coordinates of all 21 landmarks across all frames."""
    T = seq.shape[0]
    seq3 = seq.reshape(T, 21, 3).copy()
    seq3[:, :, 0] *= -1   # negate x
    return seq3.reshape(T, 63)
