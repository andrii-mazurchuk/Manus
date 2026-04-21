"""
LSTM-based dynamic gesture classifier.

Architecture: LSTM(63 → hidden=64, layers=1, batch_first=True) → Linear(64 → n_classes)
Deliberately small: single layer, 64 hidden units. At this dataset scale (tens of
recordings per class after augmentation) a larger model overfits before it benefits
from depth.

Paths:
    src/models/sequence_classifier.pt      — PyTorch state dict
    src/models/sequence_classifier_meta.pkl — {label_encoder, seq_len}
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "sequence_classifier.pt"
DEFAULT_META_PATH  = Path(__file__).parent.parent / "models" / "sequence_classifier_meta.pkl"


class GestureLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 63,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.lstm   = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 63)
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        return self.linear(out[:, -1])  # (batch, num_classes)


class SequenceClassifier:
    """
    Wraps GestureLSTM for stateless per-sequence inference.

    Thread-safety: predict() and _raw_probs() are stateless — safe to call from
    the main frame loop without locking (model is in eval mode, no gradient tracking).
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        meta_path: Path  = DEFAULT_META_PATH,
    ) -> None:
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Sequence model not found at {model_path} / {meta_path}. "
                "Record sequences and train via Studio → Dynamic Gestures → Train."
            )
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        self._label_encoder = meta["label_encoder"]
        self._seq_len        = int(meta["seq_len"])
        n_classes = len(self._label_encoder.classes_)
        self._model = GestureLSTM(num_classes=n_classes)
        self._model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self._model.eval()

    def predict(self, frames: np.ndarray) -> tuple[str, float]:
        """
        Classify a sequence of landmark frames.

        Args:
            frames: float32 ndarray of shape (T, 63). T need not equal seq_len —
                    resampled internally before inference.

        Returns:
            (label_string, confidence_float)
        """
        probs = self._raw_probs(frames)
        idx   = int(np.argmax(probs))
        label = self._label_encoder.inverse_transform([idx])[0]
        return label, float(probs[idx])

    def _raw_probs(self, frames: np.ndarray) -> np.ndarray:
        """Return softmax probability vector (n_classes,) for EMA early-exit."""
        resampled = _resample(frames, self._seq_len)          # (seq_len, 63)
        tensor    = torch.from_numpy(resampled).unsqueeze(0)  # (1, seq_len, 63)
        with torch.no_grad():
            logits = self._model(tensor)                       # (1, n_classes)
            return torch.softmax(logits, dim=-1)[0].numpy()   # (n_classes,)

    @property
    def classes(self) -> list[str]:
        return list(self._label_encoder.classes_)


def _resample(frames: np.ndarray, target_len: int) -> np.ndarray:
    """
    Linearly interpolate a (T, F) frame array to (target_len, F).

    Applied independently per feature dimension. Safe for T == target_len
    (returns a float32 copy).
    """
    T, F = frames.shape
    if T == target_len:
        return frames.astype(np.float32)
    src_idx = np.linspace(0, T - 1, T)
    dst_idx = np.linspace(0, T - 1, target_len)
    out = np.stack(
        [np.interp(dst_idx, src_idx, frames[:, f]) for f in range(F)],
        axis=1,
    )
    return out.astype(np.float32)
