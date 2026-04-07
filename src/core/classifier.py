"""
Gesture classifier inference wrapper.

Loads the trained model from models/classifier.pkl and exposes a single
predict() function used by the live pipeline.

Usage:
    from core.classifier import GestureClassifier

    clf = GestureClassifier()                        # loads model once
    label, confidence = clf.predict(landmarks)       # landmarks: 21 NormalizedLandmark
"""

import pickle
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "classifier.pkl"


def _normalize(landmarks) -> np.ndarray:
    """
    Translate all landmarks to wrist origin (landmark 0), scale to [-1, 1].

    landmarks: list/sequence of 21 objects with .x and .y attributes
               (mediapipe NormalizedLandmark from either solutions or tasks API).
    Returns a float32 array of shape (1, 42) ready for sklearn predict.
    """
    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)  # (21, 2)
    coords -= coords[0]
    scale = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale
    return coords.flatten().reshape(1, -1)  # (1, 42)


class GestureClassifier:
    """
    Wraps the sklearn model + label encoder loaded from classifier.pkl.

    Thread-safety: predict() is stateless after __init__ — safe to call
    from a single inference loop without locking.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run 'uv run train.py' to train and save the model."
            )
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        self._model = payload["model"]
        self._le = payload["label_encoder"]

    def predict(self, landmarks) -> tuple[str, float]:
        """
        Classify a single hand pose.

        Args:
            landmarks: 21 landmark objects with .x and .y (from MediaPipe).

        Returns:
            (label, confidence) — label is one of the token constants
            (STOP, PLAY, UP, DOWN, CONFIRM, CANCEL, MODE, CUSTOM);
            confidence is the highest class probability (0.0 - 1.0).
        """
        features = _normalize(landmarks)
        proba = self._model.predict_proba(features)[0]        # shape: (n_classes,)
        class_idx = int(np.argmax(proba))
        label: str = self._le.inverse_transform([class_idx])[0]
        confidence: float = float(proba[class_idx])
        return label, confidence

    @property
    def classes(self) -> list[str]:
        """All gesture token labels the model knows, in encoder order."""
        return list(self._le.classes_)
