"""
Two-hand gesture classifier inference wrapper.

Loads the trained model from models/classifier_two_hand.pkl and exposes
predict() for classifying gestures from one or two hands.

The classifier accepts an 84-float input vector built by
normalize_two_hand_landmarks(): the primary hand's 42 floats followed by the
secondary hand's 42 floats (zero-padded if absent).  This zero-masking design
means the same model handles both single-hand gestures and two-hand gestures
— the model learns that a zero block in the secondary slot means "one hand
only".

Usage:
    from src.core.two_hand_classifier import TwoHandGestureClassifier

    clf = TwoHandGestureClassifier()                       # loads model once
    label, conf = clf.predict(primary_lms)                 # one hand
    label, conf = clf.predict(primary_lms, secondary_lms)  # two hands
"""

import pickle
from pathlib import Path

import numpy as np

from .normalizer import normalize_two_hand_landmarks

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent / "models" / "classifier_two_hand.pkl"
)


class TwoHandGestureClassifier:
    """
    Wraps the sklearn model + label encoder for two-hand gesture classification.

    Thread-safety: predict() is stateless after __init__ — safe to call from
    a single inference loop without locking.

    If the model file does not exist, instantiation raises FileNotFoundError.
    Callers in main.py should catch this and skip registration of the
    two-hand classifier until the model has been trained via Studio.
    """

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Two-hand model not found at {model_path}. "
                "Capture two-hand training data and train via Studio "
                "(Training tab → model type: two-hand)."
            )
        with open(model_path, "rb") as f:
            payload = pickle.load(f)

        self._model = payload["model"]
        self._le = payload["label_encoder"]

    def predict(self, primary_landmarks, secondary_landmarks=None) -> tuple[str, float]:
        """
        Classify a gesture from one or two hands.

        Args:
            primary_landmarks:   21 landmark objects with .x and .y
                                 (right hand by convention, or the only hand).
            secondary_landmarks: 21 landmark objects with .x and .y, or None.
                                 When None, the secondary slot is zero-padded.

        Returns:
            (label, confidence) where label is a GestureToken string value
            and confidence is the highest class probability (0.0 – 1.0).
        """
        features = normalize_two_hand_landmarks(
            primary_landmarks, secondary_landmarks
        ).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        class_idx = int(np.argmax(proba))
        label: str = self._le.inverse_transform([class_idx])[0]
        confidence: float = float(proba[class_idx])
        return label, confidence

    @property
    def classes(self) -> list[str]:
        """All gesture token labels the model knows, in encoder order."""
        return list(self._le.classes_)
