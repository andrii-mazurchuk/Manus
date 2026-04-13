"""
Gesture classifier training logic.

Importable from the API layer — no argparse, no sys.exit, no CLI side-effects.
All errors are raised as ValueError so callers can convert them to HTTP responses.
"""
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

DEFAULT_CSV   = Path("src/data/gestures.csv")
DEFAULT_MODEL = Path("src/models/classifier.pkl")
TARGET_ACCURACY = 0.88


def validate_dataset(csv_path: Path = DEFAULT_CSV) -> None:
    """
    Quick pre-flight check — raise ValueError if the dataset cannot support training.
    Reads only the label column so it is fast even for large CSVs.
    """
    if not csv_path.exists():
        raise ValueError("No training data found. Capture or upload a dataset first.")
    df = pd.read_csv(csv_path, usecols=["label"])
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError(
            f"Need at least 2 gesture classes to train. Found: {len(counts)}."
        )
    if len(df) < 10:
        raise ValueError(
            f"Need at least 10 samples to train. Found: {len(df)}."
        )


def run_training(
    csv_path: Path = DEFAULT_CSV,
    out_path: Path = DEFAULT_MODEL,
    *,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """
    Train the gesture classifier and return a metrics dict.

    Args:
        csv_path:    Path to gestures.csv.
        out_path:    Where to write classifier.pkl.
        progress_cb: Optional callable(str) called with progress messages.

    Returns:
        {
            "model":     "MLP",
            "accuracy":  0.942,
            "cv_score":  0.938,
            "per_class": {"STOP": 0.96, "PLAY": 0.93, ...},
        }

    Raises:
        ValueError  — dataset missing / too small / wrong shape.
        Any sklearn exception propagates directly.
    """
    def _progress(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    _progress("Loading data…")
    X, y, le = _load_data(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        ),
    }

    results: dict[str, dict] = {}
    for name, model in models.items():
        _progress(f"Training {name} (CV 5-fold)…")
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
        )
        model.fit(X_train, y_train)
        report = classification_report(
            y_test,
            model.predict(X_test),
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        results[name] = {
            "model":    model,
            "cv_mean":  float(cv_scores.mean()),
            "test_acc": float(model.score(X_test, y_test)),
            "report":   report,
        }

    winner_name = max(results, key=lambda n: results[n]["cv_mean"])
    winner = results[winner_name]

    _progress("Saving model…")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"model": winner["model"], "label_encoder": le}, f)

    per_class = {
        lbl: round(winner["report"][lbl]["recall"], 4)
        for lbl in le.classes_
        if lbl in winner["report"]
    }

    return {
        "model":     winner_name,
        "accuracy":  round(winner["test_acc"], 4),
        "cv_score":  round(winner["cv_mean"], 4),
        "per_class": per_class,
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load and validate the full feature matrix from CSV."""
    if not csv_path.exists():
        raise ValueError(f"No training data found at {csv_path}.")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV missing 'label' column.")

    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError(
            f"Need at least 2 gesture classes to train. Found: {len(counts)}."
        )
    if len(df) < 10:
        raise ValueError(
            f"Need at least 10 samples to train. Found: {len(df)}."
        )

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    return X, y, le
