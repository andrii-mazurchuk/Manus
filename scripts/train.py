"""
Train a gesture classifier from src/data/gestures.csv.

Steps:
  1. Load CSV produced by src/data/extract_landmarks.py
  2. Inspect class balance -- print counts and warn on imbalance
  3. Encode labels, split train/test
  4. Train RandomForest and MLP, cross-validate both
  5. Pick the higher cross-val model
  6. Evaluate on held-out test set (classification report)
  7. Serialize winning model + label encoder to src/models/classifier.pkl

Target: >= 88% accuracy (per PLAN.md checkpoint requirement).

Usage:
    python train.py
    python train.py --csv src/data/gestures.csv --out src/models/classifier.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

DEFAULT_CSV   = Path(__file__).parent.parent / "src" / "data" / "gestures.csv"
DEFAULT_MODEL = Path(__file__).parent.parent / "src" / "models" / "classifier.pkl"

TARGET_ACCURACY = 0.88


def load_data(csv_path: Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        sys.exit("ERROR: CSV missing 'label' column. Re-run src/data/extract_landmarks.py.")

    expected_cols = 43  # 1 label + 42 floats
    if len(df.columns) != expected_cols:
        print(
            f"WARNING: expected {expected_cols} columns, got {len(df.columns)}. "
            "Check extract_landmarks output."
        )

    print(f"  Total rows: {len(df)}")

    counts = df["label"].value_counts().sort_index()
    print("\n  Samples per label:")
    for label, count in counts.items():
        bar = "#" * (count // 50)
        print(f"    {label:10s}: {count:5d}  {bar}")

    min_count = counts.min()
    max_count = counts.max()
    if max_count / min_count > 3:
        print(
            f"\n  WARNING: class imbalance detected "
            f"(min={min_count}, max={max_count}, ratio={max_count/min_count:.1f}x). "
            "Consider collecting more samples for underrepresented classes."
        )

    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].values.astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)

    print(f"\n  Feature shape : {X.shape}")
    print(f"  Classes       : {list(le.classes_)}")
    return X, y, le


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    le: LabelEncoder,
) -> tuple[object, float]:
    """Train RandomForest and MLP, cross-validate, return (best_model, cv_score)."""

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
        print(f"\n-- {name} --------------------------------------------------")

        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"  CV accuracy : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  CV folds    : {[f'{s:.3f}' for s in cv_scores]}")

        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  Test accuracy: {test_acc:.4f}")
        print(
            classification_report(
                y_test,
                model.predict(X_test),
                target_names=le.classes_,
                zero_division=0,
            )
        )

        results[name] = {"model": model, "cv_mean": cv_scores.mean(), "test_acc": test_acc}

    winner_name = max(results, key=lambda n: results[n]["cv_mean"])
    winner = results[winner_name]
    print(f"\n-- Winner: {winner_name} (CV={winner['cv_mean']:.4f}, Test={winner['test_acc']:.4f})")

    if winner["test_acc"] < TARGET_ACCURACY:
        print(
            f"  WARNING: test accuracy {winner['test_acc']:.4f} is below target "
            f"{TARGET_ACCURACY:.2f}. "
            "Consider: more training data, additional gestures data collection, "
            "or tuning hyperparameters."
        )
    else:
        print(f"  Target accuracy {TARGET_ACCURACY:.2f} met.")

    return winner["model"], winner["cv_mean"]


def save_model(model: object, le: LabelEncoder, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "label_encoder": le}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Model saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train gesture classifier")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if not args.csv.exists():
        sys.exit(
            f"ERROR: {args.csv} not found. Run 'python src/data/extract_landmarks.py' first."
        )

    X, y, le = load_data(args.csv)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

    model, cv_score = train_and_evaluate(X_train, y_train, X_test, y_test, le)
    save_model(model, le, args.out)

    print("\nDone. Next step: python capture.py  (live webcam inference)")


if __name__ == "__main__":
    main()
