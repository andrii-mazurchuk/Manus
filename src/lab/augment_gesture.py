"""
CLI: capture a gesture template and generate synthetic training data.

Reduces data collection from ~150 manual samples to 3-5 seconds of holding
a pose.  Augmented rows are written to the same CSV and .npy directories as
real samples, so train.py picks them up without any changes.

Usage examples:
    # Capture new gesture + generate 500 synthetic samples
    uv run src/lab/augment_gesture.py --label WAVE --samples 500

    # Use an existing template (skip webcam)
    uv run src/lab/augment_gesture.py --label WAVE --samples 500 --no-capture

    # Reproducible output
    uv run src/lab/augment_gesture.py --label PINCH --samples 1000 --seed 42

    # Tune augmentation parameters
    uv run src/lab/augment_gesture.py --label SNAP \\
        --rotation-range 20 --noise-sigma 0.015 --extension-range 0.08

After running, retrain:
    uv run scripts/train.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# Make src/ importable when running as a top-level script
sys.path.insert(0, str(Path(__file__).parent.parent))

from lab.augment import AugmentationEngine
from lab.gesture_template import GestureTemplateCapture

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC_DATA = Path(__file__).parent.parent / "data"

DEFAULT_CSV       = _SRC_DATA / "gestures.csv"
DEFAULT_GESTURES  = _SRC_DATA / "gestures"

CSV_HEADER = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _count_label_rows(csv_path: Path, label: str) -> int:
    """Count existing rows for `label` in the CSV.  Returns 0 if file missing."""
    if not csv_path.exists():
        return 0
    count = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)   # skip header
        for row in reader:
            if row and row[0] == label:
                count += 1
    return count


def _append_to_csv(
    csv_path: Path, label: str, rows: np.ndarray
) -> None:
    """
    Append synthetic rows to the gestures CSV.

    Matches data_collector.py save_sample() exactly:
    header written only if file is empty or missing; rows are appended.
    """
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([label] + row.tolist())


# ---------------------------------------------------------------------------
# NPY writer
# ---------------------------------------------------------------------------

def _save_npy_files(label_dir: Path, rows: np.ndarray) -> None:
    """
    Save each synthetic row as an individual .npy file in label_dir.

    Filenames follow the data_collector.py convention: {timestamp_ms}.npy.
    A collision guard increments the timestamp if a file already exists.
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    base_ts = int(time.time() * 1000)

    for i, row in enumerate(rows):
        ts = base_ts + i
        while (label_dir / f"{ts}.npy").exists():
            ts += 1
        np.save(label_dir / f"{ts}.npy", row.astype(np.float32))


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic gesture training data from a few webcam captures.\n"
            "Appends augmented rows to src/data/gestures.csv and saves .npy files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--label", required=True,
        help="Gesture token name (e.g. WAVE, PINCH, STOP).  Any string is valid.",
    )
    parser.add_argument(
        "--samples", type=int, default=500,
        help="Number of synthetic rows to generate (default: 500).",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="OpenCV camera index (default: 0).",
    )
    parser.add_argument(
        "--capture-frames", type=int, default=5, dest="capture_frames",
        help="Webcam frames to average into the template (default: 5).",
    )
    parser.add_argument(
        "--no-capture", action="store_true", dest="no_capture",
        help=(
            "Skip webcam; load an existing template from "
            "src/lab/templates/<label>.npy instead."
        ),
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"Output CSV path (default: {DEFAULT_CSV}).",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain classifier automatically after augmentation.",
    )

    # Augmentation tuning
    aug = parser.add_argument_group("augmentation parameters")
    aug.add_argument(
        "--rotation-range", type=float, default=30.0, dest="rotation_range",
        help="Max +/- degrees for whole-hand rotation (default: 30).",
    )
    aug.add_argument(
        "--noise-sigma", type=float, default=0.02, dest="noise_sigma",
        help="Std-dev of per-landmark Gaussian noise (default: 0.02).",
    )
    aug.add_argument(
        "--extension-range", type=float, default=0.10, dest="extension_range",
        help="Max +/- fraction for finger extension/curl perturbation (default: 0.10).",
    )
    aug.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for reproducible output (default: random).",
    )

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    label = args.label.upper()

    # ── Step 1: get template ────────────────────────────────────────────────
    print(f"\nManus Augmentation - label: {label}")
    print("-" * 50)

    capturer = GestureTemplateCapture(
        camera_index=args.camera,
        n_frames=args.capture_frames,
    )

    if args.no_capture:
        try:
            template = capturer.load(label)
            print(f"  Template loaded from src/lab/templates/{label}.npy")
        except FileNotFoundError as exc:
            sys.exit(f"ERROR: {exc}")
    else:
        print("  Opening webcam - hold your gesture and press SPACE to capture.")
        print("  Press Q or ESC to abort.\n")
        try:
            with capturer:
                template = capturer.capture(label)
        except KeyboardInterrupt as exc:
            sys.exit(f"\nAborted: {exc}")

    # ── Step 2: augment ─────────────────────────────────────────────────────
    print(f"\n  Generating {args.samples} synthetic samples...")

    engine = AugmentationEngine(
        rotation_range=args.rotation_range,
        noise_sigma=args.noise_sigma,
        extension_range=args.extension_range,
        seed=args.seed,
    )

    try:
        synthetic = engine.generate(template, args.samples)
    except ValueError as exc:
        sys.exit(f"ERROR: {exc}")

    # ── Step 3: write CSV ───────────────────────────────────────────────────
    count_before = _count_label_rows(args.csv, label)
    _append_to_csv(args.csv, label, synthetic)
    count_after = count_before + args.samples

    # ── Step 4: write NPY files ─────────────────────────────────────────────
    label_dir = DEFAULT_GESTURES / label
    _save_npy_files(label_dir, synthetic)

    # ── Step 5: summary ─────────────────────────────────────────────────────
    print(f"\n{'-' * 50}")
    print(f"  Augmentation complete")
    print(f"  Label         : {label}")
    print(f"  Rows added    : {args.samples}")
    print(f"  CSV rows      : {count_before} -> {count_after}   ({args.csv})")
    print(f"  NPY files     : {label_dir}  (+{args.samples})")
    if args.seed is not None:
        print(f"  Seed          : {args.seed}  (reproducible)")
    if args.retrain:
        print("\n  Retraining classifier...")
        import importlib.util

        train_path = Path(__file__).parent.parent.parent / "scripts" / "train.py"
        sys.argv = ["train.py"]
        spec = importlib.util.spec_from_file_location("train", train_path)
        if spec is None or spec.loader is None:
            sys.exit(f"ERROR: could not load train module from {train_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
    else:
        print(f"\n  Next step: uv run scripts/train.py")


if __name__ == "__main__":
    main()
