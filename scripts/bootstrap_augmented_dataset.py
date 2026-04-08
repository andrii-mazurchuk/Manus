"""
Bootstrap a full gesture dataset using template capture + augmentation.

This is the recommended low-manual workflow:
  1) Capture one short template per gesture token
  2) Generate many synthetic samples per token
  3) Optionally retrain classifier at the end

Usage:
    uv run scripts/bootstrap_augmented_dataset.py
    uv run scripts/bootstrap_augmented_dataset.py --samples 300 --retrain
    uv run scripts/bootstrap_augmented_dataset.py --no-capture --retrain
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lab.augment_gesture import _retrain_model, generate_for_label

TOKENS = ["STOP", "PLAY", "UP", "DOWN", "CONFIRM", "CANCEL", "MODE", "CUSTOM"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a full training dataset from templates + synthetic augmentation "
            "for all standard gesture tokens."
        )
    )
    parser.add_argument("--samples", type=int, default=400, help="Synthetic samples per token.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--capture-frames",
        type=int,
        default=5,
        dest="capture_frames",
        help="Frames averaged for each template capture.",
    )
    parser.add_argument(
        "--no-capture",
        action="store_true",
        help="Use existing templates from src/lab/templates instead of webcam capture.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).parent.parent / "src" / "data" / "gestures.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain classifier automatically after dataset generation.",
    )
    parser.add_argument("--rotation-range", type=float, default=30.0, dest="rotation_range")
    parser.add_argument("--noise-sigma", type=float, default=0.02, dest="noise_sigma")
    parser.add_argument("--extension-range", type=float, default=0.10, dest="extension_range")
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="Optional deterministic seed base; token index is added per label.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    print("Manus augmented bootstrap")
    print("=========================")
    print(f"Tokens: {', '.join(TOKENS)}")
    print(f"Samples per token: {args.samples}")
    print(f"Capture mode: {'existing templates' if args.no_capture else 'webcam capture'}")

    total_added = 0
    for idx, label in enumerate(TOKENS):
        seed = None if args.seed_base is None else args.seed_base + idx
        summary = generate_for_label(
            label=label,
            samples=args.samples,
            camera=args.camera,
            capture_frames=args.capture_frames,
            no_capture=args.no_capture,
            csv_path=args.csv,
            rotation_range=args.rotation_range,
            noise_sigma=args.noise_sigma,
            extension_range=args.extension_range,
            seed=seed,
        )
        total_added += int(summary["rows_added"])

    print("\n=========================")
    print(f"Done. Total synthetic rows added: {total_added}")
    print(f"CSV: {args.csv}")

    if args.retrain:
        _retrain_model()
    else:
        print("Next step: uv run scripts/train.py")


if __name__ == "__main__":
    main()
