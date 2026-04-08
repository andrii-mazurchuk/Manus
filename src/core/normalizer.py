"""
Shared landmark normalization utilities and hand skeleton constant.

Single source of truth used by the entire pipeline:
  - scripts/data_collector.py   (saving training samples)
  - src/data/extract_landmarks.py (processing dataset images)
  - src/core/classifier.py      (inference pre-processing)
  - src/lab/gesture_template.py (template capture + averaging)
  - src/lab/augment.py          (synthetic augmentation)

Normalization contract (identical to INTERFACE_CONTRACT.md):
  1. Translate: subtract landmark 0 (wrist) → wrist is at (0, 0)
  2. Scale: divide by max(abs(coords)) → all values in [-1, 1]
"""

from __future__ import annotations

import numpy as np


# Hand skeleton connectivity — 21 landmark pairs forming the hand skeleton.
# Indices match MediaPipe Hands:
#   0 = Wrist
#   1–4  = Thumb (CMC, MCP, IP, Tip)
#   5–8  = Index (MCP, PIP, DIP, Tip)
#   9–12 = Middle (MCP, PIP, DIP, Tip)
#   13–16 = Ring (MCP, PIP, DIP, Tip)
#   17–20 = Pinky (MCP, PIP, DIP, Tip)
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),   # thumb
    (0, 5),  (5, 6),  (6, 7),  (7, 8),   # index
    (5, 9),  (9, 10), (10, 11),(11, 12),  # middle
    (9, 13), (13, 14),(14, 15),(15, 16),  # ring
    (13, 17),(17, 18),(18, 19),(19, 20),  # pinky
    (0, 17),                              # palm base
]


def normalize_landmarks(landmarks) -> np.ndarray:
    """
    Convert a MediaPipe NormalizedLandmark list to a flat normalized array.

    Args:
        landmarks: Sequence of 21 objects with .x and .y float attributes
                   (mediapipe NormalizedLandmark, from Tasks or Solutions API).

    Returns:
        float32 ndarray of shape (42,): [x0, y0, x1, y1, ..., x20, y20]
        Wrist (index 0) is at the origin; all values are in [-1, 1].
    """
    coords = np.array([[lm.x, lm.y] for lm in landmarks], dtype=np.float32)
    return normalize_coords(coords)


def normalize_coords(coords: np.ndarray) -> np.ndarray:
    """
    Normalize a (21, 2) landmark coordinate array.

    Applies the same two-step contract as normalize_landmarks():
      1. Translate so wrist (coords[0]) is at the origin.
      2. Scale so max(abs(coords)) == 1.0 → values in [-1, 1].

    Args:
        coords: float32 ndarray of shape (21, 2) — raw or partially
                transformed landmark coordinates.

    Returns:
        float32 ndarray of shape (42,) — flat, normalized.
    """
    coords = coords.copy()
    coords -= coords[0]                   # translate: wrist → origin
    scale = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale                   # scale to [-1, 1]
    return coords.flatten().astype(np.float32)
