"""
Synthetic landmark augmentation engine.

Generates synthetic training rows from a single "template" landmark array by
applying a chain of geometric transforms in normalized hand-landmark space.

Because MediaPipe has already abstracted away lighting, skin tone, and camera
distance, the sim-to-real gap is small: all transforms operate on the relative
joint geometry that the classifier actually uses.

Usage:
    from lab.augment import AugmentationEngine
    import numpy as np

    template = np.load("src/lab/templates/WAVE.npy")   # (42,) float32
    engine   = AugmentationEngine(seed=42)
    synthetic = engine.generate(template, n_samples=500)  # (500, 42) float32
"""

from __future__ import annotations

import numpy as np
try:
    from src.core.normalizer import normalize_coords   # API import context
except ImportError:
    from core.normalizer import normalize_coords       # script / direct context

# ---------------------------------------------------------------------------
# Finger anatomy — MediaPipe landmark indices
# ---------------------------------------------------------------------------
# Wrist: 0
# Thumb: CMC=1, MCP=2, IP=3, Tip=4
# Index: MCP=5, PIP=6, DIP=7, Tip=8
# Middle: MCP=9, PIP=10, DIP=11, Tip=12
# Ring: MCP=13, PIP=14, DIP=15, Tip=16
# Pinky: MCP=17, PIP=18, DIP=19, Tip=20

FINGER_GROUPS: dict[str, dict] = {
    "thumb":  {"mcp": 2,  "joints": [3, 4]},
    "index":  {"mcp": 5,  "joints": [6, 7, 8]},
    "middle": {"mcp": 9,  "joints": [10, 11, 12]},
    "ring":   {"mcp": 13, "joints": [14, 15, 16]},
    "pinky":  {"mcp": 17, "joints": [18, 19, 20]},
}


# ---------------------------------------------------------------------------
# Private transform helpers  (all operate on (21, 2) float32 arrays)
# ---------------------------------------------------------------------------

def _renormalize(coords: np.ndarray) -> np.ndarray:
    """
    Translate wrist (index 0) to origin, scale all landmarks to [-1, 1].

    This is the canonical normalization used throughout the project
    (data_collector.py, extract_landmarks.py, classifier.py).  It MUST be
    called last on every synthetic sample to keep the feature space identical
    to real training rows.

    Input / output shape: (21, 2) float32.
    """
    return normalize_coords(coords).reshape(21, 2)


def _rotate_2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate all 21 landmarks around the wrist origin by angle_deg degrees.

    Because _renormalize has already placed the wrist at (0, 0), this is a
    pure rotation about the wrist — equivalent to tilting the hand in 2D.
    """
    theta = np.deg2rad(angle_deg)
    c, s = float(np.cos(theta)), float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (coords @ R.T).astype(np.float32)


def _add_noise(
    coords: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Add independent Gaussian noise N(0, sigma) to every landmark coordinate.

    sigma ≈ 0.02 mimics MediaPipe jitter and natural hand tremor at the scale
    of the normalized [-1, 1] coordinate space.
    """
    noise = rng.normal(0.0, sigma, size=coords.shape).astype(np.float32)
    return (coords + noise).astype(np.float32)


def _perturb_extension(
    coords: np.ndarray,
    finger_name: str,
    magnitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate a finger curling or extending slightly.

    For the named finger:
      1. Compute a unit axis vector from MCP → first proximal joint.
      2. Shift each distal joint along that axis by a random scalar, with
         weights that increase toward the tip (tip moves most).

    Skips silently if the finger is degenerate (all joints at the same point).
    """
    group = FINGER_GROUPS[finger_name]
    mcp_idx = group["mcp"]
    joints  = group["joints"]

    p_mcp   = coords[mcp_idx]
    p_first = coords[joints[0]]
    raw     = p_first - p_mcp
    norm    = float(np.linalg.norm(raw))

    if norm < 1e-6:
        return coords  # degenerate finger — skip

    axis  = (raw / norm).astype(np.float32)
    delta = float(rng.uniform(-magnitude, magnitude))

    n       = len(joints)
    weights = np.linspace(0.5, 1.0, n, dtype=np.float32)

    result = coords.copy()
    for j_idx, joint in enumerate(joints):
        result[joint] = result[joint] + delta * axis * weights[j_idx]

    return result.astype(np.float32)


def _vary_scale(
    coords: np.ndarray, magnitude: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Multiply all landmark coordinates by a random factor near 1.0.

    Applied before _renormalize so that the random scale interacts with the
    subsequent noise term, producing mild magnitude diversity after
    renormalization.
    """
    factor = float(1.0 + rng.uniform(-magnitude, magnitude))
    return (coords * factor).astype(np.float32)


def _vary_spread(
    coords: np.ndarray,
    finger_name: str,
    angle_deg_range: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Rotate a finger's distal joints around their MCP as a fixed pivot.

    Simulates lateral abduction / adduction (spreading / squeezing fingers).
    The MCP itself does not move.
    """
    group   = FINGER_GROUPS[finger_name]
    mcp_idx = group["mcp"]
    joints  = group["joints"]

    theta = np.deg2rad(float(rng.uniform(-angle_deg_range, angle_deg_range)))
    c, s  = float(np.cos(theta)), float(np.sin(theta))
    R     = np.array([[c, -s], [s, c]], dtype=np.float32)

    pivot  = coords[mcp_idx]
    result = coords.copy()

    for joint in joints:
        local          = coords[joint] - pivot
        result[joint]  = (local @ R.T) + pivot

    return result.astype(np.float32)


def _flip_horizontal(coords: np.ndarray) -> np.ndarray:
    """
    Mirror the hand by negating all x coordinates.

    Produces a left-hand equivalent of a right-hand pose (or vice versa).
    Valid for gestures where mirroring preserves the gesture class (e.g. a
    symmetric open palm).  Disable via include_flip=False for asymmetric
    gestures.
    """
    result = coords.copy()
    result[:, 0] *= -1
    return result.astype(np.float32)


def _renormalize_two_hand(all84: np.ndarray) -> np.ndarray:
    """
    Renormalize an 84-float two-hand array after augmentation transforms.

    Contract: primary wrist ends up at (0, 0). Both halves are divided by
    the same scale (primary max-abs), preserving inter-hand geometry.
    Mirrors normalize_two_hand_landmarks() in src/core/normalizer.py.

    Args:
        all84: (84,) flat array [primary_42 | secondary_42].

    Returns:
        (84,) float32 renormalized array.
    """
    primary   = all84[:42].reshape(21, 2).copy()
    secondary = all84[42:].reshape(21, 2).copy()
    secondary_absent = np.all(all84[42:] == 0.0)

    wrist = primary[0].copy()
    primary -= wrist
    if not secondary_absent:
        secondary -= wrist

    scale = float(np.max(np.abs(primary)))
    if scale > 0:
        primary /= scale
        if not secondary_absent:
            secondary /= scale

    if secondary_absent:
        secondary = np.zeros((21, 2), dtype=np.float32)

    return np.concatenate([primary.flatten(), secondary.flatten()]).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AugmentationEngine:
    """
    Generates synthetic hand-landmark training samples from a single template.

    All transforms are applied in a fixed order per sample, followed by
    mandatory renormalization to keep synthetic rows in the same feature space
    as real training data.

    Thread-safety: generate() is stateful (advances the RNG).  Do not share
    an engine instance across threads without external locking.

    Args:
        rotation_range:  Max ± degrees for whole-hand 2D rotation.
        noise_sigma:     Std-dev of per-landmark Gaussian noise.
        extension_range: Max ± fraction for finger extension/curl perturbation.
        scale_range:     Max ± fraction for global scale variation.
        spread_range:    Max ± degrees for lateral finger spread.
        include_flip:    Whether to randomly mirror the hand (x → -x).
        seed:            RNG seed for reproducibility (None = OS entropy).
    """

    def __init__(
        self,
        rotation_range: float = 30.0,
        noise_sigma: float = 0.02,
        extension_range: float = 0.10,
        scale_range: float = 0.05,
        spread_range: float = 5.0,
        include_flip: bool = True,
        seed: int | None = None,
    ) -> None:
        self.rotation_range  = rotation_range
        self.noise_sigma     = noise_sigma
        self.extension_range = extension_range
        self.scale_range     = scale_range
        self.spread_range    = spread_range
        self.include_flip    = include_flip
        self._rng = np.random.default_rng(seed)

    def generate(self, template: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate n_samples synthetic landmark rows from the given template.

        Args:
            template:  (42,) float32 normalized landmark array
                       [x0, y0, x1, y1, ..., x20, y20].
            n_samples: Number of synthetic rows to produce.

        Returns:
            (n_samples, 42) float32 array.  Each row is an independently
            augmented variant of the template, renormalized to the project's
            canonical feature space.

        Raises:
            ValueError: If template shape is wrong or template is degenerate.
        """
        if template.shape != (42,):
            raise ValueError(
                f"Expected template shape (42,), got {template.shape}."
            )

        pts = template.reshape(21, 2)
        if np.max(np.abs(pts[1:])) < 1e-6:
            raise ValueError(
                "Template is degenerate: all landmarks are at the wrist origin. "
                "The template was likely captured with no hand in frame."
            )

        rng     = self._rng
        results = np.empty((n_samples, 42), dtype=np.float32)

        for i in range(n_samples):
            coords = pts.copy()

            # 1. Rotation (always)
            angle  = float(rng.uniform(-self.rotation_range, self.rotation_range))
            coords = _rotate_2d(coords, angle)

            # 2. Gaussian noise (always)
            coords = _add_noise(coords, self.noise_sigma, rng)

            # 3. Per-finger extension perturbation (always, independent per finger)
            for finger_name in FINGER_GROUPS:
                coords = _perturb_extension(
                    coords, finger_name, self.extension_range, rng
                )

            # 4. Global scale variation (70% probability)
            if rng.random() < 0.70:
                coords = _vary_scale(coords, self.scale_range, rng)

            # 5. Lateral spread variation (50% probability per finger)
            for finger_name in FINGER_GROUPS:
                if rng.random() < 0.50:
                    coords = _vary_spread(
                        coords, finger_name, self.spread_range, rng
                    )

            # 6. Horizontal flip (30% probability, if enabled)
            if self.include_flip and rng.random() < 0.30:
                coords = _flip_horizontal(coords)

            # 7. Renormalize (mandatory — keeps wrist at origin, values in [-1,1])
            coords = _renormalize(coords)

            results[i] = coords.flatten()

        return results


class TwoHandAugmentationEngine:
    """
    Generates synthetic two-hand training samples from a single 84-float template.

    The 84-float vector is [primary_42 | secondary_42] where both halves share
    the primary wrist's coordinate frame. Transforms are applied coherently:
    rotation and global scale use the same random value for both halves so that
    inter-hand geometry (distance, relative angle) is preserved. Finger
    extension and spread are independent per hand.

    If the secondary half of the template is all zeros (single-hand capture),
    transforms are applied only to the primary half and the secondary remains
    zero-padded in all output rows.

    Thread-safety: generate() is stateful (advances the RNG). Do not share an
    engine instance across threads without external locking.

    Args:
        rotation_range:  Max ± degrees for whole-hand 2D rotation.
        noise_sigma:     Std-dev of per-landmark Gaussian noise.
        extension_range: Max ± fraction for finger extension/curl perturbation.
        scale_range:     Max ± fraction for global scale variation.
        spread_range:    Max ± degrees for lateral finger spread.
        include_flip:    Whether to randomly mirror both hands (x → -x).
        seed:            RNG seed for reproducibility (None = OS entropy).
    """

    def __init__(
        self,
        rotation_range: float = 30.0,
        noise_sigma: float = 0.02,
        extension_range: float = 0.10,
        scale_range: float = 0.05,
        spread_range: float = 5.0,
        include_flip: bool = True,
        seed: int | None = None,
    ) -> None:
        self.rotation_range  = rotation_range
        self.noise_sigma     = noise_sigma
        self.extension_range = extension_range
        self.scale_range     = scale_range
        self.spread_range    = spread_range
        self.include_flip    = include_flip
        self._rng = np.random.default_rng(seed)

    def generate(self, template: np.ndarray, n_samples: int) -> np.ndarray:
        """
        Generate n_samples augmented rows from an 84-float two-hand template.

        Args:
            template:  (84,) float32 — [primary_42 | secondary_42],
                       both in primary-wrist-origin, primary-scale frame.
            n_samples: Number of synthetic rows to produce.

        Returns:
            (n_samples, 84) float32 array. Each row is an independently
            augmented variant of the template, renormalized so the primary
            wrist is at the origin and both halves share the same scale.

        Raises:
            ValueError: if template.shape != (84,) or primary half is degenerate.
        """
        if template.shape != (84,):
            raise ValueError(
                f"Expected template shape (84,), got {template.shape}."
            )

        primary_pts = template[:42].reshape(21, 2)
        if np.max(np.abs(primary_pts[1:])) < 1e-6:
            raise ValueError(
                "Template is degenerate: primary hand has no landmark spread. "
                "The template was likely captured with no hand in frame."
            )

        secondary_absent = np.all(template[42:] == 0.0)
        rng     = self._rng
        results = np.empty((n_samples, 84), dtype=np.float32)

        for i in range(n_samples):
            primary   = template[:42].reshape(21, 2).copy()
            secondary = template[42:].reshape(21, 2).copy()

            # 1. Rotation — SAME angle for both halves (preserves inter-hand angle)
            angle   = float(rng.uniform(-self.rotation_range, self.rotation_range))
            primary = _rotate_2d(primary, angle)
            if not secondary_absent:
                secondary = _rotate_2d(secondary, angle)

            # 2. Noise — independent per coordinate on each half
            primary = _add_noise(primary, self.noise_sigma, rng)
            if not secondary_absent:
                secondary = _add_noise(secondary, self.noise_sigma, rng)

            # 3. Finger extension — independent per hand, per finger
            for finger_name in FINGER_GROUPS:
                primary = _perturb_extension(primary, finger_name, self.extension_range, rng)
            if not secondary_absent:
                for finger_name in FINGER_GROUPS:
                    secondary = _perturb_extension(secondary, finger_name, self.extension_range, rng)

            # 4. Global scale — SAME factor to both halves (70% probability)
            #    Cannot use _vary_scale() here: it generates its own rng draw internally,
            #    so calling it twice would produce two independent factors, breaking
            #    inter-hand distance. Draw once and multiply both halves manually.
            if rng.random() < 0.70:
                factor = float(1.0 + rng.uniform(-self.scale_range, self.scale_range))
                primary   = (primary   * factor).astype(np.float32)
                if not secondary_absent:
                    secondary = (secondary * factor).astype(np.float32)

            # 5. Lateral spread — independent per hand, per finger (50% per finger)
            for finger_name in FINGER_GROUPS:
                if rng.random() < 0.50:
                    primary = _vary_spread(primary, finger_name, self.spread_range, rng)
            if not secondary_absent:
                for finger_name in FINGER_GROUPS:
                    if rng.random() < 0.50:
                        secondary = _vary_spread(secondary, finger_name, self.spread_range, rng)

            # 6. Horizontal flip — SAME decision for both (30% probability, if enabled)
            if self.include_flip and rng.random() < 0.30:
                primary   = _flip_horizontal(primary)
                if not secondary_absent:
                    secondary = _flip_horizontal(secondary)

            # 7. Two-hand renormalize (mandatory — primary wrist at origin, shared scale)
            combined   = np.concatenate([primary.flatten(), secondary.flatten()])
            results[i] = _renormalize_two_hand(combined)

        return results
