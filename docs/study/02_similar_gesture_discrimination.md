# Separating Similar Gestures — Training Strategy

## The Problem

Two gestures that share most of their landmark structure confuse the model.
Classic example: **index up** vs **L-shape** (index up + thumb extended sideways).

```
Landmark indices (MediaPipe):
  0        = Wrist
  1,2,3,4  = Thumb  (CMC → MCP → IP → Tip)
  5,6,7,8  = Index  (MCP → PIP → DIP → Tip)
  9..12    = Middle
  ...

Gesture A — index up only:
  Thumb tip (lm4):  x ≈  0.0,  y ≈  0.1   (folded, near palm centre)
  Index tip (lm8):  x ≈  0.0,  y ≈ -1.0   (pointing straight up)

Gesture B — L shape (index up + thumb out):
  Thumb tip (lm4):  x ≈  0.9,  y ≈  0.0   (extended sideways)
  Index tip (lm8):  x ≈  0.0,  y ≈ -1.0   (same as A)
```

The discriminating signal lives almost entirely in **landmarks 1–4 (thumb)**.
The index finger is nearly identical in both. The model has everything it needs
in the 42-float vector — thumb tip displacement is ~0.9 units, which is large.
The issues are always in the training data, not the model architecture.

---

## Problem 1 — Rotation Augmentation Blurs the Boundary

The augmenter (`src/lab/augment.py`) applies ±30° rotation uniformly.

```
Gesture B at 0°:          Gesture B at +30° rotation:
  index: pointing up        index: pointing upper-left
  thumb: pointing right     thumb: pointing upper-right

                    ↓ at +30° the thumb starts pointing
                      more "up and in" — it begins to
                      resemble a folded thumb position
                      in gesture A's training data
```

At ±30° the two gesture clouds in feature space get closer together.
The model learns a blurry decision boundary.

**Fix:** Reduce rotation range to ±10° for gestures where the discriminating feature
is the thumb direction. Modify the rotation range in `src/lab/augment.py` when
augmenting these specific gestures.

---

## Problem 2 — Imbalanced or Insufficient Samples

```
Feature space cross-section (thumb_x axis):

Gesture A samples:  |████████████████|          (dense, thumb_x ≈ 0)
Gesture B samples:              |████|           (sparse, thumb_x ≈ 0.9)

Decision boundary:                  ↑ pushed right by imbalance
                                    → A gets predicted even when thumb
                                      is partially extended
```

If one gesture has far more augmented samples than the other, the model biases
toward the majority class in the ambiguous region between them.

**Fix:** Match sample counts exactly. Use the same `--samples N` for both gestures
when running `scripts/bootstrap_augmented_dataset.py`.

---

## Problem 3 — Sloppy Template Capture

```
Clean template:          Sloppy template:
  thumb_x ≈ 0.90           thumb_x ≈ 0.55
  (clear L)                (halfway between A and B)
                               ↑ augmentation multiplies this
                                 ambiguity by 400x
```

If the captured template has the thumb only 60% extended, every augmented sample
inherits that ambiguity. The model never sees a crisp example of what the gesture
should look like at its canonical position.

**Fix:** When capturing templates via Studio, hold the gesture deliberately —
full thumb extension, index fully vertical. Exaggerate slightly. Recapture if the
template looks ambiguous.

---

## Practical Checklist

| What to do | Where |
|---|---|
| Reduce rotation augmentation to ±10° for both gestures | `src/lab/augment.py` rotation range |
| Ensure equal sample counts for both gestures | `--samples N` arg to bootstrap script |
| Recapture template with a crisp, exaggerated pose | Studio → capture template |
| After retraining, inspect confusion matrix | `scripts/train.py` output |
| If still confused, temporarily lower threshold and log raw confidence to find the boundary | `thresholds.json` |

---

## General Rule

When two gestures share a large portion of their landmark structure, ask:

1. **Which landmarks differ?** (Here: lm1–lm4, the thumb)
2. **Does augmentation rotate those landmarks toward the other gesture's space?** (±30° does for L-shape)
3. **Is training data balanced and are templates clean?**

The 42-float vector is almost always expressive enough. The failure is in the data pipeline,
not the feature space.
