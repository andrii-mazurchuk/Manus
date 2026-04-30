# Gesture Sequence Classification — How It Works

## Overview

There are two distinct stages in the pipeline:

1. **Per-frame gesture classification** — ML model classifies each webcam frame independently
2. **Sequence recognition** — temporal pattern matching over the stream of classified tokens

Sequence quality is 100% dependent on single-gesture classifier quality. The sequence
recogniser has no ML of its own — it only sees the token stream the classifier produces.

---

## Stage 1 — Per-Frame Gesture Classification (ML)

```
Webcam frame
     │
     ▼
MediaPipe HandLandmarker
     │  detects 21 (x,y) keypoints per hand
     │
     ▼
  ┌──────────────────────────────────────────────────────┐
  │  normalize_landmarks()  ─ src/core/normalizer.py     │
  │                                                      │
  │  21 raw (x,y) pairs → 42-float normalized vector    │
  │                                                      │
  │  Step 1: subtract wrist (landmark 0) → wrist=(0,0)  │
  │  Step 2: divide by max(abs(coords)) → range [-1,1]  │
  └──────────────────────────────────────────────────────┘
     │
     ▼
  RandomForest / MLP  (classifier.pkl)
     │  input: 42 floats
     │  output: (label, confidence)  e.g. ("STOP", 0.91)
     │
     ▼
  Confidence gate (threshold = 0.70)
     │  if confidence < 0.70 → dropped, nothing emitted
     │  if confidence ≥ 0.70 → GestureEvent emitted on EventBus
     ▼
  GestureEvent { gesture=STOP, confidence=0.91, timestamp=... }
```

### Why normalization matters for training quality

```
Raw landmarks (screen-relative, position-dependent):
  [(0.62, 0.44), (0.64, 0.41), ...]   ← STOP, hand on left
  [(0.32, 0.51), (0.34, 0.48), ...]   ← STOP, hand on right
                    ↑ completely different numbers, same gesture!

After normalize_coords():
  [(0.00, 0.00), (0.18, -0.31), ...]  ← STOP, hand on left
  [(0.00, 0.00), (0.18, -0.31), ...]  ← STOP, hand on right
                    ↑ identical — model learns shape, not position
```

Normalization must be **identical** at collection, training, augmentation, and inference.
Any divergence produces garbage features. The single source of truth is `src/core/normalizer.py`.

---

## Stage 2 — Sequence Recognition (Temporal Pattern Matching)

`SequenceRecogniser` (`src/core/sequence_recogniser.py`) is registered as a `BaseAdapter`
on the EventBus. It watches the token stream and matches time-ordered patterns defined in
`src/config/sequences.json`.

```
GestureEvent stream (30 fps from webcam)
─────────────────────────────────────────────────────────────────────▶ time
  STOP  STOP  STOP  STOP  STOP  UP   UP   STOP  STOP  STOP  STOP  STOP
    │     │     │     │     │    │    │     │     │     │     │     │
    └─────┘     │     │     │    │    │     └─────┘     │     │     │
  (deduped)     │     │     │    │    │   (deduped)     │     │     │
                └─────┘     │    │    │                 └─────┘     │
              (deduped)     │    │    │               (deduped)     │
                            │    │    │                             │
                       kept:│ kept:UP │                        kept:│
                           STOP    UP                              STOP

Rolling deque (buffer_size=10):   [STOP, UP, STOP]
```

### Deduplication — the key anti-noise step

```
Without deduplication (30 fps, user holds STOP for 1 second):
  buffer = [STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP, STOP]
  → no room for any other gesture → sequences never match

With deduplication (same situation):
  buffer = [STOP]   ← only ONE entry; buffer stays free for next gesture
```

Rule: if the same token arrives again within `max_gap_ms`, it collapses (lower confidence)
or replaces (higher confidence) the existing entry, keeping the **original timestamp**.

### Pattern Matching

Example sequence definition:

```json
{
  "name": "double_stop",
  "pattern": ["STOP", "STOP", "STOP"],
  "action": "none"
}
```

The matcher checks the **tail** of the buffer against the pattern, then applies two timing
constraints:

```
buffer tail:    [STOP@t=0.0s,  STOP@t=0.8s,  STOP@t=1.6s]
pattern:        [ STOP          STOP           STOP       ]
                   ✓              ✓              ✓

Gap check (max_gap_ms=900ms):
  t=0.8 - t=0.0 = 0.8s  < 0.9s  ✓
  t=1.6 - t=0.8 = 0.8s  < 0.9s  ✓

Total duration check (max_total_ms=3000ms):
  t=1.6 - t=0.0 = 1.6s  < 3.0s  ✓

→ MATCH → emit SequenceEvent("double_stop", tokens=[STOP,STOP,STOP],
                               confidence=avg(conf), duration=1.6s)
```

Failed match examples:

```
# Too slow between gestures
buffer: [STOP@0.0s, STOP@1.1s, STOP@2.0s]
gap: 1.1s > 0.9s  ✗  → no match

# Total time exceeded
buffer: [STOP@0.0s, STOP@0.8s, STOP@3.2s]
total: 3.2s > 3.0s  ✗  → no match

# Wrong pattern
buffer: [STOP@0.0s, UP@0.6s, STOP@1.2s]
pattern[1] = STOP, got UP  ✗  → no match
```

---

## Two-Hand Classification (parallel path)

A second ML classifier handles two-hand gestures using a wider feature vector:

```
Primary hand (42 floats) + Secondary hand (42 floats) = 84-float vector
                                          ↑
              zero-padded when only one hand is present
              (model learns "zeros in slot 2" = single hand gesture)

classifier_two_hand.pkl  →  same RandomForest/MLP interface
                         →  produces same (label, confidence) output
                         →  feeds the same EventBus
```

Both hands are normalized in the **primary hand's reference frame** so inter-hand geometry
(distance, relative position) is preserved as a feature.

---

## What Impacts Model Quality

| Factor | Impact | File |
|---|---|---|
| Normalization consistency | Must be identical at collection, training, augmentation, and inference | `src/core/normalizer.py` |
| Confidence threshold (0.70) | Too low → noisy token stream; too high → valid gestures dropped | `thresholds.json` |
| Dedup `max_gap_ms` (900ms) | Too short → legitimate repetitions collapse; too long → buffer saturates | `thresholds.json` |
| `buffer_size` (10) | Too small → long sequences can't fit; too large → stale entries influence matches | `thresholds.json` |
| `max_gap_ms` per sequence | How fast user must produce each gesture — main tuning knob for sensitivity | `sequences.json` per entry |
| Training data diversity | Model must have seen the gesture at different hand sizes, angles, lighting | `src/lab/augment.py` |
| Cooldown = `max_total_ms` | Suppresses re-firing of same sequence after a match — prevents double-fire | `sequence_recogniser.py:179` |

---

## Key Insight

If your sequences are unreliable, the first place to look is always the
**single-gesture confusion matrix**, not the sequence config. Every classifier
failure propagates directly:

- Misclassification → wrong token in buffer → pattern mismatch
- Confidence below threshold → token dropped → gap breaks timing constraint
- Erratic confidence → holes in what should be a clean token run
