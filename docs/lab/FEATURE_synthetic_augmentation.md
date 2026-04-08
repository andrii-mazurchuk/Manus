# Feature: Synthetic Landmark Augmentation Engine

> Reduce gesture data collection from ~150 manual samples to 3–5 real captures
> by generating a synthetic training dataset from a single canonical pose.

---

## Problem

Adding a new gesture to the classifier currently requires manually collecting
100–150 labelled samples per gesture via `data_collector.py`. This is slow,
repetitive, and a barrier to user customisation.

---

## Key Insight

By the time MediaPipe has produced 42 normalised floats, the hard visual work
(lighting, skin tone, distance, blur) has already been abstracted away. What
remains is pure relative landmark geometry — the same space the augmentation
engine operates in. The sim-to-real gap is nearly closed before we start.

---

## User-Facing Flow

```
1. User holds gesture in front of webcam
2. Press a key → system captures 3–5 frames and averages them into a template
3. Engine generates 500–2000 synthetic variants from the template
4. Rows are appended to data/gestures.csv with the new label
5. Retrain (sklearn RandomForest/MLP, ~10 seconds)
6. Map the new label → action in the adapter config
```

Down from: 10+ minutes of manual collection to ~3 seconds of holding a pose.

---

## Engine Design

### Step 1 — Template capture

Capture N frames from the live webcam while the user holds the gesture.
Normalise each frame (wrist to origin, scale to [-1, 1]) and average the
landmark arrays into a single canonical template of shape (21, 2).

### Step 2 — Derive a logical gesture model

From the template, compute a structured description of the hand pose:

```
Per finger (thumb, index, middle, ring, pinky):
  - extension ratio  : distance(MCP → tip) / max_possible   → 0.0 (curled) – 1.0 (extended)
  - curl angle       : angle at PIP joint
  - lateral spread   : angle between adjacent finger axes at MCP

Wrist:
  - 2D orientation angle
```

These are derivable from the 42 floats alone — no 3D model required.
They serve as the structural description that constrains what synthetic
variants are plausible for this gesture.

### Step 3 — Augmentation transforms

Each synthetic sample applies a random combination of the following transforms
to the template landmark array:

| Transform | What it simulates | Notes |
|---|---|---|
| 2D rotation (±30°) | Camera angle, wrist tilt | Rotation matrix on all (x, y) pairs |
| Per-landmark Gaussian noise (σ ≈ 0.02) | Natural hand tremor, detection jitter | Applied after rotation |
| Per-finger extension perturbation (±10%) | Fingers slightly more/less curled | Perturb fingertip along the finger axis vector |
| Global scale variation (±5%) | Minor depth changes | Applied before re-normalising |
| Horizontal flip | Left vs right hand | x → -x on all landmarks |
| Finger spread variation (±5°) | Lateral spread/squeeze | Rotate adjacent fingers around their MCP |

Noise magnitudes should be configurable. Start with the values above and tune
against held-out real samples.

### Step 4 — Output

Each generated row is formatted to match the existing CSV contract:

```
label, x0, y0, x1, y1, ..., x20, y20
```

Rows are appended to `data/gestures.csv` (or written to a separate file and
merged before training, to keep real and synthetic samples distinguishable).

---

## Accuracy Expectations

| Gesture distinctiveness | Real samples needed | Expected accuracy |
|---|---|---|
| Clearly distinct topology (fist vs open palm) | 3–5 | ≥ 90% |
| Moderately similar (index up vs index down) | 8–12 | ≥ 88% |
| Very similar geometry | 15+ real + targeted noise tuning | Variable |

The classifier boundary for closely similar gestures depends more on the
quality of synthetic coverage than on volume. If accuracy falls below 88%,
collect more real samples and increase the extension perturbation range rather
than tuning hyperparameters first.

---

## Suggested Module Layout

```
src/
└── data/
    ├── augment.py          # AugmentationEngine class — all transforms
    ├── gesture_template.py # Webcam capture → template array + logical model
    └── extract_landmarks.py  (existing)
```

`augment.py` has no dependency on the webcam or FastAPI — it operates purely
on numpy arrays and can be unit-tested in isolation.

---

## Integration Points

- **`data_collector.py`** — replace or extend with a `--augment` mode that
  calls `gesture_template.py` → `augment.py` → appends to CSV.
- **`train.py`** — no changes needed; it reads the CSV regardless of whether
  rows are real or synthetic.
- **Adapter config** — label → action mapping is unchanged; the new label
  flows through the event bus like any other token.

---

## Out of Scope

- 3D landmark augmentation (MediaPipe z-values are not currently used)
- Full MANO / parametric hand model (overkill for this gesture set)
- GAN-based image synthesis (unnecessary — we operate post-MediaPipe)
- Automatic hyperparameter tuning after synthetic training

---

## Open Questions

1. Should synthetic rows be stored separately (e.g. `data/gestures_synthetic.csv`)
   and merged at train time, so real samples can be inspected independently?
2. What is the minimum viable σ (noise level) before synthetic data starts
   hurting accuracy on real webcam input? Needs empirical tuning.
3. Should the logical gesture model be serialised alongside `classifier.pkl`
   so the augmentation engine can later refine a gesture without recapture?
