# Manus — Hand Gesture Recognition Pipeline

Real-time hand gesture classification using MediaPipe and scikit-learn.

---

## Project Structure

```
manus/
├── scripts/
│   ├── data_collector.py        # Webcam data collection tool
│   ├── train.py                 # Train classifier from CSV
│   └── capture.py               # Live webcam inference pipeline
├── src/
│   ├── core/
│   │   └── classifier.py        # GestureClassifier inference wrapper
│   ├── data/
│   │   ├── extract_landmarks.py # Dataset images -> gestures.csv
│   │   ├── gestures.csv         # Landmark training data (generated)
│   │   └── gestures/            # Per-label .npy snapshots (generated)
│   └── models/
│       ├── classifier.pkl       # Trained model (generated)
│       └── hand_landmarker.task # MediaPipe hand model weights
├── docs/                        # Project brief, plan, interface contract
└── tests/                       # Webcam verification scripts
```

---

## Data Pipeline

```
Webcam
  |
  v
MediaPipe HandLandmarker              (models/hand_landmarker.task)
  | 21 landmarks — normalized to wrist origin, scaled to [-1, 1]
  | flattened to 42 floats  [x0,y0, x1,y1, ..., x20,y20]
  v
gestures.csv  +  gestures/<LABEL>/*.npy
  |
  v
Train (RandomForest + MLP, cross-validated)
  |
  v
classifier.pkl                        (winning model + LabelEncoder)
  |
  v
GestureClassifier.predict(landmarks)  -> (label, confidence)
  |
  v
Terminal output + webcam overlay      (capture.py)
  |
  v
[Event Bus -> Adapters]               (PC control / REST / WebSocket / MQTT)
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed

### 1. Install dependencies

```bash
uv sync
```

### 2. Collect training data

```bash
uv run python scripts/data_collector.py
```

A webcam window opens showing your hand skeleton. Press a key to snapshot the current pose:

| Key | Gesture token |
|-----|--------------|
| `P` | PLAY         |
| `S` | STOP         |
| `U` | UP           |
| `D` | DOWN         |
| `C` | CONFIRM      |
| `X` | CANCEL       |
| `M` | MODE         |
| `T` | CUSTOM       |
| `Q` / `ESC` | quit |

Aim for **100 samples per gesture**. The sidebar shows live counts and progress bars.
Each capture saves a `.npy` file under `src/data/gestures/<LABEL>/` and appends a row to `src/data/gestures.csv`.

If your webcam is not on index 0:

```bash
uv run python scripts/data_collector.py --camera 1
```

### 3. Train the classifier

```bash
uv run python scripts/train.py
```

This reads `src/data/gestures.csv`, trains both a RandomForest and an MLP with 5-fold cross-validation, and saves the better model to `src/models/classifier.pkl`. A classification report and class-balance warning are printed if any gesture is underrepresented.

To point at a different CSV or output path:

```bash
uv run python scripts/train.py --csv src/data/gestures.csv --out src/models/classifier.pkl
```

### 4. Run the live pipeline

```bash
uv run python scripts/capture.py
```

If your webcam is not on index 0:

```bash
uv run python scripts/capture.py --camera 1
```

Press **Q** or **ESC** to quit.

---

## What You See

The webcam window shows:

- **Hand skeleton** — 21 landmarks in yellow with grey connections
- **Gesture label** — large text at the top of the frame
- **Confidence bar** — fills left to right; green when ≥ 0.70 threshold, grey when below
- **"below threshold"** — shown when confidence < 0.70
- **FPS counter** — top right

The terminal prints the detected gesture and confidence every time the label changes.

---

## Gesture Vocabulary

| Gesture | Token | Intended action |
|---|---|---|
| Fist | `STOP` | Mute / pause |
| Open palm | `PLAY` | Unmute / resume |
| Index pointing up | `UP` | Volume up / scroll up |
| Index pointing down | `DOWN` | Volume down / scroll down |
| Thumbs up | `CONFIRM` | Next slide / confirm |
| Thumbs down | `CANCEL` | Previous slide / cancel |
| Peace / V | `MODE` | Switch adapter mode |
| OK sign | `CUSTOM` | User-defined |

---

## Architecture

```
Webcam → MediaPipe Hands (21 landmarks / 42 floats)
       → Gesture Classifier (RandomForest or MLP, scikit-learn)
       → Event Bus (custom pub/sub)
       → Adapters (PC / REST / WebSocket / MQTT)
```

Core contract between layers:

```python
GestureEvent(label: str, confidence: float, timestamp: float)
```

Every adapter subclasses `BaseAdapter` and implements one method:

```python
class BaseAdapter(ABC):
    def on_gesture(self, event: GestureEvent) -> None: ...
```

**Design decisions:**
- Confidence threshold: 0.70 — detections below this are not forwarded to adapters
- Debounce: 500 ms minimum between successive firings of the same gesture token
- Landmarks are normalized to wrist origin and scaled to `[-1, 1]` before saving and inference, making the classifier hand-position invariant
