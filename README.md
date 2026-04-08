# Manus

Real-time hand gesture recognition with MediaPipe + scikit-learn, built around a shared core contract (`GestureEvent`) and a pluggable adapter/event-bus architecture.

## What This Project Does

- Captures hand landmarks from webcam frames (21 points).
- Normalizes landmarks into a stable 42-float feature vector.
- Trains and serves a classifier that predicts gesture token + confidence.
- Emits typed events through a central `EventBus`.
- Lets adapters consume those events (terminal now, API/WS/MQTT/PC adapters later).

## Architecture

```text
Webcam -> MediaPipe HandLandmarker -> normalize_landmarks() -> GestureClassifier
      -> GestureEvent(gesture, confidence, label, timestamp) -> EventBus -> adapters
```

Core contracts live in `src/core`:

- `GestureToken`: canonical vocabulary enum.
- `GestureEvent`: dataclass flowing through the pipeline.
- `BaseAdapter`: adapter interface (`on_gesture(event)`).
- `EventBus`: thread-safe singleton pub/sub (`EventBus.get()`).
- `normalizer.py`: shared normalization logic and `HAND_CONNECTIONS`.

## Repository Layout

```text
manus/
├── main.py                        # Production pipeline entrypoint (bus + adapters)
├── scripts/
│   ├── data_collector.py          # Interactive webcam sample collection
│   ├── train.py                   # Model training and selection
│   └── capture.py                 # Legacy/debug live inference loop
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base_adapter.py
│   │   ├── classifier.py
│   │   ├── event_bus.py
│   │   ├── gesture_event.py
│   │   └── normalizer.py
│   ├── data/
│   │   ├── extract_landmarks.py   # Dataset images -> gestures.csv
│   │   ├── gestures.csv           # Training rows (generated)
│   │   └── gestures/              # Per-label .npy samples (generated)
│   ├── lab/
│   │   ├── augment.py
│   │   ├── augment_gesture.py     # Template capture + synthetic generation (+ retrain)
│   │   └── gesture_template.py
│   └── models/
│       ├── classifier.pkl         # Trained classifier payload (generated)
│       └── hand_landmarker.task   # MediaPipe task model
└── tests/
```

## Gesture Vocabulary

`STOP`, `PLAY`, `UP`, `DOWN`, `CONFIRM`, `CANCEL`, `MODE`, `CUSTOM`

## Setup

Prerequisites:

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

Install dependencies:

```bash
uv sync
```

## How To Run Everything

### 1) Collect data (manual webcam labeling)

```bash
uv run scripts/data_collector.py
```

Camera selection:

```bash
uv run scripts/data_collector.py --camera 1
```

Hotkeys while collecting:

- `P`: `PLAY`
- `S`: `STOP`
- `U`: `UP`
- `D`: `DOWN`
- `C`: `CONFIRM`
- `X`: `CANCEL`
- `M`: `MODE`
- `T`: `CUSTOM`
- `Q` or `ESC`: quit

Each capture appends one row to `src/data/gestures.csv` and one `.npy` sample under `src/data/gestures/<LABEL>/`.

### 2) (Optional) Build CSV from image dataset

```bash
uv run src/data/extract_landmarks.py
```

Custom dataset path:

```bash
uv run src/data/extract_landmarks.py --dataset src/data/leapGestRecog
```

### 3) Train classifier

```bash
uv run scripts/train.py
```

Custom input/output:

```bash
uv run scripts/train.py --csv src/data/gestures.csv --out src/models/classifier.pkl
```

Training output includes:

- class counts
- 5-fold CV for RandomForest and MLP
- test-set classification report
- saved `classifier.pkl` with model + label encoder

### 4) Run production pipeline (`EventBus`-based)

```bash
uv run main.py
```

Options:

```bash
uv run main.py --camera 1 --threshold 0.70
```

Behavior:

- Classifier predicts `(label_str, confidence)`.
- Below threshold: event is ignored.
- At/above threshold: label is validated against `GestureToken`.
- On valid token, pipeline emits `GestureEvent` to `EventBus`.
- Default adapter prints token + confidence to terminal.

### 5) Run debug capture pipeline (legacy loop)

```bash
uv run scripts/capture.py
```

Use this when you want visualization/debug behavior from the older standalone loop.

### 6) Synthetic augmentation workflow

Capture template + generate synthetic rows:

```bash
uv run src/lab/augment_gesture.py --label WAVE --samples 500
```

Reuse existing template:

```bash
uv run src/lab/augment_gesture.py --label WAVE --samples 500 --no-capture
```

Generate + retrain in one command:

```bash
uv run src/lab/augment_gesture.py --label WAVE --samples 500 --retrain
```

Note: `--no-capture` requires an existing template file at `src/lab/templates/<LABEL>.npy`.

## Data and Model Contracts

Normalization (`src/core/normalizer.py`):

1. Translate coordinates so landmark `0` (wrist) is origin.
2. Scale by max absolute coordinate so values are within `[-1, 1]`.
3. Flatten to `(42,)` float32.

Event contract (`src/core/gesture_event.py`):

```python
GestureEvent(
    gesture: GestureToken,
    confidence: float,
    label: int,
    timestamp: float,
)
```

## Adapter Integration Example

Any adapter must implement:

```python
class BaseAdapter(ABC):
    def on_gesture(self, event: GestureEvent) -> None:
        ...
```

Register an adapter with the shared bus:

```python
from src.core.event_bus import EventBus

bus = EventBus.get()
bus.register(MyAdapter())
```

## Troubleshooting

- `Model not found ... classifier.pkl`:
  - run `uv run scripts/train.py`
- `hand_landmarker.task not found`:
  - run `uv run src/data/extract_landmarks.py` once (downloads model)
- webcam open failure:
  - retry with `--camera 1` (or another index)
- `--no-capture` template error:
  - run once without `--no-capture` to create template first
