# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Manus is a real-time hand gesture recognition framework. It reads webcam input, classifies hand gestures via a trained ML model, and routes typed `GestureEvent` objects to any registered adapter (terminal, REST/WebSocket API, MQTT, PC control, etc.).

---

## Commands

This project uses `uv` for dependency management. All commands use `uv run`.

```bash
# Install dependencies
uv sync

# Run the full pipeline (webcam → classifier → adapters)
uv run main.py [--camera 0] [--threshold 0.70]

# Run the FastAPI server
uv run uvicorn src.api.server:app --reload

# Collect training data interactively (webcam, press keys to label snapshots)
uv run scripts/data_collector.py

# Bootstrap synthetic training data (recommended over manual collection)
uv run scripts/bootstrap_augmented_dataset.py --samples 400 --retrain

# Train classifier from CSV
uv run scripts/train.py

# Run tests
uv run pytest

# Run a single test file
uv run pytest tests/test_api.py -v
```

---

## Architecture

```
Webcam → MediaPipe HandLandmarker (21 landmarks / 42 floats)
       → normalize_landmarks()         # wrist-relative, scaled to [-1, 1]
       → GestureClassifier.predict()   # RandomForest or MLP
       → EventBus.emit(GestureEvent)   # thread-safe singleton pub/sub
       → Adapters                      # TerminalAdapter, WebSocketAdapter, ...
```

### Core contracts (`src/core/`)

`src/core/__init__.py` re-exports everything downstream code needs:

```python
from src.core import BaseAdapter, EventBus, GestureEvent, GestureToken
from src.core import normalize_landmarks, normalize_coords, HAND_CONNECTIONS
```

**`GestureEvent`** — the object flowing through the entire pipeline:
```python
@dataclass
class GestureEvent:
    gesture: GestureToken   # Canonical label enum
    confidence: float       # Prediction score 0.0–1.0
    label: int              # Label index from sklearn LabelEncoder
    timestamp: float        # Unix timestamp

    def to_dict(self) -> dict: ...
```

**`GestureToken`** — the canonical gesture vocabulary (single source of truth shared by ML + API layers):
`STOP | PLAY | UP | DOWN | CONFIRM | CANCEL | MODE | CUSTOM`

**Normalization contract** (`src/core/normalizer.py`) — used identically during collection, training, augmentation, and inference:
1. Translate: subtract landmark 0 (wrist) → wrist at origin
2. Scale: divide by `max(abs(coords))` → values in `[-1, 1]`
3. Flatten to 42-float array (x0, y0, x1, y1, …, x20, y20)

**`EventBus`** — thread-safe singleton (`EventBus.get()`). Adapters call `bus.register(adapter)` and receive every `GestureEvent` via `on_gesture()`. Any adapter subclasses `BaseAdapter(ABC)` and implements `on_gesture(self, event: GestureEvent) -> None`.

### API layer (`src/api/`)

FastAPI app with three endpoints:
- `POST /gesture` — receives `GestureEvent` JSON, broadcasts to all WebSocket clients
- `WS /ws/gestures` — subscribe to real-time gesture events
- `GET /tokens` — returns the 8 canonical token strings
- `GET /status` — returns connected adapter list + WS connection count

The `WebSocketAdapter` (`src/adapters/`) forwards pipeline events to `POST /gesture`, bridging the real-time pipeline to the API server.

### Training artifacts

`src/models/classifier.pkl` is a pickle-serialized dict:
```python
{"model": <RandomForestClassifier | MLPClassifier>, "label_encoder": <LabelEncoder>}
```

`GestureClassifier` (`src/core/classifier.py`) loads this and exposes `predict(landmarks) -> tuple[str, float]`.

Training data lives in `src/data/gestures.csv` (43 columns: `label` + 42 landmark floats). The `src/data/gestures/` directory mirrors the CSV as per-label `.npy` files.

### Augmentation (`src/lab/`)

`AugmentationEngine` applies rotation (±30°), Gaussian noise (σ=0.02), and scale extension (±10%) to a single captured template to produce N synthetic training rows. This is the recommended path when no large existing dataset is available.

---

## Key Design Decisions

- **Confidence threshold 0.70** — events below this are dropped before `EventBus.emit()`.
- **Debounce 500 ms** — `WebSocketAdapter` suppresses repeat firings of the same token within 500 ms.
- **`GestureToken` enum** — both the ML layer and the API Pydantic model validate against this enum; the classifier never references adapter-specific logic.
- **Known duplication** — `src/adapters/base_adapter.py` is an exact copy of `src/core/base_adapter.py`. The canonical version is `src/core/base_adapter.py`; prefer importing from `src.core`.
