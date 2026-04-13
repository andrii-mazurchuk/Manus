# Manus — Interface Contract

> This document defines every shared boundary in the system.
> Both people must agree on this before splitting work. Neither side changes these definitions unilaterally.

---

## GestureEvent

The single object that flows through the entire system.

```python
from dataclasses import dataclass
from src.core.gesture_event import GestureToken

@dataclass
class GestureEvent:
    gesture:    GestureToken  # canonical label enum value
    confidence: float         # 0.0 – 1.0, from classifier
    label:      int           # label index from sklearn LabelEncoder
    timestamp:  float         # Unix timestamp, from time.time()

    def to_dict(self) -> dict: ...
```

JSON representation (as broadcast over WebSocket and accepted by `POST /gesture`):

```json
{
  "gesture":    "STOP",
  "confidence": 0.94,
  "label":      3,
  "timestamp":  1714123456.789
}
```

**Note on field naming:** The `gesture` field carries the string token value (e.g. `"STOP"`). The `label` field is the integer index used internally by the sklearn LabelEncoder — adapters should use `gesture`, not `label`.

---

## Gesture Tokens

These are the current valid values for `GestureEvent.gesture`. The vocabulary is expected to grow beyond these 8 tokens. The canonical source of truth is the `GestureToken` enum in `src/core/gesture_event.py` — always fetch the live list from `GET /api/tokens` rather than hardcoding these values in clients.


| Token     | Gesture             | Default PC action         |
| --------- | ------------------- | ------------------------- |
| `STOP`    | Fist                | Mute / pause              |
| `PLAY`    | Open palm           | Unmute / resume           |
| `UP`      | Index pointing up   | Volume up / scroll up     |
| `DOWN`    | Index pointing down | Volume down / scroll down |
| `CONFIRM` | Thumbs up           | Next slide / confirm      |
| `CANCEL`  | Thumbs down         | Previous slide / cancel   |
| `MODE`    | Peace / V           | Switch adapter mode       |
| `CUSTOM`  | Shaka               | User-defined              |


The event bus emits tokens. Adapters decide what tokens mean — never the classifier.

---

## BaseAdapter

Every adapter must subclass this. No other interface is required.

```python
from abc import ABC, abstractmethod
from core.gesture_event import GestureEvent

class BaseAdapter(ABC):
    @abstractmethod
    def on_gesture(self, event: GestureEvent) -> None:
        """Called by the event bus for every emitted GestureEvent."""
        ...
```

Adapters are responsible for their own confidence threshold and debounce logic. The event bus passes every event through unfiltered.

**Recommended defaults (implement in your adapter):**

- Ignore events where `confidence < 0.70`
- Debounce: ignore the same `label` if it fired within the last 500ms

---

## Training Data — CSV Schema

File location: `data/gestures.csv`


| Column       | Type   | Description                               |
| ------------ | ------ | ----------------------------------------- |
| `label`      | string | One of the token constants above          |
| `x0` – `x20` | float  | Normalized x coordinate for landmark 0–20 |
| `y0` – `y20` | float  | Normalized y coordinate for landmark 0–20 |


**Total columns:** 1 label + 42 floats = 43 columns per row.

Header row must be present:

```
label,x0,y0,x1,y1,...,x20,y20
```

---

## Landmark Normalization

Raw MediaPipe coordinates are pixel positions. Before saving to CSV or feeding the classifier, normalize as follows:

1. **Translate:** subtract landmark 0 (wrist) from all landmarks so the wrist is at (0, 0)
2. **Scale:** divide all coordinates by the maximum absolute value across all 42 values in that frame, so everything fits in [-1, 1]

```python
import numpy as np

def normalize(landmarks):
    # landmarks: list of 21 (x, y) tuples from MediaPipe
    coords = np.array([[lm.x, lm.y] for lm in landmarks])  # shape (21, 2)
    coords -= coords[0]                                      # translate to wrist origin
    scale = np.max(np.abs(coords))
    if scale > 0:
        coords /= scale                                      # scale to [-1, 1]
    return coords.flatten()                                  # shape (42,)
```

This makes the classifier invariant to hand position and distance from camera.

---

## MediaPipe Landmark Index Reference

```
0  = Wrist
1  = Thumb CMC       2  = Thumb MCP       3  = Thumb IP        4  = Thumb Tip
5  = Index MCP       6  = Index PIP       7  = Index DIP       8  = Index Tip
9  = Middle MCP     10  = Middle PIP     11  = Middle DIP     12  = Middle Tip
13 = Ring MCP       14  = Ring PIP       15  = Ring DIP       16  = Ring Tip
17 = Pinky MCP      18  = Pinky PIP      19  = Pinky DIP      20  = Pinky Tip
```

The fingertips (4, 8, 12, 16, 20) and the wrist (0) are the most discriminative landmarks for gesture classification.

---

## API Schema

### REST (core endpoints)

`**POST /gesture**` — push a single gesture event (used by WebSocketAdapter to forward pipeline events to the API server)

Request body: `GestureEvent` JSON (see above). `confidence` must be in `[0.0, 1.0]`.
Response: `{ "status": "ok" }`

`**GET /status**` — live system state

```json
{
  "adapters": ["TerminalAdapter", "WebSocketAdapter", "PCAdapter", "MQTTAdapter"],
  "ws_connections": 2,
  "model_loaded": false
}
```
`adapters` is populated dynamically from the live `EventBus` — reflects whatever is actually registered.

`**GET /tokens**` — canonical gesture vocabulary

```json
{ "tokens": ["STOP", "PLAY", "UP", "DOWN", "CONFIRM", "CANCEL", "MODE", "CUSTOM"] }
```
This is the single source of truth for token names. Clients must use this endpoint rather than hardcoding the list — the vocabulary will grow.

### REST (Studio endpoints — Phase 4)

| Endpoint | Purpose |
|---|---|
| `GET /api/cameras` | List available camera indices |
| `GET /api/dataset/stats` | Sample counts per label |
| `POST /api/dataset/capture` | Trigger server-side live capture |
| `GET /api/dataset/capture/stream?camera=N` | MJPEG camera preview |
| `POST /api/dataset/upload` | Upload zip for bulk landmark extraction |
| `DELETE /api/dataset/{label}` | Remove all samples for a label |
| `POST /api/train/start` | Trigger background model training |
| `GET /api/train/status` | Poll training progress and results |
| `GET /api/config/actions` | Get gesture→action mapping |
| `PUT /api/config/actions` | Update gesture→action mapping |
| `GET /api/config/thresholds` | Get per-adapter thresholds |
| `PUT /api/config/thresholds` | Update per-adapter thresholds |
| `GET /api/sequences/status` | Sequence recognition availability (placeholder) |

### WebSocket

`**ws://localhost:8000/ws/gestures**` — live stream

The server pushes a `GestureEvent` JSON message to all connected clients every time the event bus emits. No client→server messages are expected.

```json
{ "gesture": "CONFIRM", "confidence": 0.91, "label": 1, "timestamp": 1714123456.789 }
```

---

## Gesture Sequence Events (planned — not yet implemented)

Gesture sequences are **temporal patterns of existing static tokens** observed over a time window. They are not a new vocabulary: the sequence recogniser watches the EventBus output stream and fires a secondary event when it detects a matching pattern (e.g. UP → DOWN → UP → DOWN = "wave").

**Design constraints locked in:**
- Sequences share the existing adapter layer — a `SequenceEvent` (TBD) will be emitted on the same EventBus.
- Sequence actions live in `src/config/gesture_actions.json` under the `sequence_actions` key (separate from `static_actions`).
- The Studio page Sequences tab is already structured to hold dataset/train/config sub-panels for sequences once the feature is implemented.
- Threshold for the sequence model is reserved as `sequence_model` in `src/config/thresholds.json`.

---

## File / Module Layout

```
repo/
├── main.py                          # pipeline entrypoint — registers all adapters
├── src/
│   ├── core/
│   │   ├── gesture_event.py         # GestureEvent dataclass + GestureToken enum
│   │   ├── base_adapter.py          # BaseAdapter ABC (canonical location)
│   │   ├── event_bus.py             # thread-safe singleton pub/sub
│   │   ├── classifier.py            # GestureClassifier (loads classifier.pkl)
│   │   └── normalizer.py            # normalize_landmarks(), HAND_CONNECTIONS
│   ├── adapters/
│   │   ├── websocket_adapter.py     # forwards events to POST /gesture (with debounce)
│   │   ├── pc_adapter.py            # pynput media/arrow keys (reads gesture_actions.json)
│   │   └── mqtt_adapter.py          # publishes JSON to MQTT broker
│   ├── api/
│   │   ├── server.py                # FastAPI app, static file serving, router includes
│   │   ├── connection_manager.py    # WebSocket broadcast manager
│   │   ├── models/gesture.py        # Pydantic GestureEvent model (API layer)
│   │   └── routes/
│   │       ├── dataset.py           # /api/dataset/* — capture, stats, upload
│   │       ├── training.py          # /api/train/* — trigger, status, history
│   │       ├── config.py            # /api/config/* — actions, thresholds
│   │       └── sequences.py         # /api/sequences/* — placeholder
│   ├── config/
│   │   ├── gesture_actions.json     # static_actions + sequence_actions
│   │   └── thresholds.json          # per-adapter + sequence_model threshold
│   ├── data/
│   │   ├── gestures.csv             # training data (label + 42 floats)
│   │   └── gestures/                # per-label .npy files
│   ├── models/
│   │   └── classifier.pkl           # {"model": ..., "label_encoder": ...}
│   └── frontend/
│       ├── dashboard.html           # live monitoring page
│       ├── studio.html              # data collection, training, config
│       └── shared/
│           ├── nav.js               # shared navigation component
│           └── styles.css           # shared CSS variables
├── scripts/
│   ├── train.py                     # CLI training (also importable as run_training())
│   ├── data_collector.py            # interactive CLI capture
│   ├── bootstrap_augmented_dataset.py
│   └── capture_demo.py
└── tests/
    └── test_api.py
```

