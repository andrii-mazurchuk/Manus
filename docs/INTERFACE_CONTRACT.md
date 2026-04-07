# Manus — Interface Contract

> This document defines every shared boundary in the system.
> Both people must agree on this before splitting work. Neither side changes these definitions unilaterally.

---

## GestureEvent

The single object that flows through the entire system.

```python
from dataclasses import dataclass

@dataclass
class GestureEvent:
    label: str        # one of the token constants below
    confidence: float # 0.0 – 1.0, from classifier
    timestamp: float  # Unix timestamp, from time.time()
```

JSON representation (used by REST and WebSocket):

```json
{
  "label": "STOP",
  "confidence": 0.94,
  "timestamp": 1714123456.789
}
```

---

## Gesture Tokens

These are the only valid values for `GestureEvent.label`. Classifiers and adapters must use these exact strings.

| Token | Gesture | Default PC action |
|---|---|---|
| `STOP` | Fist | Mute / pause |
| `PLAY` | Open palm | Unmute / resume |
| `UP` | Index pointing up | Volume up / scroll up |
| `DOWN` | Index pointing down | Volume down / scroll down |
| `CONFIRM` | Thumbs up | Next slide / confirm |
| `CANCEL` | Thumbs down | Previous slide / cancel |
| `MODE` | Peace / V | Switch adapter mode |
| `CUSTOM` | Shaka | User-defined |

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

| Column | Type | Description |
|---|---|---|
| `label` | string | One of the token constants above |
| `x0` – `x20` | float | Normalized x coordinate for landmark 0–20 |
| `y0` – `y20` | float | Normalized y coordinate for landmark 0–20 |

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

### REST

**`POST /gesture`** — push a single gesture event (used by webhook adapter)

Request body: `GestureEvent` JSON (see above)
Response: `{ "status": "ok" }`

**`GET /status`** — system status

```json
{
  "adapters": ["PCAdapter", "WebSocketAdapter"],
  "ws_connections": 2,
  "model_loaded": true
}
```

### WebSocket

**`ws://localhost:8000/ws/gestures`** — live stream

The server pushes a `GestureEvent` JSON message to all connected clients every time the event bus emits. No client→server messages are expected.

```json
{ "label": "CONFIRM", "confidence": 0.91, "timestamp": 1714123456.789 }
```

---

## File / Module Layout

```
manus/
├── core/
│   ├── gesture_event.py     # GestureEvent dataclass
│   ├── base_adapter.py      # BaseAdapter ABC
│   └── event_bus.py         # pub/sub bus
├── adapters/
│   ├── pc_adapter.py
│   ├── websocket_adapter.py
│   └── mqtt_adapter.py
├── api/
│   └── server.py            # FastAPI app
├── frontend/
│   └── index.html
├── data/
│   └── gestures.csv
├── models/
│   └── classifier.pkl
├── capture.py               # webcam + MediaPipe
├── data_collector.py        # label and save snapshots
├── train.py                 # load CSV, train, serialize model
└── main.py                  # wire everything together
```
