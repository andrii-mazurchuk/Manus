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

The canonical source of truth is the `GestureToken` enum in `src/core/gesture_event.py`. Always fetch the live list from `GET /tokens` — never hardcode. The vocabulary has grown beyond the original 8.

| Token     | Hand      | Gesture             | Default action       |
| --------- | --------- | ------------------- | -------------------- |
| `STOP`    | Single    | Fist                | Mute / pause         |
| `PLAY`    | Single    | Open palm           | Unmute / resume      |
| `UP`      | Single    | Index pointing up   | Volume up            |
| `DOWN`    | Single    | Index pointing down | Volume down          |
| `CONFIRM` | Single    | Thumbs up           | Next slide           |
| `CANCEL`  | Single    | Thumbs down         | Previous slide       |
| `MODE`    | Single    | Peace / V           | Switch adapter mode  |
| `CUSTOM`  | Single    | Shaka               | User-defined         |
| `SNAP`    | Single    | Snap                | None (configurable)  |
| `CLAP`    | Two-hand  | Clap                | None (configurable)  |

The event bus emits tokens. Adapters decide what tokens mean — never the classifier.

---

## BaseAdapter

Every adapter must subclass this.

```python
from abc import ABC, abstractmethod
from src.core.gesture_event import GestureEvent
from src.core.sequence_event import SequenceEvent

class BaseAdapter(ABC):
    @abstractmethod
    def on_gesture(self, event: GestureEvent) -> None:
        """Called by the event bus for every emitted GestureEvent."""
        ...

    def on_sequence(self, event: SequenceEvent) -> None:
        """Called by the event bus for every emitted SequenceEvent. Default: no-op."""
        pass
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

### Two-hand training data (`src/data/gestures_two_hand.csv`)

85 columns: `label` + 84 floats. Primary hand (suffix `_p`) first, secondary hand (suffix `_s`) second. Both hands are normalized in the primary-wrist reference frame.

```
label,x0_p,y0_p,...,x20_p,y20_p,x0_s,y0_s,...,x20_s,y20_s
```

When secondary hand is absent (single-hand gesture trained in two-hand model), the `_s` columns are all zeros (zero-masking). The model learns that a zero block means "no second hand".

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

### REST (Studio endpoints — Phases 4–5)

**Dataset / Camera Session**

| Endpoint | Purpose |
|---|---|
| `GET /api/dataset/cameras` | List available camera indices |
| `GET /api/dataset/stats` | Sample counts per label (single + two-hand) |
| `GET /api/dataset/capture/stream` | Always-on annotated MJPEG stream (reads from CameraSession buffer) |
| `POST /api/dataset/session/start` | Start/restart CameraSession `{camera, label, capture_type, mode, samples_per_trigger}` |
| `POST /api/dataset/session/stop` | Stop session, release camera |
| `POST /api/dataset/session/trigger` | Fire capture trigger (non-blocking) → `{action}` |
| `GET /api/dataset/session/status` | `{active, state, result, label, capture_type, mode, recording}` |
| `POST /api/dataset/session/label` | Hot-swap label without camera restart |
| `POST /api/dataset/upload` | Upload zip for bulk landmark extraction |
| `DELETE /api/dataset/{label}?mode=single\|two_hand` | Remove all samples for a label |

> **Removed in Phase 5d:** `POST /api/dataset/capture` and `GET /api/dataset/capture/annotated_stream`. Use the session endpoints instead.

**Training**

| Endpoint | Purpose |
|---|---|
| `POST /api/train/start` | `{model_type: "single"\|"two_hand"}` — trigger background training |
| `GET /api/train/status?model_type=single\|two_hand` | Poll training progress and results |
| `GET /api/train/history` | Last 5 runs per model type |

**Config**

| Endpoint | Purpose |
|---|---|
| `GET /api/config/actions` | Get gesture→action mapping |
| `PUT /api/config/actions` | Update gesture→action mapping |
| `GET /api/config/thresholds` | Get per-adapter thresholds |
| `PUT /api/config/thresholds` | Update per-adapter thresholds |

**Sequences**

| Endpoint | Purpose |
|---|---|
| `GET /api/sequences/status` | `{available: true, sequence_count, timing}` |
| `GET /api/sequences/list` | All defined sequences |
| `POST /api/sequences/` | Create sequence |
| `GET /api/sequences/timing` | Global timing defaults |
| `PUT /api/sequences/timing` | Update timing defaults |
| `PUT /api/sequences/{name}` | Update sequence definition |
| `DELETE /api/sequences/{name}` | Delete sequence |

> **Route ordering gotcha:** In `sequences.py`, the `GET /timing` and `PUT /timing` routes must be defined **before** the parameterized `/{name}` routes. FastAPI matches top-to-bottom; if `/{name}` comes first it will shadow the literal path `/timing`.

### WebSocket

`**ws://localhost:8000/ws/gestures**` — live stream

The server pushes a `GestureEvent` JSON message to all connected clients every time the event bus emits. No client→server messages are expected.

```json
{ "gesture": "CONFIRM", "confidence": 0.91, "label": 1, "timestamp": 1714123456.789 }
```

---

## SequenceEvent

Emitted by `SequenceRecogniser` when a token pattern is matched. Flows through the same EventBus as `GestureEvent` via `emit_sequence()` / `on_sequence()`.

```python
@dataclass
class SequenceEvent:
    name:       str                # sequence definition name, e.g. "double_stop"
    tokens:     list[GestureToken] # the matched token pattern
    confidence: float              # mean confidence of matched GestureEvents
    timestamp:  float              # time.time() of final matching token
    duration:   float = 0.0        # seconds from first to last token

    def to_dict(self) -> dict: ... # includes "type": "sequence"
```

JSON over WebSocket:
```json
{
  "type":       "sequence",
  "name":       "double_stop",
  "tokens":     ["STOP", "STOP", "STOP"],
  "confidence": 0.91,
  "timestamp":  1714123456.789,
  "duration":   1.2
}
```

Sequence definitions live in `src/config/sequences.json`. Global timing defaults live in `src/config/thresholds.json` under `sequence_model`. Both are editable live via the Studio Sequences tab or the `/api/sequences/` CRUD endpoints.

**Deduplication note:** The `SequenceRecogniser` collapses consecutive identical tokens that arrive within `max_gap_ms` into one entry (keeping the highest-confidence event). This prevents the 30fps event stream from saturating the buffer with repeated tokens, which would mask deliberate double/triple gestures.

---

## File / Module Layout

```
repo/
├── main.py                            # pipeline entrypoint — registers all adapters
├── src/
│   ├── core/
│   │   ├── gesture_event.py           # GestureEvent + GestureToken (10 tokens)
│   │   ├── sequence_event.py          # SequenceEvent dataclass
│   │   ├── base_adapter.py            # on_gesture() [abstract] + on_sequence() [no-op]
│   │   ├── event_bus.py               # emit() + emit_sequence()
│   │   ├── classifier.py              # single-hand 42-float classifier
│   │   ├── two_hand_classifier.py     # two-hand 84-float classifier
│   │   ├── sequence_recogniser.py     # SequenceRecogniser(BaseAdapter)
│   │   └── normalizer.py             # normalize_landmarks(), normalize_two_hand_landmarks(), HAND_CONNECTIONS
│   ├── adapters/
│   │   ├── websocket_adapter.py       # on_gesture() + on_sequence() → POST /gesture + /sequence
│   │   ├── pc_adapter.py              # pynput media keys
│   │   └── mqtt_adapter.py            # MQTT publish
│   ├── api/
│   │   ├── server.py                  # FastAPI app; POST /gesture, POST /sequence, WS /ws/gestures
│   │   ├── connection_manager.py      # broadcast() + broadcast_json()
│   │   ├── models/gesture.py          # Pydantic GestureEvent model
│   │   └── routes/
│   │       ├── dataset.py             # CameraSession singleton + session/* + stats/cameras/upload/delete
│   │       ├── training.py            # single + two_hand model training
│   │       ├── config.py              # actions + thresholds
│   │       └── sequences.py           # full CRUD + timing (route order matters — see gotcha above)
│   ├── config/
│   │   ├── gesture_actions.json       # static_actions (10 tokens) + sequence_actions
│   │   ├── thresholds.json            # pc_adapter, websocket_adapter, sequence_model timing
│   │   └── sequences.json             # user-defined sequence patterns
│   ├── data/
│   │   ├── gestures.csv               # single-hand training data (label + 42 floats)
│   │   ├── gestures/                  # per-label .npy files (single-hand)
│   │   ├── gestures_two_hand.csv      # two-hand training data (label + 84 floats)
│   │   └── gestures_two_hand/         # per-label .npy files (two-hand)
│   ├── models/
│   │   ├── classifier.pkl             # single-hand model artifact
│   │   └── classifier_two_hand.pkl    # two-hand model (optional — pipeline falls back if absent)
│   ├── lab/
│   │   ├── augment.py                 # AugmentationEngine (42-float single-hand only)
│   │   └── augment_gesture.py         # CLI wrapper
│   └── frontend/
│       ├── dashboard.html             # live gesture monitoring
│       ├── studio.html                # Dataset / Train / Config / Sequences tabs
│       └── shared/
│           ├── nav.js
│           └── styles.css
├── scripts/                           # fallback CLI tools only — not used by active pipeline
│   ├── train.py
│   └── bootstrap_augmented_dataset.py
├── docs/
│   ├── INTERFACE_CONTRACT.md          # this file
│   ├── PLAN.md                        # original hackathon plan (Phases 0–5)
│   ├── PHASE5_IMPLEMENTATION.md       # detailed implementation notes for Phases 5a–5d
│   └── lab/
│       └── FEATURE_synthetic_augmentation.md
└── tests/
    └── test_api.py                    # 39 tests, all passing
```

