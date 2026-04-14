# Phase 5 Implementation Reference

> Documents everything built in Phases 5a–5d. Intended for agents and developers
> picking up work in this area. Read alongside `INTERFACE_CONTRACT.md`.

---

## Phase 5a — Two-Hand Static Gesture Support

### What was added

**New gesture tokens** (`src/core/gesture_event.py`):
- `SNAP` — single-hand gesture
- `CLAP` — two-hand gesture
- Both added to `GestureToken` enum and `src/config/gesture_actions.json`

**New normalizer** (`src/core/normalizer.py`):
```python
normalize_two_hand_landmarks(primary_landmarks, secondary_landmarks=None) -> np.ndarray  # shape (84,)
```
- Both hands normalized in the **primary-wrist reference frame** (not independently)
- Primary = right hand if detected, else left hand
- Secondary is normalized relative to primary wrist + primary scale — this preserves inter-hand geometry
- If secondary absent: zeros (zero-masking)
- This means the classifier can handle both single-hand and two-hand input with the same model

**New classifier** (`src/core/two_hand_classifier.py`):
- `TwoHandGestureClassifier` — loads `src/models/classifier_two_hand.pkl`
- `predict(primary_landmarks, secondary_landmarks=None) -> tuple[str, float]`
- Graceful `FileNotFoundError` if model file is absent (caught in `main.py`)

**Training data** (`src/data/gestures_two_hand.csv`):
- 85 columns: `label` + 84 floats (`x0_p,y0_p,...x20_p,y20_p,x0_s,...y20_s`)
- Separate from `gestures.csv` — two-hand model is trained independently

**main.py changes**:
- `num_hands=1` → `num_hands=2`
- Handedness routing: `Right` = primary, `Left` = secondary
- Two-hand-first dispatch: if two-hand model loaded AND confidence ≥ threshold → emit, skip single-hand
- Graceful fallback to single-hand if two-hand model missing

**Dataset API changes** (`src/api/routes/dataset.py`):
- `CaptureRequest.mode: Literal["single", "two_hand"]`
- Two-hand capture writes to `gestures_two_hand.csv` with 84-float rows
- `GET /api/dataset/stats` returns both `labels`/`total` and `two_hand_labels`/`two_hand_total`
- `DELETE /api/dataset/{label}?mode=two_hand` targets two-hand CSV

**Training API changes** (`src/api/routes/training.py`):
- `TrainStartRequest.model_type: Literal["single", "two_hand"]`
- Two-hand path calls `run_two_hand_training()` → saves `classifier_two_hand.pkl`
- `GET /api/train/status?model_type=two_hand` for two-hand status

### Known gap
No augmentation engine for 84-float two-hand data. `AugmentationEngine` only handles 42-float single-hand arrays. Two-hand snapshot capture saves 1 raw sample. This is a known limitation to address in a future phase.

---

## Phase 5b — Gesture Sequence Recognition

### What was added

**`src/core/sequence_event.py`**:
```python
@dataclass
class SequenceEvent:
    name:       str               # e.g. "double_stop"
    tokens:     list[GestureToken]
    confidence: float             # mean confidence of matched events
    timestamp:  float             # time of final matching token
    duration:   float = 0.0       # seconds from first to last token

    def to_dict(self) -> dict:    # includes "type": "sequence"
```

**`src/core/sequence_recogniser.py`** — `SequenceRecogniser(BaseAdapter)`:
- Rolling `deque` buffer of recent `(GestureEvent, timestamp)` pairs
- Consecutive token deduplication: same token within `max_gap_ms` collapses to highest-confidence entry — prevents 30fps stream from saturating the buffer with repeated tokens
- Pattern matching: checks all defined sequences against the tail of the buffer
- Per-sequence cooldown prevents rapid refiring
- `reload_sequences()` for hot-reload after API writes
- Registered in `main.py` like any other adapter: `bus.register(SequenceRecogniser())`

**EventBus extension** (`src/core/event_bus.py`):
- `emit_sequence(event: SequenceEvent)` — calls `adapter.on_sequence()` on all subscribers

**BaseAdapter extension** (`src/core/base_adapter.py`):
- `on_sequence(self, event: SequenceEvent) -> None` — default no-op, backward compatible

**Sequence definitions** (`src/config/sequences.json`):
```json
{
  "sequences": [
    {
      "name": "double_stop",
      "pattern": ["STOP", "STOP", "STOP"],
      "action": "none"
    }
  ]
}
```

**Timing config** (`src/config/thresholds.json`):
```json
"sequence_model": {
  "default_max_gap_ms": 900,
  "default_max_total_ms": 3000,
  "buffer_size": 10
}
```

**Sequences CRUD API** (`src/api/routes/sequences.py`):
> **CRITICAL:** Route ordering matters. `GET /timing` and `PUT /timing` must be defined
> BEFORE `GET /{name}` / `PUT /{name}` / `DELETE /{name}` or FastAPI's parameterized
> route will shadow the literal `/timing` path.

| Endpoint | Purpose |
|---|---|
| `GET /api/sequences/status` | `{available: true, sequence_count, timing}` |
| `GET /api/sequences/list` | All defined sequences |
| `POST /api/sequences/` | Create sequence |
| `GET /api/sequences/timing` | Global timing defaults |
| `PUT /api/sequences/timing` | Update timing defaults |
| `PUT /api/sequences/{name}` | Update sequence |
| `DELETE /api/sequences/{name}` | Delete sequence |

**WebSocket broadcast** (`src/api/server.py`):
- `POST /sequence` endpoint receives `SequenceEvent` from `WebSocketAdapter.on_sequence()` and broadcasts JSON to all WS clients
- Payload includes `"type": "sequence"` so clients can distinguish from gesture events

**`src/core/__init__.py`** exports: `SequenceEvent`, `SequenceRecogniser`

---

## Phase 5c — Annotated Capture Preview (superseded by 5d)

Phase 5c's intent was to show live annotated camera frames during capture. This was implemented as a shared `_annotated_buf` + `/capture/annotated_stream` endpoint, but the entire implementation was replaced by Phase 5d's `CameraSession` architecture. Phase 5c's goal is fully satisfied by the always-on annotated stream in 5d.

---

## Phase 5d — Trigger-Based Capture + Always-On Annotated Preview

### Architecture change

**Before 5d:** Preview and capture each opened their own `cv2.VideoCapture` → camera conflict. Capture was a long-running HTTP request that blocked the camera for its entire duration.

**After 5d:** A single `CameraSession` background thread owns the camera and MediaPipe detector for the lifetime of a dataset session. It continuously annotates frames and writes to `_live_jpeg`. The MJPEG stream reads from that buffer — no camera opens in the stream endpoint.

### CameraSession (`src/api/routes/dataset.py`)

Module-level singleton: `_session = CameraSession()`

**State machine:** `idle → ready → capturing → done → ready …`

**Capture modes:**
- `snapshot` — one trigger → extract one frame of landmarks → `AugmentationEngine.generate(flat, N)` → write N augmented rows to CSV + NPY
- `sequence` — trigger 1 starts recording (accumulates normalized landmark arrays in `_seq_rows`), trigger 2 stops and saves all accumulated rows as raw samples

**Key design decisions:**
- Always uses `num_hands=2` in MediaPipe regardless of `mode` — avoids camera restart when user switches between single/two-hand modes mid-session
- Skeleton drawn in **green** when prediction matches target label, **orange** when it doesn't, no skeleton when no hand detected
- Saves run in background threads (`threading.Thread(daemon=True)`) so the camera loop never pauses
- Snapshot trigger is ignored if `state == "capturing"` (save still in progress) — returns `action: "busy"`
- "SAVED!" flash (white border + text for 0.45s) drawn directly on frames after snapshot
- Two-hand snapshot saves 1 raw sample (no augmentation — engine not extended to 84-float yet)

### New API endpoints

| Endpoint | Notes |
|---|---|
| `GET /api/dataset/capture/stream` | Always-on MJPEG; placeholder frame when session idle |
| `POST /api/dataset/session/start` | Body: `{camera, label, capture_type, mode, samples_per_trigger}` — starts/restarts session |
| `POST /api/dataset/session/stop` | Stops session, releases camera |
| `POST /api/dataset/session/trigger` | Returns `{action: triggered\|recording_started\|recording_stopped\|busy\|no_session}` — non-blocking |
| `GET /api/dataset/session/status` | `{active, state, result, label, capture_type, mode, recording}` |
| `POST /api/dataset/session/label` | Hot-swap label without camera restart |

**Removed endpoints:** `POST /api/dataset/capture`, `GET /api/dataset/capture/annotated_stream`

### Frontend (`src/frontend/studio.html`)

**Removed:** `startPreview()`, `stopPreview()`, `togglePreview()`, `startCapture()`, `previewActive`, `capturing`, `wasPreviewActiveOnCapture`, Start Preview button, Count input, old Capture button

**Added:**
- `syncSession()` — starts/restarts session; called on label/mode/capture-type change and on camera load
- `fireTrigger()` — calls `POST /session/trigger`; handles all action strings
- `onCaptureLabelChange()` — hot-swaps label via `POST /session/label` if session active; else calls `syncSession()`
- `onCaptureTypeChange()` — shows/hides samples-per-trigger input; calls `syncSession()`
- `startStatusPoller()` — polls `GET /session/status` every 600ms; calls `handleSessionStatus()`
- `handleSessionStatus()` — on `state=done`: shows result, calls `loadStats()`, resets button; uses `_lastDoneResult` to avoid reacting to the same result twice
- Spacebar listener — fires `fireTrigger()` when Dataset panel is active
- `beforeunload` — calls `navigator.sendBeacon("/api/dataset/session/stop")` to release camera on page close
- `sleep()` helper still defined but no longer called (dead code — harmless)

**New HTML controls:**
- `<select id="capture-type">` — `snapshot` / `sequence`
- `<span id="samples-wrap">` wrapping `<input id="samples-per-trigger">` — hidden for sequence mode
- `<button id="trigger-btn">` replacing the old capture button
- Spacebar hint text below the control row

### augment.py import fix
`src/lab/augment.py` previously used `from core.normalizer import normalize_coords` (only works when `src/` is in `sys.path`). Changed to try/except:
```python
try:
    from src.core.normalizer import normalize_coords   # API import context
except ImportError:
    from core.normalizer import normalize_coords       # script context
```

---

## Test Suite Status

**39 tests, all passing** as of Phase 5d completion. Tests live in `tests/test_api.py`.

Notable test updates made during Phase 5:
- `_state` → `_states["single"]` in training route tests (after `training.py` refactored to support multi-model)
- `_CSV_PATH` monkeypatch → `DEFAULT_CSV` in training module
- `test_all_eight_tokens_present` renamed `test_all_tokens_present` (token count now > 8)

---

## Current Gesture Token Vocabulary

| Token | Hand | Notes |
|---|---|---|
| `STOP` | Single | Fist |
| `PLAY` | Single | Open palm |
| `UP` | Single | Index pointing up |
| `DOWN` | Single | Index pointing down |
| `CONFIRM` | Single | Thumbs up |
| `CANCEL` | Single | Thumbs down |
| `MODE` | Single | Peace / V |
| `CUSTOM` | Single | Shaka |
| `SNAP` | Single | Added in Phase 5a |
| `CLAP` | Two-hand | Added in Phase 5a |

Canonical source: `src/core/gesture_event.py` `GestureToken` enum.
Always fetch live list from `GET /tokens` — never hardcode.

---

## File Layout (as of Phase 5d)

```
src/
├── core/
│   ├── gesture_event.py          # GestureEvent + GestureToken (10 tokens)
│   ├── base_adapter.py           # on_gesture() + on_sequence() (no-op default)
│   ├── event_bus.py              # emit() + emit_sequence()
│   ├── classifier.py             # single-hand 42-float classifier
│   ├── two_hand_classifier.py    # two-hand 84-float classifier
│   ├── normalizer.py             # normalize_landmarks(), normalize_two_hand_landmarks(), HAND_CONNECTIONS
│   ├── sequence_event.py         # SequenceEvent dataclass
│   ├── sequence_recogniser.py    # SequenceRecogniser(BaseAdapter)
│   └── __init__.py               # re-exports all of the above
├── adapters/
│   ├── websocket_adapter.py      # on_gesture() + on_sequence() — POSTs to /gesture and /sequence
│   ├── pc_adapter.py
│   └── mqtt_adapter.py
├── api/
│   ├── server.py                 # POST /gesture, POST /sequence, WS /ws/gestures
│   ├── connection_manager.py     # broadcast() + broadcast_json()
│   └── routes/
│       ├── dataset.py            # CameraSession + session/* + stats/cameras/upload/delete
│       ├── training.py           # single + two_hand model training
│       ├── config.py             # actions + thresholds
│       └── sequences.py          # full CRUD + timing
├── config/
│   ├── gesture_actions.json      # static_actions (10 tokens) + sequence_actions
│   ├── thresholds.json           # pc_adapter, websocket_adapter, sequence_model timing
│   └── sequences.json            # user-defined sequence patterns
├── data/
│   ├── gestures.csv              # single-hand training data (label + 42 floats)
│   ├── gestures/                 # per-label .npy files (single-hand)
│   ├── gestures_two_hand.csv     # two-hand training data (label + 84 floats)
│   └── gestures_two_hand/        # per-label .npy files (two-hand)
├── models/
│   ├── classifier.pkl            # single-hand model
│   └── classifier_two_hand.pkl   # two-hand model (optional — pipeline falls back if absent)
├── lab/
│   ├── augment.py                # AugmentationEngine (42-float only)
│   └── augment_gesture.py        # CLI wrapper
└── frontend/
    ├── dashboard.html
    ├── studio.html               # Dataset/Train/Config/Sequences tabs
    └── shared/
        ├── nav.js
        └── styles.css
```
