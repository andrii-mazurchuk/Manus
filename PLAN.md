# Manus — Hackathon Plan

> 3-day sprint · 2 people · goal: working MVP + genuine understanding of what was built

Role tags: **[BOTH]** = work together, **[A]** = pipeline person, **[B]** = backend/frontend person.
These are defaults — collapse to [BOTH] any time it makes sense.

---

## Phase 0 — Tonight (Pre-hackathon)

Do this before the first morning. Losing time to environment issues at 9am is the most avoidable failure mode.

| Task | Who | Done? |
|---|---|---|
| Create repo, push initial files | BOTH | |
| Install and verify: `mediapipe`, `opencv-python`, `scikit-learn`, `fastapi`, `uvicorn`, `pyautogui`, `pynput`, `paho-mqtt` | BOTH | |
| Confirm webcam is accessible via OpenCV (`cv2.VideoCapture(0)`) | BOTH | |
| Read through `INTERFACE_CONTRACT.md` and agree on all definitions | BOTH | |
| Skim MediaPipe Hands landmark diagram — know what indices 0–20 mean | BOTH | |

**Checkpoint:** both machines can open a webcam window. Contract is agreed.

---

## Phase 1 — Day 1: Pipeline + Classifier

**Goal:** gesture → label printed live in terminal.

### Morning — Shared: MediaPipe + Data Collection

| Task | Who | Notes |
|---|---|---|
| Scaffold repo structure (folders: `core/`, `adapters/`, `api/`, `data/`, `frontend/`) | BOTH | |
| Build `capture.py`: open webcam, run MediaPipe Hands, draw landmarks, print raw landmark array | BOTH | Work through this together — make sure both understand the 21-point model |
| **Learning checkpoint:** each person explains what landmarks 0, 4, 8, 12, 16, 20 represent (wrist + fingertips) | BOTH | Don't skip this |
| Build `data_collector.py`: press a key to snapshot landmark array + label → append row to `data/gestures.csv` | BOTH | See CSV schema in contract |
| Collect data: 100–150 samples per gesture, both people contributing | BOTH | Vary hand position, distance, lighting. Target all 6 MVP gestures first, add `MODE`/`CUSTOM` if time allows |

**Risk:** if 150 samples/gesture takes too long → drop to 80. Volume matters less than variety.

### Afternoon — Split: Classifier (A) + API Skeleton (B)

**Person A — Classifier**

| Task | Notes |
|---|---|
| Load CSV, inspect class balance, plot sample counts per label | Don't skip inspection — this is where you catch bad data |
| Engineer features: normalize all 21 landmarks relative to wrist (landmark 0), scale to unit bounding box | See normalization spec in contract |
| **Learning checkpoint:** explain why normalization is necessary before training | |
| Train RandomForest and MLP, compare accuracy. Target ≥ 88% | Use `cross_val_score`, not just train/test split |
| Serialize winning model to `models/classifier.pkl` | |
| Wire into live pipeline: `capture.py` → normalize → predict → print label + confidence | |

**Person B — API Skeleton**

| Task | Notes |
|---|---|
| Scaffold FastAPI app in `api/server.py` | |
| Implement `POST /gesture` endpoint accepting `GestureEvent` JSON | Use the contract schema exactly |
| Implement `WebSocket /ws/gestures` endpoint | Start with a mock emitter that sends fake events on a timer — don't wait for A's pipeline |
| Verify WebSocket works with a simple HTML test page | |
| Read `INTERFACE_CONTRACT.md` `BaseAdapter` spec — implement the `WebSocketAdapter` stub | |

### Evening — Integration

| Task | Who | Notes |
|---|---|---|
| Build `core/event_bus.py`: simple pub/sub, ~30 lines | A | Subscribers register; bus calls `on_gesture()` on each |
| Build `core/base_adapter.py`: `BaseAdapter` ABC | A | |
| Wire live pipeline into event bus | A | |
| Replace mock emitter in WebSocket endpoint with real event bus subscription | BOTH | First real integration point — do this together |
| Smoke test: wave hand, see label appear in terminal and over WebSocket simultaneously | BOTH | |

**Day 1 Checkpoint:** terminal prints live gesture label. WebSocket endpoint emits real events.

---

## Phase 2 — Day 2: Adapters + Dashboard

**Goal:** a gesture controls something real; API is documented; dashboard is live.

### Morning — PC Adapter (A) + FastAPI Hardening (B)

**Person A — PC Adapter**

| Task | Notes |
|---|---|
| Implement `adapters/pc_adapter.py` subclassing `BaseAdapter` | |
| Map tokens → pyautogui/pynput actions: `STOP`→mute, `PLAY`→unmute, `UP`→vol up, `DOWN`→vol down, `CONFIRM`→next slide, `CANCEL`→prev slide | |
| Add confidence threshold (ignore events below 0.70) | In the adapter, not the bus |
| Add 500ms debounce per token | In the adapter |
| **Learning checkpoint:** explain why debounce belongs in the adapter, not the classifier | |
| Test on a real presentation or YouTube video | |

**Person B — FastAPI Hardening**

| Task | Notes |
|---|---|
| Add input validation to `POST /gesture` (Pydantic model matching contract) | |
| Add `GET /status` endpoint returning adapter list and connection count | |
| Confirm `/docs` auto-generated OpenAPI page works and is accurate | |
| Add CORS middleware (needed for JS dashboard on a different port) | |

### Afternoon — MQTT Stub (A) + JS Dashboard (B)

**Person A — MQTT Adapter**

| Task | Notes |
|---|---|
| Implement `adapters/mqtt_adapter.py` stub | Publish `{ gesture, confidence, timestamp }` to a topic |
| Test with a local MQTT broker (Mosquitto) if available | If no hardware: stub is enough for MVP |
| Write adapter registration / dynamic loading in `main.py` | Config-driven: list of adapter class names to load |

**Person B — JS Dashboard**

| Task | Notes |
|---|---|
| Build `frontend/index.html` — connects to `ws://localhost:8000/ws/gestures` | |
| Display: current gesture label (large), confidence bar, rolling event log | |
| Show adapter status panel (pull from `GET /status`) | |
| No build step — plain JS is fine. Keep it simple | |

### Evening — Integration + Polish Pass

| Task | Who | Notes |
|---|---|---|
| Run full stack end-to-end: webcam → classifier → event bus → PC adapter + WebSocket adapter → dashboard | BOTH | |
| Fix any integration bugs found | BOTH | |
| Tune confidence threshold and debounce values based on live feel | BOTH | |

**Day 2 Checkpoint:** gesture controls laptop volume. Dashboard shows live feed. `/docs` is up.

---

## Phase 3 — Day 3: Polish, Docs, Demo

**Goal:** something shippable and demonstrable.

### Morning — Hardening

| Task | Who | Notes |
|---|---|---|
| Re-collect data for any gesture with accuracy < 88% | BOTH | Add more samples, don't retune hyperparameters first |
| Add graceful error handling: no webcam, model file missing, WebSocket disconnect/reconnect | A+B | |
| Test on both machines | BOTH | Catches any machine-specific issues |

### Afternoon — Docs + Demo

| Task | Who | Notes |
|---|---|---|
| Write `README.md`: what it is, quickstart (3 commands), architecture diagram, "add a custom adapter in 10 lines" tutorial | BOTH | The tutorial is the most important part — write it from scratch, not from memory |
| **Learning checkpoint:** write the adapter tutorial without looking at `BaseAdapter` code. If you can't, read it again | BOTH | |
| Record 2-minute demo video: PC control + live dashboard | BOTH | |
| Push to public GitHub | BOTH | |

### Evening — Buffer

Use for anything that slipped. In priority order:
1. Anything from Day 2 that isn't working
2. Smart home demo (if TP-Link plug was purchased)
3. Gesture sequence modelling (if everything else is done)

**Day 3 Checkpoint:** public repo, demo video, working API.

---

## Risk Register

| Risk | Trigger | Fallback |
|---|---|---|
| Accuracy < 88% by Day 1 evening | Cross-val score below threshold | Drop to 5 gestures (remove `MODE` and `CUSTOM`) |
| Data collection takes > 2 hours | Still collecting at noon | Cap at 80 samples/gesture and move on |
| pyautogui blocks on one OS | Volume control not working | Swap to keyboard shortcut approach via `pynput` |
| WebSocket integration broken | Events not reaching dashboard | Isolate: test bus → adapter independently before blaming the WS layer |
| Machine-specific webcam issue | `cv2.VideoCapture(0)` fails | Try index 1, or use `ffmpeg` to identify device index |

---

## MVP Checklist

- [ ] 6+ gestures recognized at ≥ 88% accuracy
- [ ] Live terminal output: label + confidence
- [ ] PC adapter: volume up/down, mute/unmute, next/prev slide
- [ ] `POST /gesture` and `WebSocket /ws/gestures` endpoints live
- [ ] JS dashboard connected and rendering live events
- [ ] README with 10-line adapter tutorial
