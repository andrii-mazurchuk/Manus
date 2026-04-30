# GestureOS — Hackathon Project Brief

> A real-time hand gesture recognition engine with a universal adapter layer.
> 3-day local hackathon · 2 people · budget ≤ $50

---

## What we're building

A Python-based pipeline that reads hand gestures from a webcam in real time, classifies them into named commands (e.g. "fist", "thumbs up", "open palm"), and routes those commands to any connected system through a clean adapter interface.

The key idea is the **adapter layer**: the gesture engine emits typed events. Each adapter — PC control, REST API, WebSocket, MQTT — just subscribes to those events and acts on them. This means the same gesture vocabulary can control a laptop, trigger a webhook, update a live dashboard, or talk to a smart home device without touching the core pipeline.

We are not just building a gesture detector. We are building a **framework** that others (and our future selves) can plug into any product in about 10 lines of Python.

---

## What this project aims for

**Technical goals:**
- Understand MediaPipe Hands and how production-grade landmark models work
- Build and own a full ML pipeline: data collection → feature engineering → model training → real-time inference
- Practice a plugin/adapter architecture using Python abstract base classes
- Learn FastAPI with WebSockets for real-time async Python
- Build a small but real JS frontend that consumes a live data stream

**Product goals:**
- Ship something that works and can be demonstrated to anyone in under 2 minutes
- Produce a reusable open-source framework, not a one-off script
- End the weekend with a public GitHub repo and a recorded demo video

---

## MVP definition

The MVP is exactly these six things. Nothing more is required to call the project done:

1. Real-time webcam feed → MediaPipe landmarks → classifier → gesture label with confidence printed live
2. At least **6 gestures** recognized reliably at ≥ 88% accuracy
3. **PC adapter** working: volume up/down, next/previous slide, mute/unmute via detected gestures
4. **REST/WebSocket API** exposing a live stream of `{ gesture, confidence, timestamp }` JSON
5. **JS dashboard** that connects to the WebSocket and displays the live gesture feed
6. A **README** explaining how to implement a new adapter in 10 lines of Python

Everything else is bonus territory for day 3.

---

## Gesture vocabulary

| Gesture | Command token | Default mapping |
|---|---|---|
| ✊ Fist | `STOP` | Mute / pause |
| 🖐 Open palm | `PLAY` | Unmute / resume |
| ☝️ Index up | `UP` | Volume up / scroll up |
| 👇 Index down | `DOWN` | Volume down / scroll down |
| 👍 Thumbs up | `CONFIRM` | Next slide / confirm |
| 👎 Thumbs down | `CANCEL` | Previous slide / cancel |
| ✌️ Peace / V | `MODE` | Switch adapter mode |
| 🤙 Shaka | `CUSTOM` | User-defined |

Command tokens are what the event bus emits. Adapters decide what each token *does* in their context.

---

## System architecture

```
┌─────────────────────────────────────────────────┐
│  INPUT LAYER                                    │
│  Webcam  →  MediaPipe Hands (21 landmarks)      │
└──────────────────────┬──────────────────────────┘
                       │ landmark array (42 floats)
┌──────────────────────▼──────────────────────────┐
│  CORE LAYER                                     │
│  Gesture classifier  →  Event bus               │
│  (RandomForest / MLP)    GestureEvent(label,    │
│                           confidence, ts)        │
└──────────────────────┬──────────────────────────┘
                       │ typed events
         ┌─────────────┼──────────────┬────────────┐
┌────────▼──────┐ ┌────▼────────┐ ┌──▼──────┐ ┌──▼──────┐
│  PC adapter   │ │ REST adapter│ │  WS     │ │  MQTT   │
│  pyautogui    │ │  FastAPI    │ │ adapter │ │ adapter │
│  pynput       │ │  webhooks   │ │  JS     │ │ smart   │
└───────────────┘ └─────────────┘ │dashboard│ │ home /  │
                                  └─────────┘ │ robot   │
                                              └─────────┘
```

The `BaseAdapter` abstract class defines a single method: `on_gesture(event: GestureEvent) -> None`. Writing a new adapter means subclassing it and implementing that one method.

---

## Tech stack

| Layer | Technology | Why |
|---|---|---|
| Hand tracking | **MediaPipe Hands** | Production-grade, free, runs locally on CPU/GPU, 21 landmarks at ~5ms per frame |
| Classification | **scikit-learn** (RandomForest or MLP) | Fast to train, interpretable, more than accurate enough for 8–10 gestures |
| Backend / API | **FastAPI** | Async Python, WebSocket support built in, auto-generates OpenAPI docs |
| Event bus | Custom Python pub/sub (~30 lines) | No dependencies, teaches the pattern clearly |
| Adapters | **pyautogui / pynput** (PC), **paho-mqtt** (smart home), plain HTTP (webhooks) | Minimal, purpose-built |
| Frontend dashboard | **Vanilla JS or React** | WebSocket consumer, live gesture feed display |
| Data format | CSV of 42 landmark floats + label | Simple, portable, inspectable |
| Language | **Python 3.11+** (backend) · **JS** (frontend) | Our primary stack |

---

## 3-day schedule

### Day 1 — Pipeline + classifier
**Goal:** gesture → label working end-to-end in the terminal

| Block | Task |
|---|---|
| Morning | Set up repo and environment. Install MediaPipe + FastAPI. Get webcam feed with 21 landmarks overlaid on screen. Understand what each of the 21 landmark indices represents. |
| Afternoon | Build the data collector: press a key to label and save landmark snapshots as CSV rows. Record 100–150 samples per gesture across both team members (different hands, lighting, distances). |
| Evening | Train classifier on collected CSV. Target ≥ 90% accuracy. If struggling: reduce gesture count — don't chase accuracy at the cost of time. |

**Day 1 checkpoint:** terminal prints gesture label in real time from live webcam feed.

---

### Day 2 — Adapters + API
**Goal:** gesture controls something real; API is live

| Block | Task |
|---|---|
| Morning | Build the event bus (plain Python, ~30 lines). Implement the PC adapter: fist → mute, open palm → unmute, thumbs up → next slide, using pyautogui/pynput. Test it on a real presentation. |
| Afternoon | Build FastAPI server: `POST /gesture` (webhook push) and `WebSocket /ws/gestures` (live stream). Define `BaseAdapter` as an abstract base class. Write the MQTT adapter stub if hardware is available. |
| Evening | Build JS dashboard: connects via WebSocket, shows live gesture label and confidence bar, adapter status panel. |

**Day 2 checkpoint:** REST API is documented at `/docs`, a gesture controls laptop volume, dashboard renders live data.

---

### Day 3 — Polish, docs, demo
**Goal:** something shippable and demonstrable

| Block | Task |
|---|---|
| Morning | Harden the classifier: add a confidence threshold (ignore detections below 0.70), add a 500ms debounce so the same gesture doesn't fire twice in quick succession. These two changes fix most UX issues. |
| Afternoon | Write the README with architecture diagram, quickstart guide, and "add a custom adapter in 10 lines" tutorial. Record a 2-minute demo video showing PC control + live dashboard. |
| Evening | **Buffer.** Use for anything unfinished from day 2, the smart home demo if hardware was purchased, or — if everything is done — experiment with temporal sequence modelling (gesture sequences over time rather than single frames). |

**Day 3 checkpoint:** public GitHub repo, working demo video, API that can be handed to another developer.

---

## Work split

Split by layer, not by day — both people work in parallel from end of day 1 once the `GestureEvent` interface is agreed on.

| Person A | Person B |
|---|---|
| Python pipeline: MediaPipe → classifier → event bus → adapters | FastAPI server + WebSocket layer + JS dashboard |

The interface contract between the two halves is exactly: `GestureEvent(label: str, confidence: float, timestamp: float)`.

---

## Budget

| Option | Cost | What it buys | Verdict |
|---|---|---|---|
| Nothing | $0 | Full MVP as described | Sufficient — do this by default |
| USB smart plug (e.g. TP-Link Tapo P100) | ~$15 | Control a real-world device — turns the demo from "API output" to "hand gesture turned the light on" | Recommended if demo impact matters |
| Small cloud API budget (OpenAI / Anthropic) | ~$10–20 | Add "gesture sequence → natural language intent" LLM layer | Interesting ML concept, not required for MVP |
| Depth camera (RealSense) | $150+ | Better 3D landmark accuracy | Way over budget, not needed — MediaPipe 2D is accurate enough |

**Recommendation:** start with $0. If the demo feels too abstract on day 2 evening, buy the smart plug.

---

## What we will learn

| Concept | Where it appears |
|---|---|
| MediaPipe landmark models | Input layer — understanding Google's production hand tracking approach |
| Feature engineering for ML | Normalising coordinates relative to wrist, computing finger angle ratios |
| Plugin / adapter architecture | `BaseAdapter` abstract class, dynamic adapter registration |
| FastAPI + async Python | REST endpoints, WebSocket handler, background inference loop |
| Real-time data streaming | WebSocket protocol, JS `EventSource`, live dashboard state |
| Full ML pipeline ownership | Data collection → CSV → training → serialisation → inference in production |

---

## What "done" looks like

By end of day 3, we have:

- A working webcam demo where gestures control a laptop in real time
- A live web dashboard showing gesture events as they happen
- A REST/WebSocket API that any developer can connect to
- A clean Python codebase where adding a new integration takes ~10 lines
- A public repo with a README, architecture diagram, and demo video
- A project that demonstrates ML pipeline thinking, software architecture, and real-time systems — all relevant for data science and ML roles

---

---

## Phase 4 — Web Platform (post-hackathon)

The MVP is delivered. Phase 4 turns the CLI-operated prototype into a self-contained web platform.

**Phase 4a — Infrastructure**
FastAPI serves the frontend as static files. `dashboard.html` (renamed from `index.html`) gets a shared nav bar. A new `studio.html` page is introduced with a two-tab structure: **Static Gestures** and **Sequences** (placeholder). API routes are modularised into a `routes/` package.

**Phase 4b — Studio: Dataset**
Camera selection dropdown (enumerates available OpenCV devices — designed for multi-camera smart home setups). Live MJPEG webcam preview. Capture N labeled samples from the UI. Bulk import via zip upload (LeapGestRecog-compatible structure).

**Phase 4c — Studio: Training**
Train button triggers background model training; frontend polls progress. Per-class accuracy shown after training. Training history log.

**Phase 4d — Studio: Config**
Gesture→action mapping editor (which gesture does what in the PC adapter). Per-adapter confidence threshold sliders. Persisted to `src/config/gesture_actions.json` and `src/config/thresholds.json`. Config is structured with `static_actions` and `sequence_actions` sections from day one.

**Future pages (nav placeholders from Phase 4a):**
- Statistics — which camera fired which gesture most; event heatmaps over time
- Health — registered device status, camera availability, adapter connectivity

---

## Phase 5 — Gesture Sequence Recognition (planned)

Recognise **temporal patterns of existing gesture tokens** as named sequence events. Examples:
- UP → DOWN → UP → DOWN = "wave"
- CONFIRM → CONFIRM = "double confirm"
- Snap of fingers (rapid appearance and disappearance of a hand landmark) = custom trigger

**Design decisions already locked in:**
- Sequences are patterns of existing tokens — no new vocabulary is required. The sequence recogniser subscribes to the EventBus output stream.
- The token vocabulary will grow past the current 8 tokens. The system is built to accommodate this.
- `gesture_actions.json` has a `sequence_actions` section reserved. `thresholds.json` has a `sequence_model` key reserved.
- The Studio Sequences tab is already present as a placeholder — implementing Phase 5 means filling it in, not restructuring the UI.

*GestureOS — local hackathon, expanded post-hackathon*
