# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Manus is a real-time hand gesture recognition framework with a pluggable adapter layer. It reads webcam input, classifies hand gestures via a trained ML model, and routes typed `GestureEvent` objects to any registered adapter (PC control, REST/WebSocket API, MQTT, etc.).

See `GestureOS_ProjectBrief.md` for full scope, gesture vocabulary, 3-day schedule, and work-split.

---

## Architecture

```
Webcam → MediaPipe Hands (21 landmarks / 42 floats)
       → Gesture Classifier (RandomForest or MLP, scikit-learn)
       → Event Bus (custom pub/sub, ~30 lines)
       → Adapters (PC / REST / WebSocket / MQTT)
```

**Core contract between layers:**
```python
GestureEvent(label: str, confidence: float, timestamp: float)
```

**Adapter interface** — every adapter subclasses `BaseAdapter` and implements one method:
```python
class BaseAdapter(ABC):
    def on_gesture(self, event: GestureEvent) -> None: ...
```

The event bus is plain Python (no external dependency). Adapters register themselves with the bus and receive every emitted `GestureEvent`.

---

## Tech Stack

| Concern | Library |
|---|---|
| Hand tracking | `mediapipe` (21 landmarks, CPU/GPU) |
| Classification | `scikit-learn` (RandomForest / MLP) |
| Backend / API | `FastAPI` with `uvicorn` |
| PC control | `pyautogui`, `pynput` |
| MQTT | `paho-mqtt` |
| Frontend | Vanilla JS or React (WebSocket consumer) |
| Data | CSV — 42 landmark floats + label column |
| Python version | 3.11+ |

---

## Planned Commands

Once the project is scaffolded, expected commands will include:

```bash
# Install dependencies
pip install -r requirements.txt

# Collect training data (interactive — press key to label each snapshot)
python data_collector.py

# Train classifier from CSV
python train.py

# Run the full pipeline (webcam → classifier → adapters)
python main.py

# Run the FastAPI server
uvicorn api.server:app --reload

# Run tests
pytest
```

---

## Key Design Decisions

- **Confidence threshold:** ignore detections below 0.70 to reduce false positives.
- **Debounce:** 500 ms minimum between successive firings of the same gesture token.
- **Training data format:** CSV rows of 42 normalized landmark floats (relative to wrist position) plus a string label. Normalize coordinates before saving so the classifier is hand-position invariant.
- **Gesture tokens** are the canonical string labels emitted by the event bus: `STOP`, `PLAY`, `UP`, `DOWN`, `CONFIRM`, `CANCEL`, `MODE`, `CUSTOM`. Adapters map tokens to actions — the classifier never knows about PC or MQTT specifics.
- **Work split:** Person A owns the Python pipeline (MediaPipe → classifier → event bus → adapters); Person B owns FastAPI + WebSocket + JS dashboard. The boundary is the `GestureEvent` dataclass.
