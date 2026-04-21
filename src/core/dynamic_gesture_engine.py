"""
DynamicGestureEngine — LSTM-based continuous gesture state machine.

Registered on the EventBus as a BaseAdapter. Watches GestureEvents emitted by
the static classifier; when a configured trigger token fires with sufficient
confidence, it arms and starts accumulating raw 63-float landmark frames. The
LSTM classifies the accumulated sequence and emits a SequenceEvent.

State machine:
    WATCHING  ─(trigger token, conf >= arm_threshold)─► CAPTURING
    CAPTURING ─(EMA max >= 0.80 OR 60 frames)─────────► COOLDOWN (emit SequenceEvent)
    CAPTURING ─(hand lost > 0.5 s)────────────────────► WATCHING  (silent abort)
    COOLDOWN  ─(1.5 s elapsed)────────────────────────► WATCHING

main.py usage:
    dge = DynamicGestureEngine(classifier=seq_clf, trigger_token=GestureToken.UP)
    bus.register(dge)

    # Inside the frame loop, after static classification:
    if result.hand_landmarks:
        dge.feed_frame(normalize_landmarks_xyz(primary))
    else:
        dge.notify_no_hand()
"""

from __future__ import annotations

import enum
import threading
import time

import numpy as np

from .base_adapter import BaseAdapter
from .event_bus import EventBus
from .gesture_event import GestureEvent, GestureToken
from .sequence_event import SequenceEvent

_COOLDOWN_S         = 1.5
_MAX_FRAMES         = 60
_EMA_ALPHA          = 0.45
_FIRE_THRESHOLD     = 0.80
_LOST_HAND_TIMEOUT  = 0.5   # seconds without a frame before aborting capture
_MIN_FRAMES_EMA     = 8     # minimum frames before EMA scoring starts
_EMA_CHECK_INTERVAL = 5     # run EMA every N frames


class _State(enum.Enum):
    WATCHING  = "watching"
    CAPTURING = "capturing"
    COOLDOWN  = "cooldown"


class DynamicGestureEngine(BaseAdapter):
    """
    EventBus adapter + frame consumer for LSTM dynamic gesture classification.

    Thread-safety: on_gesture() is called from the EventBus thread; feed_frame()
    and notify_no_hand() are called from the main camera loop thread. All shared
    state is protected by self._lock.
    """

    def __init__(
        self,
        classifier,                    # SequenceClassifier | None
        trigger_token: GestureToken,
        arm_threshold: float = 0.75,
    ) -> None:
        self._clf            = classifier
        self._trigger_token  = trigger_token
        self._arm_threshold  = arm_threshold

        self._lock           = threading.Lock()
        self._state          = _State.WATCHING
        self._buffer: list[np.ndarray] = []
        self._ema_probs: np.ndarray | None = None
        self._cooldown_start = 0.0
        self._last_frame_ts  = 0.0
        self._capture_start_ts = 0.0

    # ── BaseAdapter ────────────────────────────────────────────────────────────

    def on_gesture(self, event: GestureEvent) -> None:
        """Called by EventBus for every GestureEvent from the static classifier."""
        with self._lock:
            if self._state is not _State.WATCHING:
                return
            if (
                event.gesture == self._trigger_token
                and event.confidence >= self._arm_threshold
            ):
                self._enter_capturing()

    # ── Called by main.py frame loop ──────────────────────────────────────────

    def feed_frame(self, frame_63: np.ndarray) -> None:
        """
        Append one 63-float landmark frame to the capture buffer.

        Checks EMA early-exit every _EMA_CHECK_INTERVAL frames once
        _MIN_FRAMES_EMA are accumulated. Must be called every frame when a hand
        is detected in the frame.
        """
        with self._lock:
            now = time.monotonic()

            if self._state is _State.COOLDOWN:
                if now - self._cooldown_start >= _COOLDOWN_S:
                    self._state = _State.WATCHING
                return

            if self._state is not _State.CAPTURING:
                return

            self._last_frame_ts = now
            self._buffer.append(frame_63.copy())

            if len(self._buffer) >= _MAX_FRAMES:
                self._fire()
                return

            n = len(self._buffer)
            if (
                n >= _MIN_FRAMES_EMA
                and n % _EMA_CHECK_INTERVAL == 0
                and self._clf is not None
            ):
                frames = np.stack(self._buffer)
                new_probs = self._clf._raw_probs(frames)
                if self._ema_probs is None:
                    self._ema_probs = new_probs
                else:
                    self._ema_probs = (
                        _EMA_ALPHA * new_probs + (1 - _EMA_ALPHA) * self._ema_probs
                    )
                if float(np.max(self._ema_probs)) >= _FIRE_THRESHOLD:
                    self._fire()

    def notify_no_hand(self) -> None:
        """Call every frame when no hand is detected. Aborts capture if hand gone too long."""
        with self._lock:
            if self._state is not _State.CAPTURING:
                return
            if time.monotonic() - self._last_frame_ts > _LOST_HAND_TIMEOUT:
                self._reset_buffer()
                self._state = _State.WATCHING

    def get_state(self) -> str:
        with self._lock:
            return self._state.value

    def reload_config(self) -> None:
        """Re-read dynamic_gestures.json and update trigger/threshold live."""
        from src.config.loader import load_dynamic_gestures_config
        cfg = load_dynamic_gestures_config()
        with self._lock:
            try:
                self._trigger_token = GestureToken(cfg["trigger_token"])
            except ValueError:
                pass  # keep existing token if config has an invalid value
            self._arm_threshold = float(cfg.get("arm_threshold", 0.75))

    # ── Internal ───────────────────────────────────────────────────────────────

    def _enter_capturing(self) -> None:
        self._reset_buffer()
        self._state = _State.CAPTURING
        self._last_frame_ts = time.monotonic()
        self._capture_start_ts = time.time()

    def _fire(self) -> None:
        """Classify buffer, emit SequenceEvent via background thread, enter COOLDOWN."""
        if not self._buffer or self._clf is None:
            self._reset_buffer()
            self._state = _State.WATCHING
            return

        frames     = np.stack(self._buffer)
        label, confidence = self._clf.predict(frames)
        duration   = round(len(self._buffer) / 30.0, 3)
        event      = SequenceEvent(
            name=label,
            tokens=[],
            confidence=confidence,
            timestamp=time.time(),
            duration=duration,
            source="lstm",
        )

        self._reset_buffer()
        self._state = _State.COOLDOWN
        self._cooldown_start = time.monotonic()

        # Emit off the lock in a daemon thread — WebSocketAdapter.on_sequence()
        # makes a blocking HTTP POST that would stall the camera loop.
        threading.Thread(
            target=EventBus.get().emit_sequence,
            args=(event,),
            daemon=True,
        ).start()

    def _reset_buffer(self) -> None:
        self._buffer.clear()
        self._ema_probs = None
