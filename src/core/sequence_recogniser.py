"""
SequenceRecogniser — subscribes to the EventBus and emits SequenceEvents.

Architecture
------------
The recogniser is registered as a BaseAdapter on the EventBus. On every
on_gesture() call it:

  1. Appends the event to a rolling deque (buffer_size entries max).
  2. Deduplicates consecutive identical tokens: if the same token fires again
     within max_gap_ms it collapses to the highest-confidence occurrence.
     This prevents 30-fps repeated frames from poisoning the buffer.
  3. Scans all defined sequences against the tail of the buffer.
  4. On a match, emits a SequenceEvent via EventBus.emit_sequence().

Sequence definitions are loaded from src/config/sequences.json at startup
and reloaded on each call to reload_sequences(). The API layer calls this
after writing a new definition via the Studio UI.

Timing constants are read from src/config/thresholds.json under the
"sequence_model" key. Per-sequence overrides (max_gap_ms, max_total_ms)
take precedence over the global defaults.

Upgrade path
------------
When raw-landmark temporal classification is needed, replace the pattern
matching internals here. The SequenceEvent output format, the EventBus
wiring, and all downstream consumers remain unchanged.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

from .base_adapter import BaseAdapter
from .event_bus import EventBus
from .gesture_event import GestureEvent, GestureToken
from .sequence_event import SequenceEvent

_SEQUENCES_PATH  = Path("src/config/sequences.json")
_THRESHOLDS_PATH = Path("src/config/thresholds.json")

# Fallback timing constants (used when config files are absent or malformed).
_DEFAULT_MAX_GAP_MS   = 800
_DEFAULT_MAX_TOTAL_MS = 3000
_DEFAULT_BUFFER_SIZE  = 10


def _load_timing_defaults() -> dict[str, Any]:
    """Read global timing defaults from thresholds.json."""
    try:
        with open(_THRESHOLDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        sm = data.get("sequence_model") or {}
        if not isinstance(sm, dict):
            sm = {}
        return {
            "default_max_gap_ms":   int(sm.get("default_max_gap_ms",   _DEFAULT_MAX_GAP_MS)),
            "default_max_total_ms": int(sm.get("default_max_total_ms", _DEFAULT_MAX_TOTAL_MS)),
            "buffer_size":          int(sm.get("buffer_size",          _DEFAULT_BUFFER_SIZE)),
        }
    except Exception:
        return {
            "default_max_gap_ms":   _DEFAULT_MAX_GAP_MS,
            "default_max_total_ms": _DEFAULT_MAX_TOTAL_MS,
            "buffer_size":          _DEFAULT_BUFFER_SIZE,
        }


def _load_sequences() -> list[dict]:
    """Read sequence definitions from sequences.json."""
    try:
        with open(_SEQUENCES_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sequences", [])
    except Exception:
        return []


class SequenceRecogniser(BaseAdapter):
    """
    EventBus subscriber that detects token-stream sequence patterns.

    Register via EventBus.get().register(SequenceRecogniser()) in main.py.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._timing = _load_timing_defaults()
        self._sequences: list[dict] = _load_sequences()
        self._buffer: deque[tuple[GestureEvent, float]] = deque(
            maxlen=self._timing["buffer_size"]
        )
        # Cooldown: prevent the same sequence from firing multiple times in
        # quick succession. Maps sequence name → last fired timestamp.
        self._last_fired: dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def reload_sequences(self) -> None:
        """Reload definitions from sequences.json (called after Studio edits)."""
        with self._lock:
            self._timing = _load_timing_defaults()
            self._sequences = _load_sequences()
            self._buffer = deque(maxlen=self._timing["buffer_size"])

    @property
    def sequence_count(self) -> int:
        with self._lock:
            return len(self._sequences)

    # ── BaseAdapter ────────────────────────────────────────────────────────

    def on_gesture(self, event: GestureEvent) -> None:
        with self._lock:
            self._append_deduplicated(event)
            matches = self._find_matches()

        bus = EventBus.get()
        for seq_event in matches:
            bus.emit_sequence(seq_event)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _append_deduplicated(self, event: GestureEvent) -> None:
        """
        Add event to buffer, collapsing consecutive identical tokens.

        If the most recent entry in the buffer is the same token and arrived
        within default_max_gap_ms, replace it with the higher-confidence
        version. This prevents a steady 30-fps stream of CLAP events from
        filling the buffer before a second CLAP arrives.
        """
        now = time.time()
        max_gap_s = self._timing["default_max_gap_ms"] / 1000.0

        if self._buffer:
            last_event, last_ts = self._buffer[-1]
            if (
                last_event.gesture == event.gesture
                and (now - last_ts) < max_gap_s
            ):
                # Replace with higher-confidence occurrence.
                if event.confidence >= last_event.confidence:
                    self._buffer[-1] = (event, last_ts)  # keep original timestamp
                return  # either way, don't add a duplicate entry

        self._buffer.append((event, now))

    def _find_matches(self) -> list[SequenceEvent]:
        """
        Scan all defined sequences against the current buffer tail.

        Returns a list of SequenceEvent objects for all patterns that match
        at the current moment. Normally 0 or 1; multiple simultaneous matches
        are possible if sequences share a common prefix.
        """
        results: list[SequenceEvent] = []
        buffer_list = list(self._buffer)
        now = time.time()

        for seq_def in self._sequences:
            pattern = seq_def.get("pattern", [])
            if len(pattern) < 2:
                continue  # single-token patterns are just gestures

            max_gap_ms   = seq_def.get("max_gap_ms",   self._timing["default_max_gap_ms"])
            max_total_ms = seq_def.get("max_total_ms",  self._timing["default_max_total_ms"])
            max_gap_s    = max_gap_ms   / 1000.0
            max_total_s  = max_total_ms / 1000.0
            name         = seq_def.get("name", "unnamed")

            # Cooldown: skip if this sequence fired very recently.
            cooldown_s = max_total_s
            if now - self._last_fired.get(name, 0.0) < cooldown_s:
                continue

            match = self._match_pattern(buffer_list, pattern, max_gap_s, max_total_s)
            if match is not None:
                matched_events, first_ts, last_ts = match
                confidence = sum(e.confidence for e in matched_events) / len(matched_events)
                seq_event = SequenceEvent(
                    name=name,
                    tokens=[e.gesture for e in matched_events],
                    confidence=confidence,
                    timestamp=last_ts,
                    duration=round(last_ts - first_ts, 4),
                )
                results.append(seq_event)
                self._last_fired[name] = now

        return results

    @staticmethod
    def _match_pattern(
        buffer: list[tuple[GestureEvent, float]],
        pattern: list[str],
        max_gap_s: float,
        max_total_s: float,
    ) -> tuple[list[GestureEvent], float, float] | None:
        """
        Attempt to find `pattern` at the tail of `buffer`.

        Returns (matched_events, first_timestamp, last_timestamp) on success,
        None if the pattern does not match or timing constraints are violated.
        """
        n = len(pattern)
        if len(buffer) < n:
            return None

        # Check the last n buffer entries against the pattern.
        tail = buffer[-n:]
        for i, (event, _) in enumerate(tail):
            if event.gesture.value != pattern[i]:
                return None

        # Timing checks.
        timestamps = [ts for _, ts in tail]
        first_ts = timestamps[0]
        last_ts  = timestamps[-1]

        if (last_ts - first_ts) > max_total_s:
            return None

        for i in range(1, n):
            if (timestamps[i] - timestamps[i - 1]) > max_gap_s:
                return None

        matched_events = [ev for ev, _ in tail]
        return matched_events, first_ts, last_ts
