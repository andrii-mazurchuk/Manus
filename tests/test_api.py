"""
Tests for the Manus Gesture API.

Covers:
  - POST /gesture  — happy path, validation errors, enum validation
  - GET  /tokens   — returns canonical token list from GestureToken enum
  - WebSocket /ws/gestures — connect / disconnect / broadcast
  - Integration: POST triggers broadcast to connected WS client
"""

import json
import time
import pytest
from fastapi.testclient import TestClient

from src.api.server import app
from src.core.gesture_event import GestureToken

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Synchronous TestClient — safe for both HTTP and WebSocket tests."""
    with TestClient(app) as c:
        yield c


def _gesture_payload(
    gesture: str = "STOP",
    label: int = 0,
    timestamp: float | None = None,
    confidence: float = 0.9,
) -> dict:
    return {
        "gesture": gesture,
        "confidence": confidence,
        "label": label,
        "timestamp": timestamp if timestamp is not None else time.time(),
    }


# ---------------------------------------------------------------------------
# POST /gesture
# ---------------------------------------------------------------------------

class TestPostGesture:

    def test_valid_gesture_returns_ok(self, client: TestClient):
        payload = _gesture_payload("STOP", 0)
        response = client.post("/gesture", json=payload)

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_all_gesture_tokens(self, client: TestClient):
        tokens = ["STOP", "PLAY", "UP", "DOWN", "CONFIRM", "CANCEL", "MODE", "CUSTOM"]
        for token in tokens:
            resp = client.post("/gesture", json=_gesture_payload(token))
            assert resp.status_code == 200, f"Failed for token: {token}"

    def test_missing_gesture_field_returns_422(self, client: TestClient):
        payload = {"label": 0, "timestamp": time.time()}
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_missing_label_field_returns_422(self, client: TestClient):
        payload = {"gesture": "STOP", "timestamp": time.time()}
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_missing_timestamp_field_returns_422(self, client: TestClient):
        payload = {"gesture": "STOP", "label": 0}
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient):
        response = client.post("/gesture", json={})
        assert response.status_code == 422

    def test_wrong_type_label_returns_422(self, client: TestClient):
        payload = _gesture_payload()
        payload["label"] = "not-an-int"
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_wrong_type_timestamp_returns_422(self, client: TestClient):
        payload = _gesture_payload()
        payload["timestamp"] = "yesterday"
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_negative_label_accepted(self, client: TestClient):
        """Pydantic does not restrict label range by default — should be 200."""
        response = client.post("/gesture", json=_gesture_payload(label=-1))
        assert response.status_code == 200

    def test_zero_timestamp_accepted(self, client: TestClient):
        response = client.post("/gesture", json=_gesture_payload(timestamp=0.0))
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# WebSocket /ws/gestures
# ---------------------------------------------------------------------------

class TestWebSocket:

    def test_connect_and_disconnect(self, client: TestClient):
        """Client can open a connection and close it cleanly."""
        with client.websocket_connect("/ws/gestures") as ws:
            assert ws is not None
        # No exception → clean disconnect

    def test_multiple_clients_connect(self, client: TestClient):
        with client.websocket_connect("/ws/gestures") as ws1:
            with client.websocket_connect("/ws/gestures") as ws2:
                assert ws1 is not None
                assert ws2 is not None

    def test_broadcast_on_post(self, client: TestClient):
        """A POST /gesture should broadcast the event to every connected WS client."""
        payload = _gesture_payload("PLAY", label=1, timestamp=1234567890.0)

        with client.websocket_connect("/ws/gestures") as ws:
            client.post("/gesture", json=payload)
            message = ws.receive_text()

        data = json.loads(message)
        assert data["gesture"] == "PLAY"
        assert data["label"] == 1
        assert data["timestamp"] == pytest.approx(1234567890.0)

    def test_broadcast_reaches_all_clients(self, client: TestClient):
        """Both connected clients receive the same broadcast."""
        payload = _gesture_payload("UP", label=2)

        with client.websocket_connect("/ws/gestures") as ws1:
            with client.websocket_connect("/ws/gestures") as ws2:
                client.post("/gesture", json=payload)

                msg1 = json.loads(ws1.receive_text())
                msg2 = json.loads(ws2.receive_text())

        assert msg1["gesture"] == "UP"
        assert msg2["gesture"] == "UP"
        assert msg1 == msg2

    def test_disconnected_client_not_in_active_connections(self, client: TestClient):
        """After disconnect the manager's list shrinks back to zero."""
        from src.api.server import connection_manager

        with client.websocket_connect("/ws/gestures"):
            assert len(connection_manager.active_connections) == 1

        assert len(connection_manager.active_connections) == 0

    def test_broadcast_json_schema(self, client: TestClient):
        """Broadcast message contains exactly the three GestureEvent fields."""
        payload = _gesture_payload("DOWN", label=3, timestamp=9999.5)

        with client.websocket_connect("/ws/gestures") as ws:
            client.post("/gesture", json=payload)
            data = json.loads(ws.receive_text())

        assert set(data.keys()) == {"gesture", "confidence", "label", "timestamp"}

    def test_sequential_broadcasts(self, client: TestClient):
        """Multiple POSTs produce multiple WS messages in order."""
        gestures = [
            _gesture_payload("STOP",    label=0, timestamp=1.0),
            _gesture_payload("PLAY",    label=1, timestamp=2.0),
            _gesture_payload("CONFIRM", label=4, timestamp=3.0),
        ]

        with client.websocket_connect("/ws/gestures") as ws:
            for p in gestures:
                client.post("/gesture", json=p)

            received = [json.loads(ws.receive_text()) for _ in gestures]

        labels = [r["label"] for r in received]
        assert labels == [0, 1, 4]


# ---------------------------------------------------------------------------
# GestureToken enum validation
# ---------------------------------------------------------------------------

class TestGestureTokenValidation:

    def test_invalid_token_returns_422(self, client: TestClient):
        """Unknown gesture string must be rejected by FastAPI validation."""
        payload = _gesture_payload("MACHANIEREKA")
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_lowercase_token_returns_422(self, client: TestClient):
        """Tokens are case-sensitive — 'stop' is not a valid value."""
        payload = _gesture_payload("stop")
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_empty_string_returns_422(self, client: TestClient):
        payload = _gesture_payload("")
        response = client.post("/gesture", json=payload)
        assert response.status_code == 422

    def test_all_enum_members_are_accepted(self, client: TestClient):
        """Every member of GestureToken must pass validation."""
        for token in GestureToken:
            resp = client.post("/gesture", json=_gesture_payload(token.value))
            assert resp.status_code == 200, f"Token {token.value!r} rejected"

    def test_gesture_serialised_as_string_in_broadcast(self, client: TestClient):
        """Broadcast JSON must contain gesture as a plain string, not an object."""
        with client.websocket_connect("/ws/gestures") as ws:
            client.post("/gesture", json=_gesture_payload("STOP"))
            data = json.loads(ws.receive_text())

        assert isinstance(data["gesture"], str)
        assert data["gesture"] == "STOP"


# ---------------------------------------------------------------------------
# GET /tokens
# ---------------------------------------------------------------------------

class TestGetTokens:

    def test_returns_200(self, client: TestClient):
        assert client.get("/tokens").status_code == 200

    def test_contains_tokens_key(self, client: TestClient):
        data = client.get("/tokens").json()
        assert "tokens" in data

    def test_all_eight_tokens_present(self, client: TestClient):
        tokens = client.get("/tokens").json()["tokens"]
        expected = {t.value for t in GestureToken}
        assert set(tokens) == expected

    def test_tokens_are_strings(self, client: TestClient):
        tokens = client.get("/tokens").json()["tokens"]
        assert all(isinstance(t, str) for t in tokens)

    def test_tokens_match_enum_exactly(self, client: TestClient):
        """List from API must stay in sync with GestureToken definition."""
        from_api  = set(client.get("/tokens").json()["tokens"])
        from_enum = {t.value for t in GestureToken}
        assert from_api == from_enum


class TestStatus:
    def test_status_returns_200(self, client: TestClient):
        r = client.get("/status")
        assert r.status_code == 200

    def test_status_schema(self, client: TestClient):
        r = client.get("/status")
        data = r.json()
        assert set(data.keys()) == {"adapters", "ws_connections", "model_loaded"}
        assert isinstance(data["adapters"], list)
        assert isinstance(data["ws_connections"], int)
        assert isinstance(data["model_loaded"], bool)

    def test_status_adapter_list_is_dynamic(self, client: TestClient):
        # No adapters registered in the test-scoped EventBus — list should be empty
        r = client.get("/status")
        assert r.json()["adapters"] == []


# ---------------------------------------------------------------------------
# Training API  /api/train/*
# ---------------------------------------------------------------------------

class TestTrainingAPI:

    def test_status_returns_200(self, client: TestClient):
        assert client.get("/api/train/status").status_code == 200

    def test_status_schema(self, client: TestClient):
        data = client.get("/api/train/status").json()
        assert {"status", "progress", "result", "last_trained", "error"} <= data.keys()

    def test_initial_status_is_idle(self, client: TestClient):
        """State starts as idle when the server first boots."""
        import src.api.routes.training as t
        # Ensure clean state (other tests may have run first)
        with t._lock:
            t._state.update(status="idle", progress="", result=None, last_trained=None, error=None)
        assert client.get("/api/train/status").json()["status"] == "idle"

    def test_status_result_is_null_when_idle(self, client: TestClient):
        import src.api.routes.training as t
        with t._lock:
            t._state.update(status="idle", result=None)
        data = client.get("/api/train/status").json()
        assert data["result"] is None

    def test_history_returns_200(self, client: TestClient):
        assert client.get("/api/train/history").status_code == 200

    def test_history_schema(self, client: TestClient):
        data = client.get("/api/train/history").json()
        assert "history" in data
        assert isinstance(data["history"], list)

    def test_history_empty_initially(self, client: TestClient):
        import src.api.routes.training as t
        with t._lock:
            t._history.clear()
        data = client.get("/api/train/history").json()
        assert data["history"] == []

    def test_start_missing_csv_returns_400(self, client: TestClient, tmp_path, monkeypatch):
        """POST /start returns 400 when no CSV exists."""
        import src.api.routes.training as t
        monkeypatch.setattr(t, "_CSV_PATH", tmp_path / "nonexistent.csv")
        r = client.post("/api/train/start")
        assert r.status_code == 400
        assert "detail" in r.json()

    def test_start_already_running_returns_already_running(
        self, client: TestClient, monkeypatch
    ):
        """POST /start returns already_running without spawning a thread."""
        import src.api.routes.training as t
        monkeypatch.setitem(t._state, "status", "running")
        r = client.post("/api/train/start")
        assert r.status_code == 200
        assert r.json()["status"] == "already_running"
        # Restore
        monkeypatch.setitem(t._state, "status", "idle")
