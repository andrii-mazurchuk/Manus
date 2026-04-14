import json
from typing import List
from fastapi import WebSocket
from src.api.models.gesture import GestureEvent


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: GestureEvent):
        """Broadcast a Pydantic GestureEvent to all connected clients."""
        for connection in self.active_connections:
            await connection.send_text(message.model_dump_json())

    async def broadcast_json(self, payload: dict):
        """Broadcast an arbitrary JSON-serialisable dict to all connected clients."""
        text = json.dumps(payload)
        for connection in self.active_connections:
            await connection.send_text(text)
