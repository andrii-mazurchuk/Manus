from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from src.api.models.gesture import GestureEvent
from src.api.connectionMenager import ConnectionManager
from src.core.gesture_event import GestureToken

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

connection_manager = ConnectionManager()

@app.post("/gesture")
async def gesture(event: GestureEvent):
    await connection_manager.broadcast(event)
    return {"status": "ok"}


@app.websocket("/ws/gestures")
async def gestures(websocket: WebSocket):
    await connection_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)


@app.get("/status")
async def get_status():
    return {
        "adapters": ["WebSocketAdapter"],
        "ws_connections": len(connection_manager.active_connections),
        "model_loaded": False,
    }


@app.get("/tokens")
async def get_tokens():
    """Returns the canonical list of gesture tokens from the single source of truth."""
    return {"tokens": [t.value for t in GestureToken]}