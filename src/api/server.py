from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from src.api.models.gesture import GestureEvent
from src.api.connection_manager import ConnectionManager
from src.core.gesture_event import GestureToken
from src.core.event_bus import EventBus
from src.api.routes import dataset, training, config as config_router, sequences

app = FastAPI(title="Manus API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dataset.router)
app.include_router(training.router)
app.include_router(config_router.router)
app.include_router(sequences.router)

connection_manager = ConnectionManager()


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/dashboard.html")

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
        "adapters": EventBus.get().list_adapters(),
        "ws_connections": len(connection_manager.active_connections),
        "model_loaded": False,
    }


@app.get("/tokens")
async def get_tokens():
    """Returns the canonical list of gesture tokens from the single source of truth."""
    return {"tokens": [t.value for t in GestureToken]}


app.mount("/static", StaticFiles(directory="src/frontend"), name="static")