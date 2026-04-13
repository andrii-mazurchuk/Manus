import json
import os

from dotenv import load_dotenv
import paho.mqtt.client as mqtt

from src.core.base_adapter import BaseAdapter
from src.core.gesture_event import GestureEvent

load_dotenv()

_HOST      = os.getenv("MQTT_BROKER_HOST", "localhost")
_PORT      = int(os.getenv("MQTT_BROKER_PORT", "1883"))
_TOPIC     = os.getenv("MQTT_TOPIC", "manus/gestures")
_THRESHOLD = float(os.getenv("MQTT_CONFIDENCE_THRESHOLD", "0.70"))


class MQTTAdapter(BaseAdapter):
    def __init__(
        self,
        host: str   = _HOST,
        port: int   = _PORT,
        topic: str  = _TOPIC,
        threshold: float = _THRESHOLD,
    ) -> None:
        self.topic     = topic
        self.threshold = threshold
        self._connected = False

        self._client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        try:
            self._client.connect(host, port)
            self._client.loop_start()
            self._connected = True
        except Exception as exc:
            print(f"[MQTTAdapter] could not connect to broker at {host}:{port} — {exc}")
            print("[MQTTAdapter] adapter loaded but publishing is disabled until broker is reachable.")

    def on_gesture(self, event: GestureEvent) -> None:
        if event.confidence < self.threshold:
            return
        if not self._connected:
            return
        try:
            self._client.publish(self.topic, json.dumps(event.to_dict()), qos=0)
        except Exception as exc:
            print(f"[MQTTAdapter] publish failed: {exc}")
