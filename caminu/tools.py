"""Tool registry for Gemma 4 tool calling."""
from datetime import datetime

from . import camera
from .log import log


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "take_picture",
            "description": (
                "Capture a single photo from the on-device camera. "
                "Use this when the user asks about anything visual — "
                "what is visible, what they are wearing, holding, doing, "
                "or asks to describe the surroundings."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Return the current local time as an ISO-8601 string.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


def take_picture() -> dict:
    log("tool: take_picture()")
    b64 = camera.grab_jpeg_b64()
    if b64 is None:
        return {"text": "Camera unavailable — could not capture a photo."}
    log(f"tool: take_picture -> {len(b64)} b64 chars")
    return {"image_b64": b64}


def get_time() -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    log(f"tool: get_time() -> {now}")
    return {"text": now}


TOOLS = {
    "take_picture": take_picture,
    "get_time": get_time,
}
