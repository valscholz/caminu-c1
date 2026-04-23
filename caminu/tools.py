"""Tool registry for Gemma 4 tool calling."""
from datetime import datetime

from . import camera, memory
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
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save a short, important fact about the user or this home/room "
                "that you'll want to recall in future conversations. Use this "
                "for things like the user's name, preferences, ongoing projects, "
                "or stable traits. Do NOT use it for one-off chit-chat."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to remember, as a single short sentence.",
                    },
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Search your previous conversations with this user for something "
                "relevant to a query. Use this when the user references past "
                "conversations ('remember when', 'what did I say about X', 'you "
                "told me...'). Returns up to 3 past turns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in past conversations.",
                    },
                },
                "required": ["query"],
            },
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


def remember(fact: str) -> dict:
    return {"text": memory.remember_fact(fact)}


def recall(query: str) -> dict:
    hits = memory.recall(query)
    if not hits:
        return {"text": "I couldn't find anything in our past conversations about that."}
    lines = []
    for h in hits:
        ts = h.get("ts", "")
        u = h.get("user", "").strip()
        a = h.get("assistant", "").strip()
        lines.append(f"- [{ts}] user: {u}  |  you: {a}")
    return {"text": "Past turns:\n" + "\n".join(lines)}


TOOLS = {
    "take_picture": take_picture,
    "get_time": get_time,
    "remember": remember,
    "recall": recall,
}
