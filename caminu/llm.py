"""llama-server (OpenAI-compatible) client with tool-calling loop."""
from __future__ import annotations
import json
from typing import Any

import requests

from .config import LLAMA_MAX_NEW_TOKENS, LLAMA_MAX_TOOL_HOPS, LLAMA_URL, SYSTEM_PROMPT
from .log import log
from .tools import TOOL_SCHEMAS, TOOLS


def _call(messages: list[dict]) -> dict:
    payload: dict[str, Any] = {
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "max_tokens": LLAMA_MAX_NEW_TOKENS,
        "stream": False,
        "temperature": 0.7,
    }
    r = requests.post(f"{LLAMA_URL}/v1/chat/completions", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def chat_turn(user_text: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Run one user turn, including any tool hops. Returns (reply_text, new_history)."""
    if not history:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    else:
        messages = list(history)
    messages.append({"role": "user", "content": user_text})

    for hop in range(LLAMA_MAX_TOOL_HOPS):
        msg = _call(messages)
        messages.append(msg)
        calls = msg.get("tool_calls") or []

        if not calls:
            reply = (msg.get("content") or "").strip()
            return reply, messages

        for call in calls:
            name = call["function"]["name"]
            args_raw = call["function"].get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except json.JSONDecodeError:
                args = {}
            log(f"llm: hop {hop} tool_call {name}({args})")

            fn = TOOLS.get(name)
            if fn is None:
                result = {"text": f"Unknown tool: {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"text": f"Tool {name} failed: {e}"}

            if result.get("image_b64"):
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": "[photo captured]",
                })
                data_url = f"data:image/jpeg;base64,{result['image_b64']}"
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": "Here is the photo I just took. "
                                 "Answer my previous question using it."},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                })
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result.get("text", ""),
                })

    return "Sorry, I got stuck in my own thoughts.", messages


def wait_for_server(timeout_s: float = 60.0) -> bool:
    """Poll /health until ready or timeout."""
    import time
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(f"{LLAMA_URL}/health", timeout=2)
            if r.ok:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False
