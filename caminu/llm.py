"""llama-server (OpenAI-compatible) client with tool-calling loop.

Supports streaming: caller passes an `on_text` callback that receives
content tokens as they arrive. Tool calls are collected across the
stream and resolved as atomic events at end-of-stream (llama.cpp emits
tool_calls as a single finish_reason="tool_calls" event, not interleaved
with content tokens).
"""
from __future__ import annotations
import json
from typing import Any, Callable, Iterator, Optional

import requests

from . import memory
from .config import LLAMA_MAX_NEW_TOKENS, LLAMA_MAX_TOOL_HOPS, LLAMA_URL, SYSTEM_PROMPT
from .log import log
from .tools import TOOL_SCHEMAS, TOOLS


def _build_system_prompt() -> str:
    """System prompt + any persisted facts injected fresh each turn."""
    return SYSTEM_PROMPT + memory.facts_for_prompt()


OnTextFn = Callable[[str], None]


def _payload(messages: list[dict], stream: bool) -> dict[str, Any]:
    return {
        "messages": messages,
        "tools": TOOL_SCHEMAS,
        "max_tokens": LLAMA_MAX_NEW_TOKENS,
        "stream": stream,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _call_blocking(messages: list[dict]) -> dict:
    r = requests.post(
        f"{LLAMA_URL}/v1/chat/completions",
        json=_payload(messages, stream=False),
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


def _iter_sse(response: requests.Response) -> Iterator[dict]:
    """Yield parsed JSON events from an SSE stream."""
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _call_streaming(messages: list[dict], on_text: OnTextFn) -> dict:
    """Stream tokens from llama-server; fire on_text for every content delta.

    Returns a reconstructed assistant message compatible with the blocking
    path: {"role": "assistant", "content": str, "tool_calls": [...] }.
    """
    content_parts: list[str] = []
    tool_calls_by_idx: dict[int, dict] = {}

    with requests.post(
        f"{LLAMA_URL}/v1/chat/completions",
        json=_payload(messages, stream=True),
        stream=True,
        timeout=120,
    ) as r:
        r.raise_for_status()
        for event in _iter_sse(r):
            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}

            # Content tokens: fire the callback incrementally.
            if (chunk := delta.get("content")):
                content_parts.append(chunk)
                on_text(chunk)

            # Tool call fragments: accumulate by index (OpenAI streaming convention).
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls_by_idx.setdefault(
                    idx,
                    {"id": "", "type": "function",
                     "function": {"name": "", "arguments": ""}},
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]

    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
    }
    if tool_calls_by_idx:
        msg["tool_calls"] = [tool_calls_by_idx[k] for k in sorted(tool_calls_by_idx)]
    return msg


def _execute_tool_call(call: dict) -> dict:
    """Resolve a single tool_call dict; returns the tool result."""
    name = call["function"]["name"]
    args_raw = call["function"].get("arguments") or "{}"
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
    except json.JSONDecodeError:
        args = {}
    log(f"llm: tool_call {name}({args})")
    fn = TOOLS.get(name)
    if fn is None:
        return {"text": f"Unknown tool: {name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"text": f"Tool {name} failed: {e}"}


def _append_tool_result(messages: list[dict], call: dict, result: dict) -> None:
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


def chat_turn(
    user_text: str,
    history: list[dict],
    on_text: Optional[OnTextFn] = None,
) -> tuple[str, list[dict]]:
    """Run one user turn (incl. tool hops). If `on_text` is given, stream
    content tokens to it as they arrive.

    Returns (full_reply_text, new_history).
    """
    sys_content = _build_system_prompt()
    if not history:
        messages: list[dict] = [{"role": "system", "content": sys_content}]
    else:
        # refresh the system message each turn so freshly-remembered facts land
        messages = list(history)
        if messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": sys_content}
        else:
            messages.insert(0, {"role": "system", "content": sys_content})
    messages.append({"role": "user", "content": user_text})

    for hop in range(LLAMA_MAX_TOOL_HOPS):
        if on_text is None:
            msg = _call_blocking(messages)
        else:
            msg = _call_streaming(messages, on_text)
        messages.append(msg)
        calls = msg.get("tool_calls") or []

        if not calls:
            return (msg.get("content") or "").strip(), messages

        for call in calls:
            result = _execute_tool_call(call)
            _append_tool_result(messages, call, result)

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
