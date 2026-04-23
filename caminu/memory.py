"""Persistent memory for C1.

Two layers:

1. **Facts (hot memory).** A plain markdown file at `memory/facts.md`, one
   fact per line. Gemma writes to it via the `remember(fact)` tool when
   she learns something important about the user or the environment.
   The entire (small) file is injected into the system prompt each turn.

2. **Conversation log (cold memory, retrievable).** Every turn appended as
   JSON lines to `memory/conversations.jsonl`. A lightweight embedder
   (sentence-transformers all-MiniLM-L6-v2, ~90 MB CPU) indexes each entry.
   The `recall(query)` tool embeds the query and returns the top-k most
   similar past turns.

Both files are plain text — the user can edit or wipe them directly.
"""
from __future__ import annotations
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .config import (
    MEMORY_DIR,
    MEMORY_FACTS_FILENAME,
    MEMORY_CONVERSATIONS_FILENAME,
    MEMORY_MAX_FACTS,
    MEMORY_RECALL_K,
    MEMORY_EMBEDDER_MODEL,
)
from .log import log


_embedder = None
_embedder_lock = threading.Lock()
_index_cache: list[tuple[dict, np.ndarray]] = []
_index_cache_mtime: float = 0.0


def _facts_path() -> Path:
    return MEMORY_DIR / MEMORY_FACTS_FILENAME


def _conv_path() -> Path:
    return MEMORY_DIR / MEMORY_CONVERSATIONS_FILENAME


def ensure_dirs() -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    _facts_path().touch(exist_ok=True)
    _conv_path().touch(exist_ok=True)


# ---------------- Facts ----------------

def load_facts() -> list[str]:
    """Return the list of stored facts (oldest first)."""
    p = _facts_path()
    if not p.exists():
        return []
    lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def facts_for_prompt() -> str:
    """Render facts as a short markdown block for the system prompt.
    Empty string if no facts yet."""
    facts = load_facts()
    if not facts:
        return ""
    # keep the most recent MEMORY_MAX_FACTS
    facts = facts[-MEMORY_MAX_FACTS:]
    body = "\n".join(f"- {f}" for f in facts)
    return f"\n\n# What you remember\n{body}\n"


def remember_fact(fact: str) -> str:
    """Append a fact to facts.md. Idempotent — skips near-duplicates."""
    fact = fact.strip().rstrip(".").strip()
    if not fact:
        return "nothing to remember"
    if len(fact) > 200:
        fact = fact[:200].rsplit(" ", 1)[0] + "..."

    existing = load_facts()
    for e in existing:
        if e.lower() == fact.lower():
            return "already remembered"

    ensure_dirs()
    with open(_facts_path(), "a", encoding="utf-8") as f:
        f.write(fact + "\n")
    log(f"memory: remembered {fact!r}")
    return f"remembered: {fact}"


# ---------------- Conversation log ----------------

def log_turn(user_text: str, assistant_text: str) -> None:
    """Append a turn to conversations.jsonl. Non-fatal on I/O errors."""
    ensure_dirs()
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "user": user_text,
        "assistant": assistant_text,
    }
    try:
        with open(_conv_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log(f"memory: log_turn failed: {e}")


def _load_conversations() -> list[dict]:
    p = _conv_path()
    if not p.exists():
        return []
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------- Embedder + retrieval ----------------

def _get_embedder():
    """Lazy-load the sentence-transformers model on CPU."""
    global _embedder
    if _embedder is not None:
        return _embedder
    with _embedder_lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            log(f"memory: loading embedder {MEMORY_EMBEDDER_MODEL}")
            _embedder = SentenceTransformer(MEMORY_EMBEDDER_MODEL, device="cpu")
    return _embedder


def _rebuild_index_if_stale() -> None:
    """Re-read conversations.jsonl and re-embed if it changed on disk."""
    global _index_cache, _index_cache_mtime
    p = _conv_path()
    if not p.exists():
        _index_cache = []
        _index_cache_mtime = 0.0
        return
    mtime = p.stat().st_mtime
    if mtime == _index_cache_mtime and _index_cache:
        return
    entries = _load_conversations()
    if not entries:
        _index_cache = []
        _index_cache_mtime = mtime
        return
    model = _get_embedder()
    docs = [f"{e.get('user','')}\n{e.get('assistant','')}" for e in entries]
    vecs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    _index_cache = list(zip(entries, list(vecs)))
    _index_cache_mtime = mtime
    log(f"memory: indexed {len(entries)} turns")


def recall(query: str, k: int = MEMORY_RECALL_K) -> list[dict]:
    """Return top-k past turns most similar to `query`."""
    _rebuild_index_if_stale()
    if not _index_cache:
        return []
    model = _get_embedder()
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scored = [
        (float(np.dot(q, vec)), entry)
        for entry, vec in _index_cache
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _score, e in scored[:k]]


def preload() -> None:
    """Ensure memory dirs exist. Does NOT load the embedder — that's lazy.

    On an 8 GB Jetson, sentence-transformers + torch pulls in ~1.5 GB of
    shared libraries. If we preload it, the kernel starts swapping under
    memory pressure when Gemma does a big context ingest, causing 10+ s
    STT stalls. recall() loads the embedder on demand; subsequent
    recall()s reuse the cached instance.
    """
    ensure_dirs()
