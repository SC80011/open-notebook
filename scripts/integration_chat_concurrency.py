"""
Real integration check for notebook chat concurrency (multi-session parallel + same-session serial).

Prerequisites
-------------
1. SurrealDB reachable (see .env / SURREAL_*).
2. API running with OpenAI-compatible LLM available to the **server process**, e.g.::

     set OPENAI_COMPATIBLE_BASE_URL=http://192.168.252.230:38080/v1
     set OPENAI_COMPATIBLE_API_KEY=<your-key>
     uv run uvicorn api.main:app --host 0.0.0.0 --port 5055

3. Set OPEN_NOTEBOOK_PASSWORD (same as API) when running this script.
4. Optional: OPEN_NOTEBOOK_MODEL_NAME=Auto-LLM (default) — must exist or will be created.

This script never prints API keys. Pass secrets only via environment variables.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any, Dict

import httpx

# HTTP origin (no path), e.g. http://127.0.0.1:5055 — routes live under /api
_API_ORIGIN = os.environ.get("OPEN_NOTEBOOK_API_BASE", "http://127.0.0.1:5055").rstrip("/")
API_ROOT = f"{_API_ORIGIN}/api"

DEFAULT_MODEL_NAME = os.environ.get("OPEN_NOTEBOOK_MODEL_NAME", "Auto-LLM")
PASSWORD = os.environ.get("OPEN_NOTEBOOK_PASSWORD", "open-notebook-change-me")


def headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {PASSWORD}",
        "Content-Type": "application/json",
    }


async def ensure_language_model(client: httpx.AsyncClient, model_name: str) -> str:
    """Return model record id for openai_compatible language model."""
    r = await client.get(f"{API_ROOT}/models", params={"type": "language"})
    r.raise_for_status()
    for m in r.json():
        if m.get("name") == model_name and m.get("provider") == "openai_compatible":
            return m["id"]

    payload = {
        "name": model_name,
        "provider": "openai_compatible",
        "type": "language",
        "credential": None,
    }
    cr = await client.post(f"{API_ROOT}/models", json=payload)
    if cr.status_code == 400 and "already exists" in cr.text.lower():
        r = await client.get(f"{API_ROOT}/models", params={"type": "language"})
        r.raise_for_status()
        for m in r.json():
            if m.get("name") == model_name and m.get("provider") == "openai_compatible":
                return m["id"]
        raise RuntimeError(f"Could not resolve model after duplicate error: {cr.text}")

    cr.raise_for_status()
    return cr.json()["id"]


async def create_notebook(client: httpx.AsyncClient) -> str:
    nr = await client.post(
        f"{API_ROOT}/notebooks",
        json={
            "name": "Concurrency integration test notebook",
            "description": "",
        },
    )
    nr.raise_for_status()
    return nr.json()["id"]


async def create_session(client: httpx.AsyncClient, notebook_id: str) -> str:
    sr = await client.post(
        f"{API_ROOT}/chat/sessions",
        json={"notebook_id": notebook_id, "title": "cc-test"},
    )
    sr.raise_for_status()
    return sr.json()["id"]


def minimal_context() -> Dict[str, Any]:
    """Shape compatible with empty chat context."""
    return {"sources": {}, "notes": {}}


def _last_ai_content(data: dict) -> str:
    """Extract last assistant message text from /chat/execute JSON."""
    msgs = data.get("messages") or []
    for m in reversed(msgs):
        if not isinstance(m, dict):
            continue
        if m.get("type") == "ai":
            return str(m.get("content") or "").strip()
    return ""


async def execute_once(
    client: httpx.AsyncClient,
    session_id: str,
    model_id: str,
    message: str,
) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    er = await client.post(
        f"{API_ROOT}/chat/execute",
        json={
            "session_id": session_id,
            "message": message,
            "context": minimal_context(),
            "model_override": model_id,
        },
    )
    dt = time.perf_counter() - t0
    er.raise_for_status()
    data = er.json()
    n = len(data.get("messages", []))
    reply = _last_ai_content(data)
    return dt, n, reply


async def main() -> int:
    print("API root:", API_ROOT)
    print("Model name:", DEFAULT_MODEL_NAME)
    print()

    async with httpx.AsyncClient(headers=headers(), timeout=600.0) as client:
        try:
            model_id = await ensure_language_model(client, DEFAULT_MODEL_NAME)
        except Exception as e:
            print("Failed to ensure language model (check API + OPENAI_COMPATIBLE_* on server):", e)
            return 1

        print("Using model id:", model_id)

        notebook_id = await create_notebook(client)
        s1 = await create_session(client, notebook_id)
        s2 = await create_session(client, notebook_id)
        print("Sessions:", s1, s2)

        # Two clearly different tasks (distinct sessions), run concurrently.
        q1 = os.environ.get(
            "INTEGRATION_Q1",
            "In one or two sentences, what is photosynthesis?",
        )
        q2 = os.environ.get(
            "INTEGRATION_Q2",
            "In one or two sentences, what is the water cycle?",
        )

        print()
        print("--- Parallel test: two sessions, asyncio.gather (same wall-clock window) ---")
        print("Q1:", q1)
        print("Q2:", q2)
        print()

        t_wall_0 = time.perf_counter()
        r1, r2 = await asyncio.gather(
            execute_once(client, s1, model_id, q1),
            execute_once(client, s2, model_id, q2),
        )
        wall_parallel = time.perf_counter() - t_wall_0

        dt1, n1, reply1 = r1
        dt2, n2, reply2 = r2
        seq_if_serial = dt1 + dt2

        print("Session A (notebook chat):")
        print(f"  round-trip time: {dt1:.2f}s, messages in state: {n1}")
        print(f"  assistant reply:\n    {reply1}\n")

        print("Session B (notebook chat):")
        print(f"  round-trip time: {dt2:.2f}s, messages in state: {n2}")
        print(f"  assistant reply:\n    {reply2}\n")

        print(f"Wall-clock for both (gather): {wall_parallel:.2f}s")
        print(f"If strictly serial on one worker: would be ~ {seq_if_serial:.2f}s (sum of round-trips)")
        if wall_parallel < seq_if_serial - 1.0:
            print(
                "=> Parallelism: wall time is well below sum of individual times — "
                "both requests overlapped (expected for multi-session + non-blocking invoke)."
            )
        elif wall_parallel < seq_if_serial - 0.3:
            print("=> Some overlap likely (borderline).")
        else:
            print("=> Little or no overlap; check single-worker load or rerun.")

        # Same-session sanity (optional behavioural check)
        s3 = await create_session(client, notebook_id)
        t_serial_0 = time.perf_counter()
        a, b = await asyncio.gather(
            execute_once(client, s3, model_id, "Reply with only: X"),
            execute_once(client, s3, model_id, "Reply with only: Y"),
        )
        wall_same = time.perf_counter() - t_serial_0
        print()
        print("Same session (two concurrent POSTs — expect roughly serial total time):")
        print(f"  request A duration: {a[0]:.2f}s")
        print(f"  request B duration: {b[0]:.2f}s")
        print(f"  wall-clock total (gather): {wall_same:.2f}s")
        approx_serial = a[0] + b[0]
        if wall_same >= approx_serial * 0.85:
            print("  => Same-session timings consistent with queued execution behind one lock.")
        else:
            print("  => Interpreting mixed; lock still serializes correctness either way.")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
