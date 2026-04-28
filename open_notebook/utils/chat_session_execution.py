"""Per-session asyncio locks LangGraph notebook/source chat mutation paths.

Notebook chat and source chat share the same ``chat_session:<id>`` thread_id space;
one lock per thread_id ensures get_state → append → invoke checkpoint runs are serialized
while different sessions keep independent locks so their graph work can overlap (when
combined with invoke running in asyncio.to_thread in routers).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from langchain_core.runnables import RunnableConfig

_lock_registry: Dict[str, asyncio.Lock] = {}
_registry_guard = asyncio.Lock()


async def get_session_graph_lock(thread_id: str) -> asyncio.Lock:
    """Stable ``asyncio.Lock`` for LangGraph ``thread_id`` (full chat session id).

    Same thread always maps to one lock object; different sessions use different keys.
    Locks are retained for process lifetime after first use so we never recreate a Lock
    for an id still held by an in-flight mutation.
    """
    async with _registry_guard:
        if thread_id not in _lock_registry:
            _lock_registry[thread_id] = asyncio.Lock()
        return _lock_registry[thread_id]


def invoke_graph_sync(graph: Any, state_values: dict, configurable: dict) -> Any:
    """Run compiled LangGraph synchronously — use via ``await asyncio.to_thread(...)`` from async routes."""
    return graph.invoke(
        input=state_values,
        config=RunnableConfig(configurable=configurable),
    )
