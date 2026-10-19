import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

# Types our state stores; allowed explicitly so LangGraph can restore them safely.
ALLOWED_TYPES = [("argus.guardrails.evidence", "Claim")]

def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)

def thread_config(thread_id: str) -> dict:
    """Config that tells LangGraph which investigation (thread) to save or resume."""
    return {"configurable": {"thread_id": thread_id}}


def make_checkpointer(db_path=None):
    """In-memory saver by default; SQLite file saver when a path is given."""
    if db_path is None:
        return MemorySaver().with_allowlist(ALLOWED_TYPES)
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(path, check_same_thread=False), serde=_serializer())