from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from argus.graph.checkpoints import make_checkpointer, thread_config


class Counter(TypedDict):
    count: int


def build(checkpointer):
    g = StateGraph(Counter)
    g.add_node("inc", lambda s: {"count": s["count"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    return g.compile(checkpointer=checkpointer)


def test_thread_config_shape():
    assert thread_config("case-1") == {"configurable": {"thread_id": "case-1"}}


def test_no_path_gives_in_memory_saver():
    assert isinstance(make_checkpointer(), MemorySaver)


def test_state_is_saved_per_thread():
    graph = build(make_checkpointer())
    graph.invoke({"count": 0}, thread_config("a"))
    assert graph.get_state(thread_config("a")).values == {"count": 1}
    assert graph.get_state(thread_config("b")).values == {}


def test_sqlite_checkpoint_survives_a_new_checkpointer(tmp_path):
    db = tmp_path / "cp" / "checkpoints.db"
    build(make_checkpointer(db)).invoke({"count": 41}, thread_config("case-1"))
    assert db.exists()
    reopened = build(make_checkpointer(db))
    assert reopened.get_state(thread_config("case-1")).values == {"count": 42}