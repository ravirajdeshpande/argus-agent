import logging
from typing import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from argus.graph.checkpoints import make_checkpointer, thread_config
from argus.guardrails.evidence import Claim


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
    
CLAIM = Claim(
    claim="Controls take effect January 15, 2027", claim_type="effective_date", subject="controls",
    value="2027-01-15", sources=["gov_1"], primary_source=True, source_authority=0.95,
    recency_score=0.9, corroboration_score=0.5, contradiction=False, confidence=0.8,
)


class ClaimState(TypedDict):
    claims: list


def build_claims(checkpointer):
    g = StateGraph(ClaimState)
    g.add_node("make", lambda s: {"claims": [CLAIM]})
    g.add_edge(START, "make")
    g.add_edge("make", END)
    return g.compile(checkpointer=checkpointer)


def test_claims_round_trip_through_sqlite_without_a_warning(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    graph = build_claims(make_checkpointer(tmp_path / "cp.db"))
    graph.invoke({"claims": []}, thread_config("c"))
    assert graph.get_state(thread_config("c")).values["claims"] == [CLAIM]
    assert "unregistered type" not in caplog.text


def test_claims_round_trip_in_memory():
    graph = build_claims(make_checkpointer())
    graph.invoke({"claims": []}, thread_config("c"))
    assert graph.get_state(thread_config("c")).values["claims"] == [CLAIM]