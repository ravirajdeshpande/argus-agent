from collections.abc import Callable
from datetime import date

from argus.guardrails.access_control import Confidentiality
from argus.rag.reranker import rerank
from argus.rag.retrieval import retrieve


def build_query(state: dict) -> str:
    parts = [
        state["investigation_objective"] or state["user_query"],
        *state["entities"],
        *state["companies"],
        *state["countries"],
    ]
    return " ".join(dict.fromkeys(p for p in parts if p))


def _key(item: dict) -> tuple:
    meta = item["metadata"]
    return (meta.get("source"), meta.get("chunk_index"), item["text"])


def make_internal_analyst_node(
    store,
    clearance: Confidentiality,
    *,
    k: int = 5,
    min_score: float = 0.2,
    filters: dict | None = None,
    today: Callable[[], date] = date.today,
):
    """Retrieve internal knowledge: access-controlled, reranked, scrubbed."""

    def analyst(state: dict) -> dict:
        def search(query: str) -> list[dict]:
            hits = retrieve(store, query, clearance, k=k, filters=filters)
            return [h for h in hits if h["score"] >= min_score]

        hits = search(build_query(state))
        if not hits and state["user_query"] != build_query(state):
            hits = search(state["user_query"])  # reformulate once (brief, Section 21)

        existing = state["internal_documents"]
        seen = {_key(d) for d in existing}
        docs = list(existing)
        for h in rerank(hits, today()):
            if _key(h) in seen:
                continue
            seen.add(_key(h))
            docs.append(
                {
                    "source_id": f"internal_{len(docs) + 1}",
                    "text": h["text"],
                    "score": h["score"],
                    "final_score": h["final_score"],
                    "metadata": h["metadata"],
                }
            )
        return {"internal_documents": docs}

    return analyst