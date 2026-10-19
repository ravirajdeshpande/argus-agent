from datetime import date

import pytest

from argus.rag.reranker import rerank

AS_OF = date(2026, 10, 17)


def hit(text, score, **meta):
    return {"text": text, "metadata": meta, "score": score}


def test_expired_chunk_is_downranked_below_valid_one():
    hits = [
        hit("old contract", 0.95, expires="2025-12-31"),
        hit("new contract", 0.60, expires="2027-12-31"),
    ]
    ranked = rerank(hits, AS_OF)
    assert [h["text"] for h in ranked] == ["new contract", "old contract"]


def test_expired_flag_is_set_in_metadata():
    ranked = rerank(
        [hit("a", 0.9, expires="2025-01-01"), hit("b", 0.8, expires="2030-01-01")], AS_OF
    )
    flags = {h["text"]: h["metadata"]["expired"] for h in ranked}
    assert flags == {"a": True, "b": False}


def test_expiry_on_the_as_of_day_is_still_valid():
    ranked = rerank([hit("a", 0.9, expires="2026-10-17")], AS_OF)
    assert ranked[0]["metadata"]["expired"] is False


def test_newer_document_beats_older_with_same_score():
    hits = [
        hit("older", 0.8, date="2020-01-01"),
        hit("newer", 0.8, date="2026-09-01"),
    ]
    assert [h["text"] for h in rerank(hits, AS_OF)] == ["newer", "older"]


def test_recency_math_one_half_life():
    ranked = rerank(
        [hit("a", 1.0, date="2025-10-17")],
        AS_OF,
        recency_weight=0.2,
        half_life_days=365,
    )
    assert ranked[0]["final_score"] == pytest.approx(0.9)


def test_undated_chunk_gets_no_recency_adjustment():
    ranked = rerank([hit("a", 0.7)], AS_OF)
    assert ranked[0]["final_score"] == pytest.approx(0.7)


def test_invalid_dates_are_ignored():
    ranked = rerank([hit("a", 0.7, date="not-a-date", expires="soon")], AS_OF)
    assert ranked[0]["final_score"] == pytest.approx(0.7)
    assert ranked[0]["metadata"]["expired"] is False


def test_future_dated_chunk_is_not_boosted_above_its_score():
    ranked = rerank([hit("a", 0.7, date="2030-01-01")], AS_OF)
    assert ranked[0]["final_score"] == pytest.approx(0.7)


def test_drop_expired_removes_them():
    hits = [hit("old", 0.9, expires="2020-01-01"), hit("ok", 0.5)]
    ranked = rerank(hits, AS_OF, drop_expired=True)
    assert [h["text"] for h in ranked] == ["ok"]


def test_original_score_kept_and_inputs_not_mutated():
    original = hit("a", 0.9, expires="2020-01-01")
    ranked = rerank([original], AS_OF)
    assert ranked[0]["score"] == 0.9
    assert "final_score" not in original
    assert "expired" not in original["metadata"]


def test_ties_keep_original_order():
    hits = [hit("first", 0.5), hit("second", 0.5)]
    assert [h["text"] for h in rerank(hits, AS_OF)] == ["first", "second"]