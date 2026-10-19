import pytest

from argus.evaluation.datasets.demo_scenario import (
    DEMO_DOCUMENTS,
    DEMO_QUESTION,
    build_demo_store,
    demo_transport,
)
from argus.guardrails.access_control import Confidentiality
from argus.guardrails.injection import check_input
from argus.rag.retrieval import retrieve

VOCAB = ["supplier", "inventory", "contract", "expired", "acquisition"]


def fake_embed(texts):
    return [[float(t.lower().count(w)) for w in VOCAB] for t in texts]


def test_news_search_contains_the_conflicting_date_claims():
    articles = demo_transport("search_news", {"query": "x"})
    by_publisher = {a["publisher"]: a["summary"] for a in articles}
    assert "January 15, 2027" in by_publisher["Reuters"]
    assert "immediately" in by_publisher["Semiconductor Weekly"]


def test_supplier_announcement_only_for_the_matching_company():
    hits = demo_transport("search_company_news", {"company": "ABC Semiconductor"})
    assert "unaffected" in hits[0]["summary"]
    assert demo_transport("search_company_news", {"company": "Other Corp"}) == []


def test_government_notice_limits_scope_and_states_the_date():
    notice = demo_transport("search_regulation", {"query": "x", "jurisdiction": "EU"})[0]
    assert notice["publisher"] == "Ministry of Economic Affairs"
    assert "categories X, Y and Z" in notice["summary"]
    assert "January 15, 2027" in notice["summary"]


def test_article_and_notice_lookup():
    assert demo_transport("get_article", {"article_id": "news-001"})["publisher"] == "Reuters"
    assert demo_transport("get_article", {"article_id": "nope"}) == {"error": "not found"}
    notice = demo_transport("get_notice", {"document_id": "notice-2026-117"})
    assert notice["document_id"] == "notice-2026-117"


def test_market_tools_return_simple_data():
    assert "price" in demo_transport("get_stock_price", {"symbol": "ASML"})
    assert "rate" in demo_transport("get_fx_rate", {"pair": "EURUSD"})


def test_unknown_tool_raises():
    with pytest.raises(ValueError):
        demo_transport("launch_missiles", {})


def test_transport_returns_copies():
    first = demo_transport("search_news", {})
    first[0]["summary"] = "changed"
    assert demo_transport("search_news", {})[0]["summary"] != "changed"


def test_documents_have_required_metadata():
    for doc in DEMO_DOCUMENTS:
        assert doc["name"] and doc["text"]
        assert doc["metadata"]["confidentiality"] in {"public", "internal", "confidential", "restricted"}
        assert "doc_type" in doc["metadata"]


def test_documents_include_an_expired_contract_and_its_replacement():
    expired = [d for d in DEMO_DOCUMENTS if d["metadata"].get("expires", "9999") < "2026-10-17"]
    assert len(expired) == 1
    assert "replaced by C-104" in expired[0]["text"]
    current = [d for d in DEMO_DOCUMENTS if d["metadata"].get("expires", "") > "2026-10-17"]
    assert any("Contract C-104" in d["text"] for d in current)


def test_documents_include_inventory_figure_and_a_confidential_memo():
    assert any("45 days" in d["text"] for d in DEMO_DOCUMENTS)
    assert any(d["metadata"]["confidentiality"] == "confidential" for d in DEMO_DOCUMENTS)


def test_demo_store_holds_every_document():
    assert len(build_demo_store(fake_embed)) == len(DEMO_DOCUMENTS)


def test_access_control_hides_the_confidential_memo():
    store = build_demo_store(fake_embed)
    internal = retrieve(store, "acquisition", Confidentiality.INTERNAL)
    assert all("acquisition" not in r["text"].lower() for r in internal)
    confidential = retrieve(store, "acquisition", Confidentiality.CONFIDENTIAL)
    assert "acquisition" in confidential[0]["text"].lower()


def test_inventory_question_finds_the_procurement_report():
    results = retrieve(build_demo_store(fake_embed), "inventory coverage", Confidentiality.INTERNAL)
    assert "45 days" in results[0]["text"]


def test_demo_question_passes_the_input_guardrail():
    assert check_input(DEMO_QUESTION).allowed