import copy

from argus.rag.faiss_store import Embedder, FaissStore
from argus.rag.ingest import build_chunks

DEMO_QUESTION = "Will the new semiconductor export restrictions affect our company?"

REUTERS = {
    "article_id": "news-001",
    "publisher": "Reuters",
    "date": "2026-10-14",
    "title": "New chip export restrictions announced",
    "summary": "Officials said the new export restrictions on advanced chip-making equipment "
    "will begin on January 15, 2027.",
}
INDUSTRY = {
    "article_id": "news-002",
    "publisher": "Semiconductor Weekly",
    "date": "2026-10-15",
    "title": "Export rules hit shipments",
    "summary": "Industry sources say the new export restrictions take effect immediately "
    "and some shipments are already on hold.",
}
SUPPLIER_ANNOUNCEMENT = {
    "article_id": "news-003",
    "publisher": "ABC Semiconductor",
    "date": "2026-10-16",
    "title": "Statement on export rules",
    "summary": "ABC Semiconductor states that deliveries to customers are unaffected "
    "by the new export restrictions.",
}
GOVERNMENT_NOTICE = {
    "document_id": "notice-2026-117",
    "publisher": "Ministry of Economic Affairs",
    "date": "2026-10-13",
    "title": "Export controls on advanced semiconductor manufacturing equipment",
    "summary": "Only equipment in technical categories X, Y and Z is controlled. "
    "The controls take effect January 15, 2027.",
}

ARTICLES = [REUTERS, INDUSTRY]
ALL_ARTICLES = [REUTERS, INDUSTRY, SUPPLIER_ANNOUNCEMENT]

DEMO_DOCUMENTS = [
    {
        "name": "contract_c104.txt",
        "text": "Contract C-104: ABC Semiconductor supplies category X components for Product Alpha. "
        "The contract runs until 30 June 2028. ABC Semiconductor is the sole approved supplier "
        "of category X components.",
        "metadata": {"doc_type": "contract", "supplier": "ABC Semiconductor",
                     "product": "Product Alpha", "region": "EU", "date": "2026-01-10",
                     "expires": "2028-06-30", "confidentiality": "internal"},
    },
    {
        "name": "contract_c088_expired.txt",
        "text": "Contract C-088 (superseded): ABC Semiconductor supplied category X components for "
        "Product Alpha under older terms. This contract expired on 31 December 2025 and was "
        "replaced by C-104.",
        "metadata": {"doc_type": "contract", "supplier": "ABC Semiconductor",
                     "product": "Product Alpha", "region": "EU", "date": "2022-01-10",
                     "expires": "2025-12-31", "confidentiality": "internal"},
    },
    {
        "name": "procurement_report_2026_10.txt",
        "text": "Procurement report October 2026: inventory coverage of category X components for "
        "Product Alpha is 45 days. Committed orders cover the next two months.",
        "metadata": {"doc_type": "report", "supplier": "ABC Semiconductor",
                     "product": "Product Alpha", "region": "EU", "date": "2026-10-01",
                     "confidentiality": "internal"},
    },
    {
        "name": "bom_product_alpha.txt",
        "text": "Bill of materials for Product Alpha: category X components come from ABC Semiconductor "
        "and category Z components come from XYZ Components.",
        "metadata": {"doc_type": "bom", "product": "Product Alpha", "region": "EU",
                     "date": "2026-03-01", "confidentiality": "internal"},
    },
    {
        "name": "board_memo_2027.txt",
        "text": "Confidential board memo: acquisition strategy and target list for 2027. "
        "Not related to supply risk.",
        "metadata": {"doc_type": "memo", "date": "2026-09-20", "confidentiality": "confidential"},
    },
]


def demo_transport(tool: str, args: dict):
    """In-memory stand-in for the MCP servers, seeded with conflicting evidence."""
    if tool == "search_news":
        return copy.deepcopy(ARTICLES)
    if tool == "search_company_news":
        if "abc" in str(args.get("company", "")).lower():
            return [copy.deepcopy(SUPPLIER_ANNOUNCEMENT)]
        return []
    if tool == "get_article":
        found = next((a for a in ALL_ARTICLES if a["article_id"] == args.get("article_id")), None)
        return copy.deepcopy(found) if found else {"error": "not found"}
    if tool in ("search_regulation", "search_trade_controls"):
        return [copy.deepcopy(GOVERNMENT_NOTICE)]
    if tool == "get_notice":
        if args.get("document_id") == GOVERNMENT_NOTICE["document_id"]:
            return copy.deepcopy(GOVERNMENT_NOTICE)
        return {"error": "not found"}
    if tool == "get_stock_price":
        return {"symbol": args.get("symbol"), "price": 812.4}
    if tool == "get_fx_rate":
        return {"pair": args.get("pair"), "rate": 1.08}
    if tool == "get_commodity_price":
        return {"symbol": args.get("symbol"), "price": 74.2}
    if tool == "get_market_news":
        return []
    raise ValueError(f"unknown demo tool: {tool}")


def build_demo_store(embedder: Embedder, *, chunk_size: int = 120, overlap: int = 20) -> FaissStore:
    store = FaissStore(embedder)
    for doc in DEMO_DOCUMENTS:
        store.add(
            build_chunks(doc["text"], {**doc["metadata"], "source": doc["name"]}, chunk_size, overlap)
        )
    return store