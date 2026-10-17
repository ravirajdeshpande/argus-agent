import pytest

from argus.mcp.company_client import CompanyClient
from argus.mcp.government_client import GovernmentClient
from argus.mcp.market_client import MarketClient
from argus.mcp.news_client import NewsClient


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, tool, args):
        self.calls.append((tool, args))
        return {"ok": True}


CASES = [
    (NewsClient, "search_news", dict(query="chips", date_range="7d", region="EU")),
    (NewsClient, "get_article", dict(article_id="a1")),
    (NewsClient, "search_company_news", dict(company="Acme")),
    (GovernmentClient, "search_regulation", dict(query="export", jurisdiction="EU")),
    (GovernmentClient, "get_notice", dict(document_id="n1")),
    (GovernmentClient, "search_trade_controls", dict(product="chips", country="CN")),
    (MarketClient, "get_stock_price", dict(symbol="ASML")),
    (MarketClient, "get_fx_rate", dict(pair="EURUSD")),
    (MarketClient, "get_commodity_price", dict(symbol="CL")),
    (MarketClient, "get_market_news", dict(symbol="ASML")),
    (CompanyClient, "company_profile", dict(company="Acme")),
    (CompanyClient, "supplier_lookup", dict(company="Acme")),
    (CompanyClient, "get_company_events", dict(company="Acme")),
]


@pytest.mark.parametrize("cls, tool, kwargs", CASES)
def test_method_calls_matching_tool_with_args(cls, tool, kwargs):
    rec = Recorder()
    result = getattr(cls(rec), tool)(**kwargs)
    assert rec.calls == [(tool, kwargs)]
    assert result == {"ok": True}


def test_search_news_optional_args_default_to_none():
    rec = Recorder()
    NewsClient(rec).search_news("chips")
    assert rec.calls == [
        ("search_news", {"query": "chips", "date_range": None, "region": None})
    ]