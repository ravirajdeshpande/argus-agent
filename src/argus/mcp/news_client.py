from argus.mcp.base import ToolClient


class NewsClient(ToolClient):
    def search_news(self, query: str, date_range: str | None = None, region: str | None = None):
        return self._call("search_news", query=query, date_range=date_range, region=region)

    def get_article(self, article_id: str):
        return self._call("get_article", article_id=article_id)

    def search_company_news(self, company: str):
        return self._call("search_company_news", company=company)