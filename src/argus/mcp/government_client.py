from argus.mcp.base import ToolClient


class GovernmentClient(ToolClient):
    def search_regulation(self, query: str, jurisdiction: str):
        return self._call("search_regulation", query=query, jurisdiction=jurisdiction)

    def get_notice(self, document_id: str):
        return self._call("get_notice", document_id=document_id)

    def search_trade_controls(self, product: str, country: str):
        return self._call("search_trade_controls", product=product, country=country)