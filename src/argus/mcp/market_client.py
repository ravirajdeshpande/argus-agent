from argus.mcp.base import ToolClient


class MarketClient(ToolClient):
    def get_stock_price(self, symbol: str):
        return self._call("get_stock_price", symbol=symbol)

    def get_fx_rate(self, pair: str):
        return self._call("get_fx_rate", pair=pair)

    def get_commodity_price(self, symbol: str):
        return self._call("get_commodity_price", symbol=symbol)

    def get_market_news(self, symbol: str):
        return self._call("get_market_news", symbol=symbol)