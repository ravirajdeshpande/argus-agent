from argus.mcp.base import ToolClient


class CompanyClient(ToolClient):
    def company_profile(self, company: str):
        return self._call("company_profile", company=company)

    def supplier_lookup(self, company: str):
        return self._call("supplier_lookup", company=company)

    def get_company_events(self, company: str):
        return self._call("get_company_events", company=company)