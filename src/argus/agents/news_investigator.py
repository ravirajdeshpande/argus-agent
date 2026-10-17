import json

from argus.guardrails.budgets import ControlBudget
from argus.guardrails.injection import sanitize_retrieved
from argus.mcp.base import ToolError
from argus.models.cascade import ModelCaller, run_cascade

# action -> (client key, allowed args, required args)
ACTIONS = {
    "search_news": ("news", ("query", "date_range", "region"), ("query",)),
    "get_article": ("news", ("article_id",), ("article_id",)),
    "search_company_news": ("news", ("company",), ("company",)),
    "search_regulation": ("government", ("query", "jurisdiction"), ("query", "jurisdiction")),
    "get_notice": ("government", ("document_id",), ("document_id",)),
    "search_trade_controls": ("government", ("product", "country"), ("product", "country")),
    "get_stock_price": ("market", ("symbol",), ("symbol",)),
    "get_fx_rate": ("market", ("pair",), ("pair",)),
    "get_commodity_price": ("market", ("symbol",), ("symbol",)),
    "get_market_news": ("market", ("symbol",), ("symbol",)),
}

# client key -> (state field, source id prefix)
FIELDS = {
    "news": ("news_articles", "news"),
    "government": ("government_sources", "gov"),
    "market": ("market_signals", "market"),
}

STEP_PROMPT = (
    "You are the external news investigator of a risk-intelligence agent. "
    "Decide the single best next research action. "
    'Put a JSON object in the "answer" field with keys: thought (string), '
    "action (string), args (object). "
    'Use action "finish" when the evidence already answers the objective. '
    "Available actions and their arguments: {actions}. "
    "Objective: {objective}. Entities: {entities}. "
    "Evidence so far: {news} news items, {gov} government sources, {market} market signals. "
    "Missing information: {missing}."
)


def _actions_text() -> str:
    return "; ".join(f"{name}({', '.join(spec[1])})" for name, spec in ACTIONS.items())


def parse_action(text: str) -> dict | None:
    """Return a validated step, or None if the reply is unusable."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    action = data.get("action")
    thought = str(data.get("thought", ""))
    if action == "finish":
        return {"action": "finish", "args": {}, "thought": thought}
    if action not in ACTIONS:
        return None
    _, allowed, required = ACTIONS[action]
    args = data.get("args")
    args = {k: v for k, v in args.items() if k in allowed} if isinstance(args, dict) else {}
    if any(args.get(name) in (None, "") for name in required):
        return None
    return {"action": action, "args": args, "thought": thought}


def _sanitize(value):
    """Neutralise injection phrases anywhere in a tool result."""
    if isinstance(value, str):
        cleaned = sanitize_retrieved(value)
        return cleaned.text, cleaned.flagged
    if isinstance(value, list):
        pairs = [_sanitize(v) for v in value]
        return [p[0] for p in pairs], any(p[1] for p in pairs)
    if isinstance(value, dict):
        pairs = {k: _sanitize(v) for k, v in value.items()}
        return {k: p[0] for k, p in pairs.items()}, any(p[1] for p in pairs.values())
    return value, False


def make_news_investigator_node(call: ModelCaller, budget: ControlBudget, tools: dict):
    """One ReAct step per run; the graph loops via the evidence gate.

    `tools` maps "news", "government" and "market" to their MCP clients.
    """

    def investigator(state: dict) -> dict:
        prompt = STEP_PROMPT.format(
            actions=_actions_text(),
            objective=state["investigation_objective"] or state["user_query"],
            entities=", ".join(state["entities"]) or "none",
            news=len(state["news_articles"]),
            gov=len(state["government_sources"]),
            market=len(state["market_signals"]),
            missing="; ".join(state["missing_information"]) or "none",
        )
        result = run_cascade(prompt, call, budget)
        step = parse_action(result.text)
        update = {"investigation_iterations": state["investigation_iterations"] + 1}
        if step is None or step["action"] == "finish":
            return update

        action, args = step["action"], step["args"]
        client_key = ACTIONS[action][0]
        field, prefix = FIELDS[client_key]
        existing = state[field]
        if any(e.get("tool") == action and e.get("args") == args for e in existing):
            return update

        try:
            raw = getattr(tools[client_key], action)(**args)
        except ToolError as exc:
            update["missing_information"] = [*state["missing_information"], f"{action} failed: {exc}"]
            return update

        cleaned, flagged = _sanitize(raw)
        update[field] = [
            *existing,
            {
                "source_id": f"{prefix}_{len(existing) + 1}",
                "tool": action,
                "args": args,
                "thought": step["thought"],
                "result": cleaned,
                "injection_flagged": flagged,
            },
        ]
        return update

    return investigator