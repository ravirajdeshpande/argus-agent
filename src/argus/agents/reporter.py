def _lines(lines: list[str]) -> str:
    return "\n".join(lines) if lines else "None."


def _claim_line(c) -> str:
    flags = f"confidence {c.confidence:.0%}" + (", primary source" if c.primary_source else "")
    return f"- {c.claim} [{', '.join(c.sources)}] ({flags})"


def _risk_section(state: dict) -> str:
    a = state["risk_assessment"]
    if a.get("status") == "assessed":
        lines = [f"- Risk level: {a['risk_level']} - {state['risk_score']:.1f} / 10"]
    else:
        lines = ["- Risk level: UNDETERMINED", "- Risk assessment could not be completed."]
    lines.append(f"- Confidence: {state['confidence_score']:.0%} (shown separately from risk)")
    lines.append(f"- Evidence quality: {state['evidence_quality_score']:.0%}")
    return "\n".join(lines)


def _affected_section(state: dict) -> str:
    suppliers = ", ".join(state["impacted_suppliers"]) or "none identified"
    products = ", ".join(state["impacted_products"]) or "none identified"
    return f"- Suppliers: {suppliers}\n- Products: {products}"


def _inference_section(state: dict) -> str:
    a = state["risk_assessment"]
    if a.get("status") != "assessed":
        return "None."
    lines = []
    if a.get("expected_impact"):
        lines.append(f"- Expected impact: {a['expected_impact']}")
    if a.get("rationale"):
        lines.append(f"- Rationale: {a['rationale']}")
    return _lines(lines)


def _uncertainty_section(state: dict) -> str:
    items = [
        *state["risk_assessment"].get("uncertainty", []),
        *state["evaluation"].get("issues", []),
        *state["evaluation"].get("repairs", []),
    ]
    return _lines([f"- {i}" for i in dict.fromkeys(items)])


def _conflict_line(c: dict) -> str:
    label = f"{c['subject']} ({c['claim_type']})"
    if c.get("resolved"):
        rejected = "; ".join(c.get("rejected_claims", []))
        return (
            f"- Resolved: {label} - primary source supports '{c['winning_value']}'"
            + (f"; rejected: {rejected}" if rejected else "")
        )
    return (
        f"- UNRESOLVED: {label} - conflicting values: {', '.join(c['values'])}; "
        f"sources: {', '.join(c['sources'])}"
    )


def _evidence_section(state: dict) -> str:
    return "\n".join(
        [
            f"- {len(state['news_articles'])} news items",
            f"- {len(state['government_sources'])} government sources",
            f"- {len(state['market_signals'])} market signals",
            f"- {len(state['internal_documents'])} internal documents",
        ]
    )


def _action_lines(state: dict) -> str:
    lines = []
    for a in state["recommended_actions"]:
        head = (
            f"{a['priority']}. {a['action']} - owner: {a['owner']}; urgency: {a['urgency']}; "
            f"category: {a['category']}"
        )
        if a.get("requires_approval"):
            head += "; REQUIRES HUMAN APPROVAL"
        if a.get("sources"):
            head += f" (sources: {', '.join(a['sources'])})"
        lines.append(head)
        if a.get("rationale"):
            lines.append(f"   Rationale: {a['rationale']}")
    return _lines(lines)


def _scenario_lines(state: dict) -> str:
    scenarios = state["risk_assessment"].get("scenarios", [])
    return _lines(
        [f"- {s['name']}: {s['description']} - {round(s['probability'] * 100)}%" for s in scenarios]
    )


def build_report(state: dict) -> str:
    """Decision-oriented report; facts, inference, uncertainty and actions kept apart."""
    verified = state["verified_claims"]
    unverified = [c for c in state["claims"] if c not in verified and not c.contradiction]
    sections = [
        ("Objective", state["investigation_objective"] or state["user_query"]),
        ("Risk and confidence", _risk_section(state)),
        ("Potentially affected", _affected_section(state)),
        ("Facts (verified, with sources)", _lines([_claim_line(c) for c in verified])),
        ("Reported but not independently verified", _lines([_claim_line(c) for c in unverified])),
        ("Inference", _inference_section(state)),
        ("Uncertainty and limitations", _uncertainty_section(state)),
        ("Conflicts", _lines([_conflict_line(c) for c in state["conflicting_claims"]])),
        ("Evidence reviewed", _evidence_section(state)),
        ("Recommended actions", _action_lines(state)),
        ("Scenarios", _scenario_lines(state)),
    ]
    body = "\n\n".join(f"## {title}\n{text}" for title, text in sections)
    return f"# GLOBAL RISK INTELLIGENCE REPORT\n\n{body}\n"


def make_reporter_node():
    def reporter(state: dict) -> dict:
        return {"final_report": build_report(state)}

    return reporter