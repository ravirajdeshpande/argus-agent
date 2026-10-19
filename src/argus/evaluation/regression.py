import argparse
import json
import time
import uuid
from pathlib import Path

from argus.evaluation.datasets.cases import CASES
from argus.evaluation.metrics import (
    contradiction_handled,
    grounding_rate,
    retrieval_scores,
    run_stats,
    text_checks,
    unresolved_conflicts,
    unsupported_entities,
)
from argus.graph.graph import run_investigation
from argus.guardrails.injection import check_input


def _result(case: dict, failures: list[str], metrics: dict, stats: dict) -> dict:
    return {"id": case["id"], "passed": not failures, "failures": failures, "metrics": metrics, "stats": stats}


def evaluate_case(case: dict, graph_factory, new_budget, *, run_id: str = "run", clock=time.perf_counter) -> dict:
    expect = case["expect"]
    if not check_input(case["question"]).allowed:
        failures = [] if expect.get("rejected") else ["unexpected rejection"]
        return _result(case, failures, {}, {})
    if expect.get("rejected"):
        return _result(case, ["expected the question to be rejected"], {}, {})

    budget = new_budget()
    graph = graph_factory(budget=budget)
    start = clock()
    state = run_investigation(graph, case["question"], thread_id=f"eval-{run_id}-{case['id']}")
    stats = run_stats(budget, clock() - start)

    failures = text_checks(state["final_report"], expect.get("must_include", []), expect.get("must_exclude", []))
    grounding = grounding_rate(state)
    if grounding < expect.get("min_grounding", 0.0):
        failures.append(f"grounding {grounding} below {expect['min_grounding']}")
    unsupported = unsupported_entities(state)
    if unsupported:
        failures.append(f"unsupported entities: {', '.join(unsupported)}")

    retrieval = None
    if "relevant_sources" in expect:
        retrieval = retrieval_scores(state, set(expect["relevant_sources"]))
        if retrieval["recall"] < expect.get("min_recall", 0.0):
            failures.append(f"retrieval recall {retrieval['recall']} below {expect['min_recall']}")
        if retrieval["precision"] < expect.get("min_precision", 0.0):
            failures.append(f"retrieval precision {retrieval['precision']} below {expect['min_precision']}")

    if "contradiction" in expect:
        c = expect["contradiction"]
        if not contradiction_handled(state, c["claim_type"], c["winning_value"]):
            failures.append("contradiction not handled")

    limit, cost = expect.get("max_cost_usd"), stats["cost_usd"]
    if limit is not None and cost is not None and cost > limit:
        failures.append(f"cost {cost} above {limit}")

    metrics = {
        "grounding": grounding,
        "unsupported": unsupported,
        "retrieval": retrieval,
        "unresolved_conflicts": unresolved_conflicts(state),
        "conflicts_seen": len(state["conflicting_claims"]),
    }
    return _result(case, failures, metrics, stats)


def summarize(results: list[dict]) -> dict:
    answered = [r for r in results if r["metrics"]]
    n = len(results)
    passed = sum(1 for r in results if r["passed"])
    return {
        "total": n,
        "passed": passed,
        "pass_rate": round(passed / n, 2) if n else 0.0,
        "avg_grounding": round(sum(r["metrics"]["grounding"] for r in answered) / len(answered), 2) if answered else 1.0,
        "total_cost_usd": round(sum(r["stats"].get("cost_usd") or 0 for r in results), 4),
        "total_elapsed_s": round(sum(r["stats"].get("elapsed_s") or 0 for r in results), 2),
    }


def run_suite(cases: list[dict], graph_factory, new_budget) -> dict:
    run_id = uuid.uuid4().hex[:6]
    results = [evaluate_case(c, graph_factory, new_budget, run_id=run_id) for c in cases]
    return {"results": results, "summary": summarize(results)}


def compare(baseline: dict, current: dict, *, tolerance: float = 0.05, cost_tolerance: float = 0.5) -> list[str]:
    """Regressions of the current run against a saved baseline."""
    was_passing = {r["id"] for r in baseline["results"] if r["passed"]}
    issues = [f"case now failing: {r['id']}" for r in current["results"] if r["id"] in was_passing and not r["passed"]]
    b, c = baseline["summary"], current["summary"]
    if c["pass_rate"] < b["pass_rate"] - tolerance:
        issues.append(f"pass rate dropped {b['pass_rate']} -> {c['pass_rate']}")
    if c["avg_grounding"] < b["avg_grounding"] - tolerance:
        issues.append(f"grounding dropped {b['avg_grounding']} -> {c['avg_grounding']}")
    if b["total_cost_usd"] > 0 and c["total_cost_usd"] > b["total_cost_usd"] * (1 + cost_tolerance):
        issues.append(f"cost grew {b['total_cost_usd']} -> {c['total_cost_usd']}")
    return issues


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="argus-eval", description="ARGUS regression evaluation")
    p.add_argument("--baseline", default=None, help="baseline JSON path (default: data/evaluation/baseline.json)")
    p.add_argument("--save-baseline", action="store_true", help="save this run as the new baseline")
    p.add_argument("--only", nargs="*", default=None, help="run only these case ids")
    return p.parse_args(argv)


def main(argv=None, *, graph_factory=None, new_budget=None, cases=None) -> int:
    args = parse_args(argv)
    settings = None
    if graph_factory is None or new_budget is None or args.baseline is None:
        from argus.config.settings import get_settings

        settings = get_settings()
    if graph_factory is None:
        from argus.app import build_app

        graph_factory = lambda **kw: build_app(settings=settings, **kw)  # noqa: E731
    if new_budget is None:
        from argus.guardrails.budgets import ControlBudget

        new_budget = lambda: ControlBudget.from_settings(settings)  # noqa: E731
    path = Path(args.baseline) if args.baseline else Path(settings.data_dir) / "evaluation" / "baseline.json"

    chosen = [c for c in (cases or CASES) if not args.only or c["id"] in args.only]
    report = run_suite(chosen, graph_factory, new_budget)
    for r in report["results"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'} {r['id']}" + ("" if r["passed"] else f": {'; '.join(r['failures'])}"))
    print(json.dumps(report["summary"]))

    issues = []
    if args.save_baseline:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
        print(f"Baseline saved to {path}")
    elif path.exists():
        issues = compare(json.loads(path.read_text()), report)
        for issue in issues:
            print(f"REGRESSION: {issue}")
    else:
        print(f"No baseline at {path}; run with --save-baseline first.")

    failed = report["summary"]["passed"] < report["summary"]["total"]
    return 1 if failed or issues else 0


if __name__ == "__main__":
    raise SystemExit(main())