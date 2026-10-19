import argparse
import sys
from pathlib import Path

from argus.config.settings import Settings, get_settings
from argus.evaluation.datasets.demo_scenario import DEMO_QUESTION, build_demo_store, demo_transport
from argus.graph.checkpoints import make_checkpointer, thread_config
from argus.graph.graph import InputRejected, build_graph, resume_investigation, run_investigation
from argus.guardrails.access_control import Confidentiality
from argus.guardrails.budgets import ControlBudget
from argus.mcp.government_client import GovernmentClient
from argus.mcp.market_client import MarketClient
from argus.mcp.news_client import NewsClient
from argus.models.providers import build_client, make_caller
from argus.rag.faiss_store import Embedder, FaissStore, make_local_embedder

DEFAULT_CHUNK_SIZE = 120
DEFAULT_CHUNK_OVERLAP = 20


def load_store(settings: Settings, embedder: Embedder, *, use_demo: bool) -> FaissStore:
    """Demo documents if asked, else the saved FAISS index, else an empty store."""
    if use_demo:
        return build_demo_store(
            embedder,
            chunk_size=getattr(settings, "chunk_size", DEFAULT_CHUNK_SIZE),
            overlap=getattr(settings, "chunk_overlap", DEFAULT_CHUNK_OVERLAP),
        )
    if (Path(settings.faiss_index_dir) / "index.faiss").exists():
        return FaissStore.load(settings.faiss_index_dir, embedder)
    return FaissStore(embedder)


def build_app(
    settings: Settings | None = None,
    *,
    clearance: Confidentiality = Confidentiality.INTERNAL,
    pause_at_gates: bool = False,
    notify=None,
    use_demo_data: bool = True,
    call=None,
    client=None,
    budget=None,
    embedder: Embedder | None = None,
):
    """Create the real dependencies and compile the ARGUS graph."""
    settings = settings or get_settings()
    if call is None:
        call = make_caller(client or build_client(settings), settings)
    embedder = embedder or make_local_embedder()
    store = load_store(settings, embedder, use_demo=use_demo_data)
    tools = {  # demo MCP transports for now; real MCP servers plug in here
        "news": NewsClient(demo_transport),
        "government": GovernmentClient(demo_transport),
        "market": MarketClient(demo_transport),
    }
    return build_graph(
        call=call,
        budget=budget or ControlBudget.from_settings(settings),
        store=store,
        clearance=clearance,
        tools=tools,
        checkpointer=make_checkpointer(settings.checkpoint_db),
        notify=notify,
        pause_at_gates=pause_at_gates,
    )


def _notify(entry: dict) -> None:
    print(f"[reviewer notified] {entry['reason']}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="argus", description="ARGUS risk-intelligence agent")
    p.add_argument("query", nargs="?", default=None, help="question to investigate (default: demo question)")
    p.add_argument("--thread-id", default="default", help="name of this investigation")
    p.add_argument("--pause", action="store_true", help="pause at the human-review gates")
    p.add_argument("--resume", action="store_true", help="resume a paused investigation")
    p.add_argument("--no-demo", action="store_true", help="use the saved FAISS index, not demo documents")
    return p.parse_args(argv)


def main(argv=None, *, graph_factory=build_app) -> int:
    args = parse_args(argv)
    graph = graph_factory(
        use_demo_data=not args.no_demo,
        pause_at_gates=args.pause or args.resume,
        notify=_notify,
    )
    try:
        if args.resume:
            final = resume_investigation(graph, thread_id=args.thread_id)
        else:
            final = run_investigation(graph, args.query or DEMO_QUESTION, thread_id=args.thread_id)
    except InputRejected as exc:
        print(f"Question rejected: {exc}", file=sys.stderr)
        return 2

    waiting = graph.get_state(thread_config(args.thread_id)).next
    if waiting:
        print(
            f"Paused before '{waiting[0]}' for human review. "
            f"Resume with: python -m argus.app --resume --thread-id {args.thread_id}"
        )
        return 0
    print(final["final_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())