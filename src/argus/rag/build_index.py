import argparse
import json
from pathlib import Path

from argus.config.settings import get_settings
from argus.evaluation.datasets.demo_scenario import DEMO_DOCUMENTS, build_demo_store
from argus.rag.faiss_store import Embedder, FaissStore, make_local_embedder
from argus.rag.ingest import build_chunks

TEXT_SUFFIXES = {".txt", ".md"}
DEFAULT_CHUNK_SIZE = 120
DEFAULT_CHUNK_OVERLAP = 20


def read_documents(raw_dir) -> list[dict]:
    """Read .txt/.md files; an optional '<file>.meta.json' beside each holds its metadata."""
    root = Path(raw_dir)
    if not root.is_dir():
        return []
    docs = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        metadata: dict = {}
        meta_path = path.with_name(path.name + ".meta.json")
        if meta_path.exists():
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                raise ValueError(f"invalid metadata file {meta_path.name}: {exc}") from exc
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata file {meta_path.name} must hold a JSON object")
        docs.append({"name": path.relative_to(root).as_posix(), "text": text, "metadata": metadata})
    return docs


def build_store(
    docs: list[dict],
    embedder: Embedder,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> FaissStore:
    store = FaissStore(embedder)
    for doc in docs:
        store.add(build_chunks(doc["text"], {**doc["metadata"], "source": doc["name"]}, chunk_size, overlap))
    return store


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="argus-index", description="Build the ARGUS FAISS index")
    p.add_argument("--raw-dir", default=None, help="folder of .txt/.md files (default: data/raw)")
    p.add_argument("--out-dir", default=None, help="index folder (default: Settings.faiss_index_dir)")
    p.add_argument("--demo", action="store_true", help="index the built-in demo documents instead")
    return p.parse_args(argv)


def main(argv=None, *, embedder: Embedder | None = None, settings=None) -> int:
    args = parse_args(argv)
    settings = settings or get_settings()
    chunk_size = getattr(settings, "chunk_size", DEFAULT_CHUNK_SIZE)
    overlap = getattr(settings, "chunk_overlap", DEFAULT_CHUNK_OVERLAP)
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(settings.data_dir) / "raw"
    out_dir = Path(args.out_dir) if args.out_dir else Path(settings.faiss_index_dir)

    docs = [] if args.demo else read_documents(raw_dir)
    if not args.demo and not docs:
        print(f"No .txt or .md documents found in {raw_dir}")
        return 1

    embedder = embedder or make_local_embedder()
    if args.demo:
        store = build_demo_store(embedder, chunk_size=chunk_size, overlap=overlap)
    else:
        store = build_store(docs, embedder, chunk_size=chunk_size, overlap=overlap)

    out_dir.mkdir(parents=True, exist_ok=True)
    store.save(out_dir)
    print(f"Indexed {len(store)} chunks from {len(docs) or len(DEMO_DOCUMENTS)} documents into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())