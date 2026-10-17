from pathlib import Path

from argus.guardrails.access_control import parse_confidentiality

DEFAULT_METADATA = {
    "doc_type": "unknown",
    "supplier": None,
    "product": None,
    "region": None,
    "date": None,
    "expires": None,
}


def clean_text(raw: str) -> str:
    return " ".join(raw.split())


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split into overlapping chunks. Sizes are counted in words."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("need chunk_size > 0 and 0 <= overlap < chunk_size")
    words = clean_text(text).split()
    if not words:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while True:
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def enrich_metadata(raw: dict) -> dict:
    """Add defaults; a missing or unknown confidentiality becomes 'restricted'."""
    meta = {**DEFAULT_METADATA, **raw}
    meta["confidentiality"] = parse_confidentiality(raw.get("confidentiality")).name.lower()
    return meta


def build_chunks(text: str, metadata: dict, chunk_size: int, overlap: int) -> list[dict]:
    base = enrich_metadata(metadata)
    return [
        {"text": t, "metadata": {**base, "chunk_index": i}}
        for i, t in enumerate(chunk_text(text, chunk_size, overlap))
    ]


def ingest_document(
    path: str | Path, metadata: dict, chunk_size: int, overlap: int
) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return build_chunks(text, {**metadata, "source": path.name}, chunk_size, overlap)