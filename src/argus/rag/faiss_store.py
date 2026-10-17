import json
from collections.abc import Callable
from pathlib import Path

import faiss
import numpy as np

Embedder = Callable[[list[str]], list[list[float]]]

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _normalise(vectors: list[list[float]]) -> np.ndarray:
    """Unit-length vectors, so inner product equals cosine similarity."""
    arr = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def make_embedder(client, model: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    """Embedder backed by an OpenAI-style client."""

    def embed(texts: list[str]) -> list[list[float]]:
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    return embed


class FaissStore:
    """FAISS index plus the chunk texts and metadata it was built from."""

    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._index = None
        self._chunks: list[dict] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        vectors = _normalise(self._embedder([c["text"] for c in chunks]))
        if self._index is None:
            self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors)
        self._chunks.extend(chunks)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._index is None or not self._chunks:
            return []
        q = _normalise(self._embedder([query]))
        scores, ids = self._index.search(q, min(k, len(self._chunks)))
        return [
            {
                "text": self._chunks[i]["text"],
                "metadata": self._chunks[i]["metadata"],
                "score": float(score),
            }
            for score, i in zip(scores[0], ids[0])
        ]

    def save(self, folder: str | Path) -> None:
        if self._index is None:
            raise ValueError("nothing to save: the store is empty")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(folder / "index.faiss"))
        (folder / "chunks.json").write_text(json.dumps(self._chunks), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path, embedder: Embedder) -> "FaissStore":
        folder = Path(folder)
        store = cls(embedder)
        store._index = faiss.read_index(str(folder / "index.faiss"))
        store._chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        return store