from types import SimpleNamespace

import pytest

from argus.rag.faiss_store import FaissStore, make_embedder

VOCAB = ["chip", "port", "contract", "fruit"]


def fake_embed(texts):
    return [[float(t.lower().count(w)) for w in VOCAB] for t in texts]


def chunk(text, **meta):
    return {"text": text, "metadata": meta}


def make_store():
    store = FaissStore(fake_embed)
    store.add(
        [
            chunk("chip supply chip", supplier="Acme"),
            chunk("port closed", supplier="Globex"),
            chunk("fruit market", supplier="Fresh"),
        ]
    )
    return store


def test_len_counts_chunks():
    assert len(make_store()) == 3


def test_search_returns_most_similar_first_with_score_and_metadata():
    results = make_store().search("chip", k=2)
    assert results[0]["text"] == "chip supply chip"
    assert results[0]["metadata"]["supplier"] == "Acme"
    assert results[0]["score"] == pytest.approx(1.0)
    assert len(results) == 2


def test_search_k_larger_than_store_returns_all():
    assert len(make_store().search("chip", k=10)) == 3


def test_query_with_no_known_words_does_not_crash():
    assert len(make_store().search("zzz", k=2)) == 2


def test_search_on_empty_store_returns_empty_list():
    assert FaissStore(fake_embed).search("chip") == []


def test_add_empty_list_is_noop():
    store = FaissStore(fake_embed)
    store.add([])
    assert len(store) == 0


def test_save_and_load_roundtrip(tmp_path):
    make_store().save(tmp_path / "idx")
    loaded = FaissStore.load(tmp_path / "idx", fake_embed)
    assert len(loaded) == 3
    results = loaded.search("port", k=1)
    assert results[0]["text"] == "port closed"
    assert results[0]["metadata"]["supplier"] == "Globex"


def test_save_empty_store_raises(tmp_path):
    with pytest.raises(ValueError):
        FaissStore(fake_embed).save(tmp_path / "idx")


def test_make_embedder_calls_client_and_returns_vectors():
    calls = {}

    def create(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2]), SimpleNamespace(embedding=[0.3, 0.4])]
        )

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    embed = make_embedder(client, model="emb-model")
    assert embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
    assert calls == {"model": "emb-model", "input": ["a", "b"]}