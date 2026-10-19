import json

import pytest

from argus.config.settings import Settings
from argus.evaluation.datasets.demo_scenario import DEMO_DOCUMENTS
from argus.guardrails.access_control import Confidentiality
from argus.rag.build_index import build_store, main, read_documents
from argus.rag.faiss_store import FaissStore
from argus.rag.retrieval import retrieve

VOCAB = ["supplier", "inventory", "contract"]


def fake_embed(texts):
    return [[float(t.lower().count(w)) for w in VOCAB] for t in texts]


def make_settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path, faiss_index_dir=tmp_path / "faiss")


def write(raw, name, text, meta=None):
    path = raw / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if meta is not None:
        (raw / f"{name}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_read_documents_reads_text_files_with_sidecar_metadata(tmp_path):
    raw = tmp_path / "raw"
    write(raw, "contracts/c1.txt", "supplier contract", {"doc_type": "contract", "confidentiality": "internal"})
    write(raw, "notes.md", "inventory note")
    write(raw, "image.png", "not text")
    docs = read_documents(raw)
    assert [d["name"] for d in docs] == ["contracts/c1.txt", "notes.md"]
    assert docs[0]["metadata"]["doc_type"] == "contract"
    assert docs[1]["metadata"] == {}


def test_read_documents_skips_empty_files_and_missing_folders(tmp_path):
    raw = tmp_path / "raw"
    write(raw, "empty.txt", "   ")
    assert read_documents(raw) == []
    assert read_documents(tmp_path / "missing") == []


def test_invalid_sidecar_metadata_is_reported_by_file_name(tmp_path):
    raw = tmp_path / "raw"
    write(raw, "a.txt", "supplier")
    (raw / "a.txt.meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="a.txt.meta.json"):
        read_documents(raw)


def test_non_object_sidecar_is_rejected(tmp_path):
    raw = tmp_path / "raw"
    write(raw, "a.txt", "supplier", ["list"])
    with pytest.raises(ValueError, match="JSON object"):
        read_documents(raw)


def test_build_store_splits_long_documents_into_chunks():
    docs = [{"name": "a.txt", "text": " ".join(["supplier"] * 30), "metadata": {"confidentiality": "internal"}}]
    store = build_store(docs, fake_embed, chunk_size=10, overlap=2)
    assert len(store) > 1


def test_documents_without_metadata_fail_closed(tmp_path):
    docs = [{"name": "a.txt", "text": "supplier contract", "metadata": {}}]
    store = build_store(docs, fake_embed)
    assert retrieve(store, "supplier", Confidentiality.INTERNAL) == []
    assert retrieve(store, "supplier", Confidentiality.RESTRICTED)


def test_main_builds_and_saves_a_loadable_index(tmp_path, capsys):
    raw = tmp_path / "raw"
    write(raw, "c1.txt", "supplier contract", {"confidentiality": "internal", "doc_type": "contract"})
    s = make_settings(tmp_path)
    assert main(["--raw-dir", str(raw)], embedder=fake_embed, settings=s) == 0
    assert "Indexed" in capsys.readouterr().out
    loaded = FaissStore.load(s.faiss_index_dir, fake_embed)
    assert len(loaded) == 1
    assert retrieve(loaded, "supplier", Confidentiality.INTERNAL)[0]["metadata"]["source"] == "c1.txt"


def test_main_without_documents_returns_1(tmp_path, capsys):
    assert main(["--raw-dir", str(tmp_path / "empty")], embedder=fake_embed, settings=make_settings(tmp_path)) == 1
    assert "No .txt or .md documents" in capsys.readouterr().out


def test_main_demo_indexes_the_demo_documents(tmp_path):
    s = make_settings(tmp_path)
    assert main(["--demo"], embedder=fake_embed, settings=s) == 0
    assert len(FaissStore.load(s.faiss_index_dir, fake_embed)) == len(DEMO_DOCUMENTS)


def test_main_writes_to_a_custom_output_folder(tmp_path):
    raw = tmp_path / "raw"
    write(raw, "c1.txt", "supplier", {"confidentiality": "internal"})
    out = tmp_path / "custom"
    assert main(["--raw-dir", str(raw), "--out-dir", str(out)], embedder=fake_embed, settings=make_settings(tmp_path)) == 0
    assert (out / "index.faiss").exists()