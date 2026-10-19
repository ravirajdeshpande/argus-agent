import pytest

from argus.rag.ingest import (
    build_chunks,
    chunk_text,
    clean_text,
    enrich_metadata,
    ingest_document,
)


def test_clean_text_collapses_whitespace():
    assert clean_text("  a \n\n b\t c  ") == "a b c"


def test_chunk_text_short_text_is_one_chunk():
    assert chunk_text("one two three", chunk_size=10, overlap=2) == ["one two three"]


def test_chunk_text_overlaps_and_covers_all_words():
    text = " ".join(str(i) for i in range(10))
    chunks = chunk_text(text, chunk_size=4, overlap=1)
    assert chunks == ["0 1 2 3", "3 4 5 6", "6 7 8 9"]


def test_chunk_text_empty_returns_no_chunks():
    assert chunk_text("   ", chunk_size=5, overlap=1) == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (5, 5), (5, 6), (5, -1)])
def test_chunk_text_rejects_bad_sizes(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a b c", size, overlap)


def test_enrich_metadata_fills_defaults_and_fails_closed():
    meta = enrich_metadata({"supplier": "Acme"})
    assert meta["supplier"] == "Acme"
    assert meta["doc_type"] == "unknown"
    assert meta["confidentiality"] == "restricted"
    assert meta["date"] is None
    assert meta["expires"] is None


def test_enrich_metadata_normalises_confidentiality():
    assert enrich_metadata({"confidentiality": " Internal "})["confidentiality"] == "internal"


def test_enrich_metadata_keeps_extra_keys():
    assert enrich_metadata({"owner": "ops"})["owner"] == "ops"


def test_build_chunks_attach_metadata_and_index():
    chunks = build_chunks(
        "a b c d e f",
        {"supplier": "Acme", "confidentiality": "internal"},
        chunk_size=3,
        overlap=0,
    )
    assert [c["text"] for c in chunks] == ["a b c", "d e f"]
    assert chunks[1]["metadata"]["chunk_index"] == 1
    assert chunks[0]["metadata"]["supplier"] == "Acme"
    assert chunks[0]["metadata"]["confidentiality"] == "internal"


def test_chunk_metadata_is_not_shared_between_chunks():
    chunks = build_chunks("a b c d", {}, chunk_size=2, overlap=0)
    chunks[0]["metadata"]["x"] = 1
    assert "x" not in chunks[1]["metadata"]


def test_ingest_document_reads_file_and_tags_source(tmp_path):
    f = tmp_path / "contract.txt"
    f.write_text("Supplier   Acme\n\nsupplies chips.", encoding="utf-8")
    chunks = ingest_document(f, {"doc_type": "contract"}, chunk_size=50, overlap=5)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Supplier Acme supplies chips."
    assert chunks[0]["metadata"]["source"] == "contract.txt"
    assert chunks[0]["metadata"]["doc_type"] == "contract"