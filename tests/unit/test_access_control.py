from argus.guardrails.access_control import (
    Confidentiality,
    allowed_levels,
    filter_allowed,
    is_allowed,
    parse_confidentiality,
)


def test_parse_is_case_insensitive():
    assert parse_confidentiality("Internal") == Confidentiality.INTERNAL


def test_unknown_or_missing_level_fails_closed():
    assert parse_confidentiality("secret-ish") == Confidentiality.RESTRICTED
    assert parse_confidentiality(None) == Confidentiality.RESTRICTED


def test_is_allowed_respects_clearance():
    clearance = Confidentiality.INTERNAL
    assert is_allowed({"confidentiality": "public"}, clearance)
    assert is_allowed({"confidentiality": "internal"}, clearance)
    assert not is_allowed({"confidentiality": "confidential"}, clearance)


def test_missing_confidentiality_is_blocked():
    assert not is_allowed({}, Confidentiality.INTERNAL)


def test_allowed_levels_for_metadata_filter():
    assert allowed_levels(Confidentiality.INTERNAL) == {"public", "internal"}


def test_filter_allowed_keeps_order_and_drops_blocked():
    docs = [
        {"text": "a", "metadata": {"confidentiality": "public"}},
        {"text": "b", "metadata": {"confidentiality": "restricted"}},
        {"text": "c", "metadata": {"confidentiality": "internal"}},
        {"text": "d"},
    ]
    kept = filter_allowed(docs, Confidentiality.INTERNAL)
    assert [d["text"] for d in kept] == ["a", "c"]