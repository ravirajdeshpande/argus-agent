from enum import IntEnum


class Confidentiality(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3


def parse_confidentiality(value) -> Confidentiality:
    """Unknown or missing labels are treated as RESTRICTED (fail closed)."""
    try:
        return Confidentiality[str(value).strip().upper()]
    except KeyError:
        return Confidentiality.RESTRICTED


def is_allowed(metadata: dict, clearance: Confidentiality) -> bool:
    return parse_confidentiality(metadata.get("confidentiality")) <= clearance


def allowed_levels(clearance: Confidentiality) -> set[str]:
    """Level names a user may see, for use as a retrieval metadata filter."""
    return {level.name.lower() for level in Confidentiality if level <= clearance}


def filter_allowed(items: list[dict], clearance: Confidentiality) -> list[dict]:
    """Drop items the user may not see. Each item has a 'metadata' dict."""
    return [i for i in items if is_allowed(i.get("metadata", {}), clearance)]