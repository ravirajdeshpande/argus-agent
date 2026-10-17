import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 4000
REMOVED_MARKER = "[REMOVED: possible injection]"

_INJECTION = re.compile(
    r"(ignore|disregard)\s+(all\s+|any\s+)?(previous|prior|above)\s+instructions"
    r"|reveal\s+(your\s+)?(system\s+)?prompt",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class InputCheck:
    allowed: bool
    text: str
    reason: str | None = None


@dataclass(frozen=True)
class SanitizedText:
    text: str
    flagged: bool


def check_input(text: str, max_chars: int = MAX_INPUT_CHARS) -> InputCheck:
    cleaned = text.strip()
    if not cleaned:
        return InputCheck(False, cleaned, "empty input")
    if len(cleaned) > max_chars:
        return InputCheck(False, cleaned, "input too long")
    if _INJECTION.search(cleaned):
        return InputCheck(False, cleaned, "possible prompt injection")
    return InputCheck(True, cleaned)


def sanitize_retrieved(text: str) -> SanitizedText:
    """Neutralise instruction-like phrases inside retrieved content."""
    cleaned, count = _INJECTION.subn(REMOVED_MARKER, text)
    return SanitizedText(cleaned, flagged=count > 0)


def wrap_untrusted(text: str, source: str) -> str:
    """Fence retrieved text so the model treats it as data, not instructions."""
    safe = text.replace("</untrusted_content>", "")
    return f'<untrusted_content source="{source}">{safe}</untrusted_content>'