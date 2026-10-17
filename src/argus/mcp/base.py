import time
from collections.abc import Callable
from typing import Any

Transport = Callable[[str, dict], Any]

RETRYABLE = (TimeoutError, ConnectionError)


class ToolError(RuntimeError):
    """A tool call failed after all retries and fallbacks."""


class ToolClient:
    """Calls MCP tools with bounded retry/backoff and optional fallback sources."""

    def __init__(
        self,
        transport: Transport,
        *fallbacks: Transport,
        max_retries: int = 2,
        base_delay: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._transports = [transport, *fallbacks]
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._sleep = sleep

    def _call_one(self, transport: Transport, tool: str, args: dict):
        for attempt in range(self._max_retries + 1):
            try:
                return transport(tool, args)
            except RETRYABLE as exc:
                if attempt == self._max_retries:
                    raise ToolError(f"{tool} failed after {attempt + 1} attempts") from exc
                self._sleep(self._base_delay * 2**attempt)

    def _call(self, tool: str, **args):
        last: ToolError | None = None
        for transport in self._transports:
            try:
                return self._call_one(transport, tool, args)
            except ToolError as exc:
                last = exc
        raise ToolError(f"{tool}: all sources failed") from last