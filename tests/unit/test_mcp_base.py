import pytest

from argus.mcp.base import ToolClient, ToolError


class Flaky:
    def __init__(self, failures, error=TimeoutError):
        self.failures = failures
        self.error = error
        self.calls = 0

    def __call__(self, tool, args):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error("boom")
        return {"ok": True}


def no_sleep(_):
    return None


def test_success_calls_once():
    t = Flaky(0)
    assert ToolClient(t, sleep=no_sleep)._call("x") == {"ok": True}
    assert t.calls == 1


def test_retries_with_exponential_backoff():
    sleeps = []
    t = Flaky(2)
    client = ToolClient(t, max_retries=2, base_delay=0.5, sleep=sleeps.append)
    assert client._call("x") == {"ok": True}
    assert t.calls == 3
    assert sleeps == [0.5, 1.0]


def test_gives_up_after_max_retries():
    t = Flaky(10)
    with pytest.raises(ToolError):
        ToolClient(t, max_retries=2, sleep=no_sleep)._call("x")
    assert t.calls == 3


def test_connection_errors_are_retried():
    t = Flaky(1, ConnectionError)
    assert ToolClient(t, sleep=no_sleep)._call("x") == {"ok": True}


def test_other_errors_are_not_retried():
    t = Flaky(5, ValueError)
    with pytest.raises(ValueError):
        ToolClient(t, sleep=no_sleep)._call("x")
    assert t.calls == 1


def test_falls_back_to_alternate_source():
    primary, backup = Flaky(10), Flaky(0)
    client = ToolClient(primary, backup, max_retries=1, sleep=no_sleep)
    assert client._call("x") == {"ok": True}
    assert primary.calls == 2
    assert backup.calls == 1


def test_all_sources_failing_raises_tool_error():
    client = ToolClient(Flaky(10), Flaky(10), max_retries=1, sleep=no_sleep)
    with pytest.raises(ToolError):
        client._call("x")