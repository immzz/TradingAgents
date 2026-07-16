import pytest

from tradingagents.dataflows.input_capture import capture_session, execute, records_sha256


def test_capture_and_replay_return_identical_tool_output_without_live_call():
    live_calls = []
    with capture_session() as captured:
        result = execute(
            "get_stock_data",
            ("AAPL", "2026-01-01", "2026-01-02"),
            {},
            lambda: live_calls.append(1) or "rows",
        )
    assert result == "rows"
    assert live_calls == [1]
    assert records_sha256(captured.records)

    with capture_session(captured.records):
        replayed = execute(
            "get_stock_data",
            ("AAPL", "2026-01-01", "2026-01-02"),
            {},
            lambda: (_ for _ in ()).throw(AssertionError("live call must not execute")),
        )
    assert replayed == "rows"


def test_replay_rejects_unrecorded_tool_call_and_raw_news_tool():
    with capture_session([]):
        with pytest.raises(RuntimeError, match="unrecorded"):
            execute("get_stock_data", ("MSFT", "a", "b"), {}, lambda: "fresh")
    with pytest.raises(ValueError, match="disallowed tool"):
        records_sha256(
            [
                {
                    "schema_version": "research-tool-capture.v1",
                    "tool": "get_news",
                    "args": ["AAPL"],
                    "kwargs": {},
                    "call_key": "bad",
                    "result": "headline",
                    "result_sha256": "bad",
                }
            ]
        )
