"""Quote prints inside a read_url body stay evidence; wrapper lengths do not.

Synthetic fixtures only. The numbers below are labelled reconstructions of a
public ticker shape (string-encoded last / open24h / high24h / low24h inside a
reader envelope whose ``length`` is the character count). They are not a live
capture and not an instruction to trade.

``get_market_data`` sampling is unchanged: an every-Nth candle payload still
exposes only the returned rows and its stride policy, and this module does not
treat those rows as a continuous series.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.loop import AgentLoop
from src.agent.tools import BaseTool, ToolRegistry

pytestmark = pytest.mark.unit

# Clearly synthetic. Lows are fillers so each fixture has last/open/high/low;
# the other three prints are the values a reader body must be able to support.
ETH_QUOTES = {
    "last": "2246.35",
    "open24h": "2210.17",
    "high24h": "2254.25",
    "low24h": "2198.40",
}
BTC_QUOTES = {
    "last": "76025.6",
    "open24h": "74887.9",
    "high24h": "76171.7",
    "low24h": "74512.3",
}
ETH_LENGTH = 453
BTC_LENGTH = 460
ETH_URL = "https://www.okx.com/api/v5/market/ticker?instId=ETH-USDT"
BTC_URL = "https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT"
VOLUME = "12345.6"


def _ticker(inst_id: str, quotes: dict[str, str], *, comma_last: bool = False) -> dict[str, Any]:
    """One synthetic ticker object. Prices are strings, as a public ticker encodes them."""
    last = "2,246.35" if comma_last else quotes["last"]
    return {
        "code": "0",
        "msg": "",
        "data": [
            {
                "instType": "SPOT",
                "instId": inst_id,
                "last": last,
                "open24h": quotes["open24h"],
                "high24h": quotes["high24h"],
                "low24h": quotes["low24h"],
                "vol24h": VOLUME,
                "ts": "1700000000000",
            }
        ],
    }


def _json_text(inst_id: str, quotes: dict[str, str], *, comma_last: bool = False) -> str:
    return json.dumps(_ticker(inst_id, quotes, comma_last=comma_last), separators=(",", ":"))


def _markdown(url: str, inst_id: str, quotes: dict[str, str]) -> str:
    """Reader text: a short header, then the ticker JSON. The header is not the body."""
    return (
        f"Title: {inst_id}\n"
        f"URL Source: {url}\n\n"
        "Markdown Content:\n"
        + _json_text(inst_id, quotes)
    )


def _envelope(url: str, content: Any, length: int, *, title: str = "") -> str:
    return json.dumps(
        {
            "status": "ok",
            "title": title,
            "url": url,
            "content": content,
            "length": length,
        },
        ensure_ascii=False,
    )


def _ingest(
    tmp_path: Path,
    envelopes: list[tuple[str, str, str]],
) -> GroundingLedger:
    ledger = GroundingLedger(
        run_dir=tmp_path,
        user_message="Report the ETH-USDT and BTC-USDT ticker prints.",
    )
    for call_id, url, result in envelopes:
        ledger.ingest_tool_result(
            tool_name="read_url",
            arguments={"url": url},
            result=result,
            call_id=call_id,
            success=True,
        )
    return ledger


def _answer(prose: str, *rows: str) -> str:
    return prose + "\n\n```figures\n" + "\n".join(rows) + "\n```"


def _reasons(result) -> list[str]:
    return [str(issue.get("reason")) for issue in result.issues]


ETH_PROSE = (
    "ETH-USDT (OKX) last 2246.35 USDT, open 2210.17 USDT, "
    "high 2254.25 USDT, low 2198.40 USDT."
)
BTC_PROSE = (
    "BTC-USDT (OKX) last 76025.6 USDT, open 74887.9 USDT, "
    "high 76171.7 USDT, low 74512.3 USDT."
)
ETH_ROWS = (
    "2246.35 | observed | ETH-USDT last | eth::content.data[0].last",
    "2210.17 | observed | ETH-USDT open | eth::open24h",
    "2254.25 | observed | ETH-USDT high | eth::high24h",
    "2198.40 | observed | ETH-USDT low | eth::low24h",
)
BTC_ROWS = (
    "76025.6 | observed | BTC-USDT last | btc::last",
    "74887.9 | observed | BTC-USDT open | btc::open24h",
    "76171.7 | observed | BTC-USDT high | btc::high24h",
    "74512.3 | observed | BTC-USDT low | btc::low24h",
)


def _pair(tmp_path: Path, eth_content: Any, btc_content: Any) -> GroundingLedger:
    return _ingest(
        tmp_path,
        [
            ("eth", ETH_URL, _envelope(ETH_URL, eth_content, ETH_LENGTH, title="ETH-USDT")),
            ("btc", BTC_URL, _envelope(BTC_URL, btc_content, BTC_LENGTH, title="BTC-USDT")),
        ],
    )


def test_markdown_wrapped_ticker_json_keeps_quoted_ohlc_and_drops_length(tmp_path: Path) -> None:
    """The reported shape: markdown around JSON text, plus a wrapper length."""
    ledger = _pair(tmp_path, _markdown(ETH_URL, "ETH-USDT", ETH_QUOTES), _markdown(BTC_URL, "BTC-USDT", BTC_QUOTES))

    fields = {(record.call_id, record.field, record.value) for record in ledger._evidence}
    assert ("eth", "length", ETH_LENGTH) not in fields
    assert ("btc", "length", BTC_LENGTH) not in fields
    assert not any(record.field == "length" or record.value in {ETH_LENGTH, BTC_LENGTH} for record in ledger._evidence)
    eth_last = next(record for record in ledger._evidence if record.call_id == "eth" and record.field.endswith(".last"))
    assert eth_last.value == 2246.35
    assert eth_last.symbol == "ETH-USDT"
    assert eth_last.source == "okx"
    assert eth_last.tool == "read_url"
    assert eth_last.status == "observed"
    assert eth_last.currency == "USDT"
    btc_last = next(record for record in ledger._evidence if record.call_id == "btc" and record.field.endswith(".last"))
    assert btc_last.symbol == "BTC-USDT"
    assert btc_last.value == 76025.6
    # String-encoded volume and timestamp are not quote leaves and stay out.
    assert all(record.field.rsplit(".", 1)[-1] not in {"vol24h", "ts", "code"} for record in ledger._evidence)

    result = ledger.validate_final_answer(_answer(ETH_PROSE + "\n" + BTC_PROSE, *ETH_ROWS, *BTC_ROWS))

    assert result.valid is True, result.issues
    assert "2246.35" in result.released_text
    assert "76025.6" in result.released_text
    assert "```figures" not in result.released_text

    # A ref that names the call, rather than one leaf, still sees only that call's body.
    by_call = ledger.validate_final_answer(
        _answer(
            ETH_PROSE + "\n" + BTC_PROSE,
            "2246.35 | observed | ETH-USDT last | eth",
            "76025.6 | observed | BTC-USDT last | btc",
            *ETH_ROWS[1:],
            *BTC_ROWS[1:],
        )
    )
    assert by_call.valid is True, by_call.issues


@pytest.mark.parametrize(
    "pack",
    [
        "raw-json",
        "fenced",
        "double-encoded",
        "structured-object",
        "text-block-list",
        "comma-grouped",
        "numeric-json",
    ],
)
def test_reader_body_shapes_keep_the_cited_last(tmp_path: Path, pack: str) -> None:
    """JSON-in-text, fenced, wrapped, and already-parsed bodies are the same evidence."""
    if pack == "raw-json":
        content: Any = _json_text("ETH-USDT", ETH_QUOTES)
    elif pack == "fenced":
        content = "```json\n" + _json_text("ETH-USDT", ETH_QUOTES) + "\n```"
    elif pack == "double-encoded":
        content = json.dumps(_json_text("ETH-USDT", ETH_QUOTES))
    elif pack == "structured-object":
        content = _ticker("ETH-USDT", ETH_QUOTES)
    elif pack == "text-block-list":
        content = [{"type": "text", "text": _json_text("ETH-USDT", ETH_QUOTES)}]
    elif pack == "comma-grouped":
        content = _json_text("ETH-USDT", ETH_QUOTES, comma_last=True)
    else:
        body = _ticker("ETH-USDT", ETH_QUOTES)
        row = body["data"][0]
        for key in ("last", "open24h", "high24h", "low24h"):
            row[key] = float(str(row[key]).replace(",", ""))
        content = json.dumps(body)
    ledger = _pair(
        tmp_path,
        content,
        _markdown(BTC_URL, "BTC-USDT", BTC_QUOTES),
    )
    prose = "ETH-USDT (OKX) last 2,246.35 USDT.\n" + BTC_PROSE
    result = ledger.validate_final_answer(
        _answer(
            prose,
            "2,246.35 | observed | ETH-USDT last | eth::last",
            *BTC_ROWS,
        )
    )
    assert result.valid is True, (pack, result.issues)


def test_a_derived_gap_must_still_be_the_arithmetic(tmp_path: Path) -> None:
    """Observed prints can anchor a derivation; they do not bless a wrong result."""
    ledger = _pair(
        tmp_path,
        _markdown(ETH_URL, "ETH-USDT", ETH_QUOTES),
        _markdown(BTC_URL, "BTC-USDT", BTC_QUOTES),
    )
    prose = (
        ETH_PROSE
        + "\nETH-USDT last-minus-open gap is 36.18 USDT.\n"
        + BTC_PROSE
    )
    derived = "36.18 | derived | 2246.35 - 2210.17 | eth"
    ok = ledger.validate_final_answer(_answer(prose, *ETH_ROWS, derived, *BTC_ROWS))
    assert ok.valid is True, ok.issues

    bad = ledger.validate_final_answer(
        _answer(
            ETH_PROSE + "\nETH-USDT last-minus-open gap is 99.99 USDT.",
            *ETH_ROWS,
            "99.99 | derived | 2246.35 - 2210.17 | eth",
        )
    )
    assert bad.valid is False
    assert _reasons(bad) == ["derivation_result_mismatch"]


def test_fabricated_cross_call_cross_symbol_wrong_field_and_length_are_rejected(
    tmp_path: Path,
) -> None:
    ledger = _pair(
        tmp_path,
        _markdown(ETH_URL, "ETH-USDT", ETH_QUOTES),
        _markdown(BTC_URL, "BTC-USDT", BTC_QUOTES),
    )

    fabricated = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 1999.01 USDT.",
            "1999.01 | observed | ETH-USDT last | eth::last",
        )
    )
    borrowed = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 2246.35 USDT.",
            "2246.35 | observed | ETH-USDT last | btc::last",
        )
    )
    other_instrument = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 76025.6 USDT.",
            "76025.6 | observed | ETH-USDT last | btc::content.data[0].last",
        )
    )
    wrong_field = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 2254.25 USDT.",
            "2254.25 | observed | ETH-USDT last | eth::last",
        )
    )
    length_as_print = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 453.0 USDT.",
            "453.0 | observed | ETH-USDT last | eth",
        )
    )
    length_field = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 453 USDT.",
            "453 | observed | ETH-USDT last | eth::length",
        )
    )
    other_length = ledger.validate_final_answer(
        _answer(
            "BTC-USDT (OKX) last 460.0 USDT.",
            "460.0 | observed | BTC-USDT last | btc::length",
        )
    )
    volume_string = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 12345.6 USDT.",
            "12345.6 | observed | ETH-USDT last | eth::vol24h",
        )
    )

    assert _reasons(fabricated) == ["not_in_referenced_call"]
    assert _reasons(borrowed) == ["not_in_referenced_call"]
    assert _reasons(other_instrument) == ["not_in_referenced_call"]
    assert _reasons(wrong_field) == ["not_in_referenced_call"]
    assert _reasons(length_as_print) == ["not_in_referenced_call"]
    assert _reasons(length_field) == ["not_in_referenced_call"]
    assert _reasons(other_length) == ["not_in_referenced_call"]
    assert _reasons(volume_string) == ["not_in_referenced_call"]


def test_invisible_citation_is_still_rejected(tmp_path: Path) -> None:
    ledger = _pair(
        tmp_path,
        _markdown(ETH_URL, "ETH-USDT", ETH_QUOTES),
        _markdown(BTC_URL, "BTC-USDT", BTC_QUOTES),
    )
    result = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 2246.35 USDT.",
            "2246.35 | cited | venue ticker print | eth",
        )
    )
    assert _reasons(result) == ["citation_not_visible"]


def test_another_tool_string_body_is_not_indexed(tmp_path: Path) -> None:
    """The reader exception does not turn every tool's text blob into prints."""
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Report the ETH-USDT ticker.")
    ledger.ingest_tool_result(
        tool_name="echo_report",
        arguments={},
        result=_envelope(ETH_URL, _json_text("ETH-USDT", ETH_QUOTES), ETH_LENGTH),
        call_id="echo",
        success=True,
    )
    assert not any(record.value == 2246.35 for record in ledger._evidence)
    result = ledger.validate_final_answer(
        _answer(
            "ETH-USDT last 2246.35 USDT.",
            "2246.35 | observed | ETH-USDT last | echo::last",
        )
    )
    assert result.valid is False
    assert "not_in_referenced_call" in _reasons(result)


class _Response:
    def __init__(self, *, content: str = "", tool_calls: list[SimpleNamespace] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.reasoning_content = None
        self.has_tool_calls = bool(self.tool_calls)


class _ScriptedLLM:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        on_text_chunk: Callable[[str], None] | None = None,
        on_reasoning_chunk: Callable[[str], None] | None = None,
        timeout: int | None = None,
        idle_timeout_s: float | None = None,
        should_cancel: Callable[[], bool] | None = None,
        on_waiting_chunk=None,
    ) -> _Response:
        self.calls += 1
        response = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if response.content and on_text_chunk:
            on_text_chunk(response.content)
        return response

    def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> _Response:
        return _Response()


class _ReaderTool(BaseTool):
    """Offline stand-in. It returns the synthetic envelope and does not fetch."""

    name = "read_url"
    description = "Return a synthetic public ticker envelope for an offline grounding run."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }
    repeatable = True
    is_readonly = True

    def execute(self, **kwargs: Any) -> str:
        url = str(kwargs.get("url") or "")
        if "ETH-USDT" in url:
            return _envelope(ETH_URL, _markdown(ETH_URL, "ETH-USDT", ETH_QUOTES), ETH_LENGTH, title="ETH-USDT")
        if "BTC-USDT" in url:
            return _envelope(BTC_URL, _markdown(BTC_URL, "BTC-USDT", BTC_QUOTES), BTC_LENGTH, title="BTC-USDT")
        return json.dumps({"status": "error", "error": "synthetic fixture has no such url"})


def _tool_call(call_id: str, url: str) -> SimpleNamespace:
    return SimpleNamespace(id=call_id, name="read_url", arguments={"url": url})


def _run(tmp_path: Path, responses: list[_Response]) -> dict[str, Any]:
    registry = ToolRegistry()
    registry.register(_ReaderTool())
    agent = AgentLoop(registry=registry, llm=_ScriptedLLM(responses), max_iterations=8)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    agent.memory.run_dir = str(run_dir)
    return agent.run("Report the ETH-USDT and BTC-USDT ticker prints from the public pages.")


_OBSERVED_ANSWER = _answer(ETH_PROSE + "\n" + BTC_PROSE, *ETH_ROWS, *BTC_ROWS)


def test_offline_loop_retains_sourced_prints_without_degrading(tmp_path: Path) -> None:
    """A scripted loop, no model and no network: cited body prints are the answer."""
    result = _run(
        tmp_path,
        [
            _Response(tool_calls=[_tool_call("eth", ETH_URL)]),
            _Response(tool_calls=[_tool_call("btc", BTC_URL)]),
            _Response(content=_OBSERVED_ANSWER),
        ],
    )
    assert result["status"] == "success"
    assert result.get("degraded") is not True
    for number in ("2246.35", "2210.17", "2254.25", "2198.40", "76025.6", "74887.9", "76171.7", "74512.3"):
        assert number in result["content"]
    assert "```figures" not in result["content"]
    assert "453" not in result["content"]
    assert "460" not in result["content"]


def test_offline_loop_still_redacts_a_fabricated_print(tmp_path: Path) -> None:
    """The same envelopes still reject a number that was not in either body."""
    invented = _answer(
        ETH_PROSE + " Invented print 1999.01 USDT.\n" + BTC_PROSE,
        *ETH_ROWS,
        "1999.01 | observed | ETH-USDT last | eth::last",
        *BTC_ROWS,
    )
    result = _run(
        tmp_path,
        [
            _Response(tool_calls=[_tool_call("eth", ETH_URL)]),
            _Response(tool_calls=[_tool_call("btc", BTC_URL)]),
            _Response(content=invented),
        ],
    )
    assert result["status"] == "success"
    assert result.get("degraded") is True
    assert "1999.01" not in result["content"]
    assert "2246.35" in result["content"]
    assert "76025.6" in result["content"]
