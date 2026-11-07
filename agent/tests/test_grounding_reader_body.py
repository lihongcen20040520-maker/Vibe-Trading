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

import src.agent.grounding.evidence as reader_evidence
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
# The public ticker host the isolated run actually calls. Provenance is the
# registrable venue id ``okx``, not the tool name ``read_url``.
ETH_MY_URL = "https://my.okx.com/api/v5/market/ticker?instId=ETH-USDT"
BTC_MY_URL = "https://my.okx.com/api/v5/market/ticker?instId=BTC-USDT"
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
    # Volume and the status code stay out. ``ts`` is source time, not a quote.
    assert all(record.field.rsplit(".", 1)[-1] not in {"vol24h", "code"} for record in ledger._evidence)
    eth_ts = next(record for record in ledger._evidence if record.call_id == "eth" and record.field.endswith(".ts"))
    assert eth_ts.value == 1700000000000
    assert eth_ts.symbol == "ETH-USDT"
    assert eth_ts.currency is None
    assert eth_ts not in ledger._price_records()
    assert eth_last.timestamp == "1700000000000"

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


def test_my_okx_host_instid_query_source_is_okx_and_must_be_named(tmp_path: Path) -> None:
    """``my.okx.com?instId=`` is source okx. Naming OKX passes; naming only the tool does not."""
    assert "instId=ETH-USDT" in ETH_MY_URL and "instId=BTC-USDT" in BTC_MY_URL
    eth = json.loads(_envelope(ETH_MY_URL, _markdown(ETH_MY_URL, "ETH-USDT", ETH_QUOTES), ETH_LENGTH, title="ETH-USDT"))
    btc = json.loads(_envelope(BTC_MY_URL, _markdown(BTC_MY_URL, "BTC-USDT", BTC_QUOTES), BTC_LENGTH, title="BTC-USDT"))
    eth["headers"] = {"content-length": ETH_LENGTH, "elapsed_ms": 12}
    btc["headers"] = {"content-length": BTC_LENGTH, "elapsed_ms": 12}
    ledger = _ingest(
        tmp_path,
        [
            ("eth", ETH_MY_URL, json.dumps(eth)),
            ("btc", BTC_MY_URL, json.dumps(btc)),
        ],
    )
    eth_last = next(record for record in ledger._evidence if record.call_id == "eth" and record.field.endswith(".last"))
    btc_last = next(record for record in ledger._evidence if record.call_id == "btc" and record.field.endswith(".last"))
    assert eth_last.source == "okx" and btc_last.source == "okx"
    assert eth_last.tool == "read_url" and btc_last.tool == "read_url"
    assert eth_last.symbol == "ETH-USDT" and btc_last.symbol == "BTC-USDT"
    assert eth_last.value == 2246.35 and btc_last.value == 76025.6
    assert not any(record.value in {ETH_LENGTH, BTC_LENGTH, 12} for record in ledger._evidence)

    named = ledger.validate_final_answer(_answer(ETH_PROSE + "\n" + BTC_PROSE, *ETH_ROWS, *BTC_ROWS))
    assert named.valid is True, named.issues

    tool_name_only = (
        "ETH-USDT (read_url) last 2246.35 USDT, open 2210.17 USDT, "
        "high 2254.25 USDT, low 2198.40 USDT.\n"
        "BTC-USDT (read_url) last 76025.6 USDT, open 74887.9 USDT, "
        "high 76171.7 USDT, low 74512.3 USDT."
    )
    unnamed = ledger.validate_final_answer(_answer(tool_name_only, *ETH_ROWS, *BTC_ROWS))
    assert unnamed.valid is False
    assert any(issue.get("code") == "data_source_not_surfaced" for issue in unnamed.issues)
    assert "not_in_referenced_call" not in _reasons(unnamed)

    borrowed = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 76025.6 USDT.",
            "76025.6 | observed | ETH-USDT last | btc::last",
        )
    )
    assert _reasons(borrowed) == ["not_in_referenced_call"]


@pytest.mark.parametrize(
    "url",
    [
        "https://my.okx.com.evil.example/api/v5/market/ticker?instId=ETH-USDT",
        "https://okx.com.evil.example/api/v5/market/ticker?instId=ETH-USDT",
        "https://my.okx.com@evil.example/api/v5/market/ticker?instId=ETH-USDT",
        "https://notokx.com/api/v5/market/ticker?instId=ETH-USDT",
    ],
)
def test_lookalike_host_stays_read_url(tmp_path: Path, url: str) -> None:
    """The okx source is the hostname, not the letters okx.com somewhere in the URL."""
    ledger = _ingest(
        tmp_path,
        [("eth", url, _envelope(url, _markdown(url, "ETH-USDT", ETH_QUOTES), ETH_LENGTH, title="ETH-USDT"))],
    )
    last = next(record for record in ledger._evidence if record.field.endswith(".last"))
    assert last.source == "read_url"
    assert last.symbol == "ETH-USDT"
    assert last.value == 2246.35
    via_tool = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (read_url) last 2246.35 USDT, open 2210.17 USDT, high 2254.25 USDT, low 2198.40 USDT.",
            *ETH_ROWS,
        )
    )
    assert via_tool.valid is True, via_tool.issues
    via_okx = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 2246.35 USDT, open 2210.17 USDT, high 2254.25 USDT, low 2198.40 USDT.",
            *ETH_ROWS,
        )
    )
    assert via_okx.valid is False
    assert any(issue.get("code") == "data_source_not_surfaced" for issue in via_okx.issues)


def test_unlabeled_print_in_a_multi_instrument_body_is_not_evidence(tmp_path: Path) -> None:
    """A second instrument's number without instId does not inherit the URL symbol."""
    body = {
        "data": [
            {
                "instId": "ETH-USDT",
                "last": "2246.35",
                "open24h": "2210.17",
                "high24h": "2254.25",
                "low24h": "2198.40",
            },
            {
                "instId": "BTC-USDT",
                "last": "76025.6",
                "open24h": "74887.9",
                "high24h": "76171.7",
                "low24h": "74512.3",
            },
            {"last": "1111.11"},
        ]
    }
    ledger = _ingest(
        tmp_path,
        [("page", ETH_MY_URL, _envelope(ETH_MY_URL, json.dumps(body), ETH_LENGTH, title="tickers"))],
    )
    assert not any(record.value == 1111.11 for record in ledger._evidence)
    assert {record.symbol for record in ledger._evidence} == {"ETH-USDT", "BTC-USDT"}
    kept = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 2246.35 USDT.\nBTC-USDT (OKX) last 76025.6 USDT.",
            "2246.35 | observed | ETH-USDT last | page::last",
            "76025.6 | observed | BTC-USDT last | page::last",
        )
    )
    assert kept.valid is True, kept.issues
    unlabeled = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 1111.11 USDT.",
            "1111.11 | observed | ETH-USDT last | page::last",
        )
    )
    assert _reasons(unlabeled) == ["not_in_referenced_call"]
    crossed = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX) last 76025.6 USDT.",
            "76025.6 | observed | ETH-USDT last | page::last",
        )
    )
    assert _reasons(crossed) == ["not_in_referenced_call"]


def test_reader_json_scan_and_walk_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A late or very deep body cannot add prints or raise."""
    late = ("x" * 40) + _json_text("ETH-USDT", ETH_QUOTES)
    assert reader_evidence._json_documents(late)
    monkeypatch.setattr(reader_evidence, "_MAX_READER_BODY_CHARS", 24)
    assert reader_evidence._json_documents(late) == []

    node: Any = {"instId": "ETH-USDT", "last": "2246.35"}
    for _ in range(40):
        node = {"nest": node}
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Report the ETH-USDT ticker.")
    ledger.ingest_tool_result(
        tool_name="read_url",
        arguments={"url": ETH_MY_URL},
        result=json.dumps(
            {"status": "ok", "url": ETH_MY_URL, "content": node, "length": ETH_LENGTH}
        ),
        call_id="deep",
        success=True,
    )
    assert not any(record.value == 2246.35 for record in ledger._evidence)


# Source times from the 2026-11-07 acceptance window, labelled synthetic here.
# 1794055824067 ms == 2026-11-07T12:50:24.067Z == 2026-11-07T20:50:24.067+08:00.
ETH_TS = "1794055824067"
BTC_TS = "1794055824074"
ETH_UTC = "2026-11-07T12:50:24.067Z"
ETH_SGT = "2026-11-07T20:50:24.067+08:00"


def _acceptance_ticker(inst_id: str, quotes: dict[str, str], ts: str) -> dict[str, Any]:
    row = {"instId": inst_id, "ts": ts, "vol24h": VOLUME, **quotes}
    return {"code": "0", "data": [row]}


def test_source_ts_is_kept_for_its_own_call_and_symbol(tmp_path: Path) -> None:
    """A declared source instant is evidence. It is not a price and not portable."""
    eth_json = json.dumps(
        _acceptance_ticker(
            "ETH-USDT",
            {"last": "2241.08", "open24h": "2238.04", "high24h": "2254.25", "low24h": "2210.17"},
            ETH_TS,
        )
    )
    btc_json = json.dumps(
        _acceptance_ticker(
            "BTC-USDT",
            {"last": "75883.7", "open24h": "76072", "high24h": "76251.1", "low24h": "74887.9"},
            BTC_TS,
        )
    )
    ledger = _ingest(
        tmp_path,
        [
            ("eth", ETH_MY_URL, _envelope(ETH_MY_URL, eth_json, ETH_LENGTH, title="ETH-USDT")),
            ("btc", BTC_MY_URL, _envelope(BTC_MY_URL, btc_json, BTC_LENGTH, title="BTC-USDT")),
        ],
    )
    eth_ts = next(record for record in ledger._evidence if record.call_id == "eth" and record.field.endswith(".ts"))
    btc_ts = next(record for record in ledger._evidence if record.call_id == "btc" and record.field.endswith(".ts"))
    assert (eth_ts.value, eth_ts.symbol, eth_ts.currency, eth_ts.status) == (
        1794055824067, "ETH-USDT", None, "observed",
    )
    assert (btc_ts.value, btc_ts.symbol) == (1794055824074, "BTC-USDT")
    assert eth_ts.source == "okx" and btc_ts.source == "okx"
    assert eth_ts not in ledger._price_records()
    assert not any(record.value in {ETH_LENGTH, BTC_LENGTH} for record in ledger._evidence)
    quote = next(record for record in ledger._evidence if record.call_id == "eth" and record.field.endswith(".last"))
    assert quote.timestamp == ETH_TS
    assert quote.value == 2241.08
    assert quote.currency == "USDT"

    prose = (
        "ETH-USDT (OKX, USDT) last 2241.08, open 2238.04, high 2254.25, low 2210.17, "
        "source time 1794055824067.\n"
        "BTC-USDT (OKX, USDT) last 75883.7, open 76072, high 76251.1, low 74887.9, "
        "source time 1794055824074.\n"
        "ETH-USDT (OKX) high-low gap is 44.08 USDT."
    )
    rows = (
        "2241.08 | observed | ETH-USDT last | eth::last",
        "2238.04 | observed | ETH-USDT open | eth::open24h",
        "2254.25 | observed | ETH-USDT high | eth::high24h",
        "2210.17 | observed | ETH-USDT low | eth::low24h",
        "1794055824067 | observed | ETH-USDT ts | eth::ts",
        "75883.7 | observed | BTC-USDT last | btc::last",
        "76072 | observed | BTC-USDT open | btc::open24h",
        "76251.1 | observed | BTC-USDT high | btc::high24h",
        "74887.9 | observed | BTC-USDT low | btc::low24h",
        "1794055824074 | observed | BTC-USDT ts | btc::ts",
        "44.08 | derived | 2254.25 - 2210.17 | eth",
    )
    kept = ledger.validate_final_answer(_answer(prose, *rows))
    assert kept.valid is True, kept.issues

    grouped = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX, USDT) source time 1,794,055,824,067.",
            "1,794,055,824,067 | observed | ETH-USDT ts | eth::ts",
        )
    )
    assert grouped.valid is True, grouped.issues

    with_utc = ledger.validate_final_answer(
        _answer(
            f"ETH-USDT (OKX, USDT) source time 1794055824067 ({ETH_UTC}).",
            "1794055824067 | observed | ETH-USDT ts | eth::ts",
        )
    )
    assert with_utc.valid is True, with_utc.issues
    with_sgt = ledger.validate_final_answer(
        _answer(
            f"ETH-USDT (OKX, USDT) source time 1794055824067 ({ETH_SGT}).",
            "1794055824067 | observed | ETH-USDT ts | eth::ts",
        )
    )
    assert with_sgt.valid is True, with_sgt.issues

    wrong_clock = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX, USDT) source time 1794055824067 (2026-11-07T12:50:24.000Z).",
            "1794055824067 | observed | ETH-USDT ts | eth::ts",
        )
    )
    assert _reasons(wrong_clock) == ["timestamp_format_mismatch"]

    invented = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX, USDT) source time 1794055824999.",
            "1794055824999 | observed | ETH-USDT ts | eth::ts",
        )
    )
    swapped_call = ledger.validate_final_answer(
        _answer(
            "BTC-USDT (OKX, USDT) source time 1794055824074.",
            "1794055824074 | observed | BTC-USDT ts | eth::ts",
        )
    )
    attributed_to_eth = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX, USDT) source time 1794055824074.",
            "1794055824074 | observed | ETH-USDT ts | btc::ts",
        )
    )
    as_price = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX, USDT) last 1794055824067.",
            "1794055824067 | observed | ETH-USDT last | eth::last",
        )
    )
    length_as_ts = ledger.validate_final_answer(
        _answer(
            "ETH-USDT (OKX, USDT) source time 453.0.",
            "453.0 | observed | ETH-USDT ts | eth::ts",
        )
    )
    assert _reasons(invented) == ["not_in_referenced_call"]
    assert _reasons(swapped_call) == ["not_in_referenced_call"]
    assert _reasons(attributed_to_eth) == ["not_in_referenced_call"]
    assert "ETH-USDT" in attributed_to_eth.issues[0]["message"]
    assert _reasons(as_price) == ["not_in_referenced_call"]
    assert _reasons(length_as_ts) == ["not_in_referenced_call"]
    assert not any(record.value == 453 for record in ledger._evidence)


def test_a_calendar_rendering_without_the_ts_digits_is_not_checked() -> None:
    """A clock string is not a substitute for the source digits."""
    from src.agent.grounding.figures import parse_figures_block, scan_figures

    text = f"ETH-USDT source time {ETH_UTC}."
    measured = [figure for figure in scan_figures(text, parse_figures_block(text)) if figure.shape == "measured"]
    assert measured == []


def test_non_epoch_ts_string_is_not_evidence(tmp_path: Path) -> None:
    body = json.dumps({"data": [{"instId": "ETH-USDT", "ts": "453", "last": "2241.08"}]})
    ledger = _ingest(
        tmp_path,
        [("eth", ETH_MY_URL, _envelope(ETH_MY_URL, body, ETH_LENGTH, title="ETH-USDT"))],
    )
    assert not any(record.field.endswith(".ts") or record.value == 453 for record in ledger._evidence)
    assert any(record.value == 2241.08 for record in ledger._evidence)


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
        if "instId=ETH-USDT" in url:
            return _envelope(url, _markdown(url, "ETH-USDT", ETH_QUOTES), ETH_LENGTH, title="ETH-USDT")
        if "instId=BTC-USDT" in url:
            return _envelope(url, _markdown(url, "BTC-USDT", BTC_QUOTES), BTC_LENGTH, title="BTC-USDT")
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
            _Response(tool_calls=[_tool_call("eth", ETH_MY_URL)]),
            _Response(tool_calls=[_tool_call("btc", BTC_MY_URL)]),
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
            _Response(tool_calls=[_tool_call("eth", ETH_MY_URL)]),
            _Response(tool_calls=[_tool_call("btc", BTC_MY_URL)]),
            _Response(content=invented),
        ],
    )
    assert result["status"] == "success"
    assert result.get("degraded") is True
    assert "1999.01" not in result["content"]
    assert "2246.35" in result["content"]
    assert "76025.6" in result["content"]
