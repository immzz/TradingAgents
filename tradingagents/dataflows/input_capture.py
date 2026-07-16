"""Deterministic capture/replay boundary for confirmation research inputs.

Only tool outputs actually exposed to a confirmation graph are recorded.
Replay refuses unrecorded calls, so cached research cannot silently consume
fresh or different data.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator
from pathlib import Path


CAPTURE_CONTRACT = "research-tool-capture.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":")).encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def call_key(tool: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return _sha({"tool": tool, "args": args, "kwargs": kwargs})


@dataclass
class Session:
    mode: str
    records: list[dict[str, Any]] = field(default_factory=list)
    replay: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    used_keys: list[str] = field(default_factory=list)
    journal_path: Path | None = None


_SESSION: ContextVar[Session | None] = ContextVar("confirmation_input_capture", default=None)


@contextmanager
def capture_session(
    records: list[dict[str, Any]] | None = None,
    *,
    journal_path: str | Path | None = None,
) -> Iterator[Session]:
    journal = Path(journal_path) if journal_path else None
    journal_records: list[dict[str, Any]] = []
    if journal and journal.exists():
        journal_records = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
        validate_records(journal_records)
    if records is not None and journal_records:
        raise ValueError("research capture cannot combine explicit replay records with a capture journal")
    mode = "replay" if records is not None else "capture"
    replay: dict[str, list[dict[str, Any]]] = {}
    for record in (records if records is not None else journal_records):
        replay.setdefault(str(record["call_key"]), []).append(record)
    session = Session(mode=mode, records=list(journal_records), replay=replay, journal_path=journal)
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)


def execute(tool: str, args: tuple[Any, ...], kwargs: dict[str, Any], fn: Callable[[], Any]) -> Any:
    session = _SESSION.get()
    if session is None:
        return fn()
    key = call_key(tool, args, kwargs)
    if session.mode == "replay" or key in session.replay:
        matches = session.replay.get(key) or []
        if not matches:
            raise RuntimeError(f"research input replay rejected unrecorded tool call: {tool}:{key}")
        record = matches[0]
        session.used_keys.append(key)
        return record["result"]
    result = fn()
    record = {
        "schema_version": CAPTURE_CONTRACT,
        "captured_at": _now(),
        "tool": tool,
        "args": list(args),
        "kwargs": kwargs,
        "call_key": key,
        "result": result,
        "result_sha256": _sha(result),
        "input_category": (
            "structured_market"
            if tool in {"get_stock_data", "get_indicators", "get_verified_market_snapshot"}
            else "structured_fundamentals"
            if tool in {"get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement"}
            else "instrument_identity"
        ),
    }
    session.records.append(record)
    session.replay.setdefault(key, []).append(record)
    if session.journal_path:
        session.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with session.journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    return result


def validate_records(records: list[dict[str, Any]]) -> None:
    allowed = {
        "get_stock_data",
        "get_indicators",
        "get_verified_market_snapshot",
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "resolve_instrument_identity",
    }
    for index, record in enumerate(records):
        if record.get("schema_version") != CAPTURE_CONTRACT:
            raise ValueError(f"unsupported research capture record at index {index}")
        if record.get("tool") not in allowed:
            raise ValueError(f"research capture contains disallowed tool: {record.get('tool')}")
        args = tuple(record.get("args") or [])
        kwargs = record.get("kwargs") if isinstance(record.get("kwargs"), dict) else {}
        if record.get("call_key") != call_key(str(record.get("tool")), args, kwargs):
            raise ValueError(f"research capture call hash mismatch at index {index}")
        if record.get("result_sha256") != _sha(record.get("result")):
            raise ValueError(f"research capture result hash mismatch at index {index}")


def records_sha256(records: list[dict[str, Any]]) -> str:
    validate_records(records)
    stable = [{key: value for key, value in record.items() if key != "captured_at"} for record in records]
    return _sha(stable)
