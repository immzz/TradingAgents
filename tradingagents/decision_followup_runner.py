"""Automated follow-up research for decision-intelligence observations.

The runner reads an existing forward calibration ledger, researches only
missing business/financial observations, and writes source-linked observation
artifacts.  It never changes a scorecard, proposal, position, or order.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from tradingagents.decision_intelligence_runner import (
    DEFAULT_ANALYSTS,
    _atomic_json,
    _default_graph_factory,
    _default_synthesis_llm,
    _hash,
    _now,
    _reports_from_state,
    invoke_validated_output,
)
from tradingagents.default_config import DEFAULT_CONFIG


OBSERVATION_SCHEMA_VERSION = "decision_intelligence_observations_v1"
FOLLOWUP_SCHEMA_VERSION = "decision_intelligence_followup_v1"
DEFAULT_MAX_TASKS = 10
DEFAULT_REPORT_CHARS = 18_000


def _text(value: object) -> str:
    return str(value or "").strip()


def _num(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _date(value: object) -> date | None:
    try:
        return date.fromisoformat(_text(value)[:10])
    except ValueError:
        return None


def _metric(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _text(value).lower()).strip("_")


class SourcedObservation(BaseModel):
    task_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    kind: Literal["monitoring_trigger", "causal_chain_outcome"]
    status: Literal["observed"] = "observed"
    metric: str = Field(min_length=1)
    value: float | None = None
    unit: str = ""
    trigger_fired: bool | None = None
    chain_id: str = ""
    actual_impact_pct: float | None = None
    observed_at: str = Field(min_length=10)
    published_at: str = Field(min_length=10)
    source_url: str = Field(min_length=8)
    source_type: str = Field(min_length=1)
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_measurement(self):
        if self.kind == "monitoring_trigger" and self.value is None:
            raise ValueError("monitoring observation requires numeric value")
        if self.kind == "causal_chain_outcome" and (not self.chain_id or self.actual_impact_pct is None):
            raise ValueError("causal-chain observation requires chain_id and actual_impact_pct")
        if not self.source_url.startswith(("https://", "http://")):
            raise ValueError("source_url must be HTTP(S)")
        return self


class UnresolvedObservation(BaseModel):
    task_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    kind: Literal["monitoring_trigger", "causal_chain_outcome"]
    metric: str = Field(min_length=1)
    chain_id: str = ""
    reason: str = Field(min_length=1)
    next_source_to_check: str = Field(min_length=1)


class FollowupSynthesis(BaseModel):
    observations: list[SourcedObservation]
    unresolved: list[UnresolvedObservation]
    summary: str


def followup_tasks(calibration: dict[str, Any], *, max_tasks: int | None = DEFAULT_MAX_TASKS) -> list[dict[str, Any]]:
    tasks = []
    for row in calibration.get("rows") or []:
        if not isinstance(row, dict) or row.get("state") not in {"open", "matured"}:
            continue
        verification = row.get("business_verification") if isinstance(row.get("business_verification"), dict) else {}
        monitoring = [item for item in verification.get("monitoring") or [] if isinstance(item, dict) and not item.get("observed")]
        chains = [item for item in verification.get("causal_chains") or [] if isinstance(item, dict) and not item.get("observed")]
        if not monitoring and not chains:
            continue
        tasks.append(
            {
                "task_id": row.get("task_id"),
                "ticker": row.get("ticker"),
                "strategy": row.get("strategy"),
                "prediction_date": row.get("prediction_date"),
                "review_decision": row.get("review_decision"),
                "derived_action": row.get("derived_action"),
                "monitoring": monitoring,
                "causal_chains": chains,
                "source_review_json": row.get("source_review_json"),
            }
        )
        if max_tasks is not None and max_tasks > 0 and len(tasks) >= max_tasks:
            break
    return tasks


def build_followup_prompt(task: dict[str, Any], reports: dict[str, str], as_of_date: str) -> str:
    return f"""You are verifying a prior paper-only equity forecast with later evidence.

Research cutoff: {as_of_date}
Ticker: {task.get('ticker')}
Prior prediction date: {task.get('prediction_date')}
Task ID: {task.get('task_id')}

Use only facts explicitly present in TRADINGAGENTS REPORTS. Do not invent a URL, publication date,
metric, number, or causal impact. An observation is valid only when the report provides an HTTP(S)
source URL, publication date, numeric value, unit, and enough comparison context to explain it.
The observation date must be after the prior prediction date and no later than the research cutoff.

For monitoring_trigger rows, return only requested metrics and a numeric value. For
causal_chain_outcome rows, return only requested chain IDs and quantify actual_impact_pct only when
the supplied report provides enough prior/current financial data to calculate it. Put everything
else in unresolved with the exact missing fact and next primary source to check. Returning zero
observations is correct when evidence is insufficient.

REQUESTED FOLLOW-UP:
{json.dumps(task, indent=2, ensure_ascii=False, default=str)}

TRADINGAGENTS REPORTS:
{json.dumps(reports, indent=2, ensure_ascii=False, default=str)}
"""


def _validated_observations(
    synthesis: FollowupSynthesis,
    task: dict[str, Any],
    *,
    as_of_date: str,
    report_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected_metrics = {_metric(row.get("metric")) for row in task.get("monitoring") or []}
    expected_chains = {str(row.get("chain_id") or "") for row in task.get("causal_chains") or []}
    prediction_date = _date(task.get("prediction_date"))
    cutoff = _date(as_of_date)
    valid = []
    rejected = []
    seen: set[tuple[str, str]] = set()
    for observation in synthesis.observations:
        row = observation.model_dump(mode="json")
        reason = ""
        if row["task_id"] != task.get("task_id") or row["ticker"].upper() != _text(task.get("ticker")).upper():
            reason = "task/ticker identity mismatch"
        elif row["kind"] == "monitoring_trigger" and _metric(row["metric"]) not in expected_metrics:
            reason = "metric was not requested"
        elif row["kind"] == "causal_chain_outcome" and row["chain_id"] not in expected_chains:
            reason = "chain_id was not requested"
        observed = _date(row["observed_at"])
        published = _date(row["published_at"])
        if not reason and (
            observed is None
            or published is None
            or prediction_date is None
            or cutoff is None
            or observed <= prediction_date
            or observed > cutoff
            or published > observed
        ):
            reason = "invalid point-in-time chronology"
        if not reason and row["source_url"] not in report_text:
            reason = "source URL is not present in TradingAgents reports"
        if not reason and row["published_at"][:10] not in report_text:
            reason = "publication date is not present in TradingAgents reports"
        key = (row["kind"], row["chain_id"] or _metric(row["metric"]))
        if not reason and key in seen:
            reason = "duplicate observation"
        if reason:
            rejected.append({**row, "rejection_reason": reason})
        else:
            seen.add(key)
            valid.append(row)
    return valid, rejected


def run_followup(
    *,
    calibration_path: str | Path,
    out_dir: str | Path,
    as_of_date: str | None = None,
    max_tasks: int | None = DEFAULT_MAX_TASKS,
    analysts: list[str] | None = None,
    config_overrides: dict[str, Any] | None = None,
    report_chars: int = DEFAULT_REPORT_CHARS,
    run_id: str | None = None,
    force: bool = False,
    debug: bool = False,
    graph_factory: Callable[[], Any] | None = None,
    synthesis_llm: Any | None = None,
) -> dict[str, Any]:
    calibration_file = Path(calibration_path)
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict):
        raise ValueError("calibration JSON root must be an object")
    cutoff = as_of_date or date.today().isoformat()
    tasks = followup_tasks(calibration, max_tasks=max_tasks)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    config = DEFAULT_CONFIG.copy()
    if config_overrides:
        config.update({key: value for key, value in config_overrides.items() if value is not None})
    run_root = output_dir / "tradingagents_followup_reports" / f"{cutoff}_{run_id}"
    config.update(
        {
            "results_dir": str(run_root / "logs"),
            "data_cache_dir": str(run_root / "cache"),
            "memory_log_path": str(run_root / "memory" / "trading_memory.md"),
        }
    )
    analyst_list = list(analysts or DEFAULT_ANALYSTS)
    graph_factory = graph_factory or _default_graph_factory(config, analyst_list, debug)
    structured_llm = None
    if tasks:
        synthesis_llm = synthesis_llm or _default_synthesis_llm(config)
        structured_llm = synthesis_llm.with_structured_output(FollowupSynthesis)

    signature = {
        "source_calibration_sha256": _hash(calibration),
        "as_of_date": cutoff,
        "provider": config.get("llm_provider"),
        "quick_model": config.get("quick_think_llm"),
        "deep_model": config.get("deep_think_llm"),
        "analysts": analyst_list,
        "schema_version": FOLLOWUP_SCHEMA_VERSION,
    }
    checkpoint_path = output_dir / "decision_intelligence_followup_checkpoint.json"
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.exists() and not force:
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            checkpoint = loaded.get("tasks") if isinstance(loaded, dict) and isinstance(loaded.get("tasks"), dict) else {}
        except (OSError, json.JSONDecodeError):
            checkpoint = {}

    ticker_cache: dict[str, dict[str, Any]] = {}
    results = []
    observations = []
    unresolved = []
    for task in tasks:
        task_id = _text(task.get("task_id"))
        ticker = _text(task.get("ticker")).upper()
        fingerprint = _hash({"task": task, "signature": signature})
        cached = checkpoint.get(fingerprint)
        if isinstance(cached, dict) and cached.get("status") == "ok" and not force:
            result = dict(cached)
            result["reused_from_checkpoint"] = True
            results.append(result)
            observations.extend(result.get("observations") or [])
            unresolved.extend(result.get("unresolved") or [])
            continue
        started = _now()
        try:
            if ticker not in ticker_cache:
                graph = graph_factory()
                state, graph_decision = graph.propagate(ticker, cutoff, asset_type="stock")
                report_path = graph.save_reports(state, ticker, save_path=run_root / ticker)
                ticker_cache[ticker] = {"state": state, "decision": graph_decision, "report_path": str(report_path)}
            research = ticker_cache[ticker]
            reports = _reports_from_state(research["state"], max_chars=report_chars)
            prompt = build_followup_prompt(task, reports, cutoff)
            if structured_llm is None:
                raise RuntimeError("structured follow-up LLM is unavailable")
            synthesis = FollowupSynthesis.model_validate(
                invoke_validated_output(structured_llm, synthesis_llm, prompt, FollowupSynthesis)
            )
            valid, rejected = _validated_observations(
                synthesis,
                task,
                as_of_date=cutoff,
                report_text="\n".join(reports.values()),
            )
            task_unresolved = [row.model_dump(mode="json") for row in synthesis.unresolved]
            task_unresolved.extend(
                {
                    "task_id": task_id,
                    "ticker": ticker,
                    "kind": row.get("kind"),
                    "metric": row.get("metric"),
                    "chain_id": row.get("chain_id"),
                    "reason": f"model observation rejected: {row.get('rejection_reason')}",
                    "next_source_to_check": "primary company filing or earnings release",
                }
                for row in rejected
            )
            result = {
                "task_id": task_id,
                "ticker": ticker,
                "status": "ok",
                "started_at": started,
                "finished_at": _now(),
                "task_fingerprint": fingerprint,
                "tradingagents_report_path": research["report_path"],
                "observations": valid,
                "unresolved": task_unresolved,
                "summary": synthesis.summary,
                "reused_from_checkpoint": False,
            }
            observations.extend(valid)
            unresolved.extend(task_unresolved)
        except Exception as exc:  # noqa: BLE001 - preserve partial follow-up progress
            result = {
                "task_id": task_id,
                "ticker": ticker,
                "status": "error",
                "started_at": started,
                "finished_at": _now(),
                "task_fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
                "observations": [],
                "unresolved": [],
                "reused_from_checkpoint": False,
            }
        results.append(result)
        checkpoint[fingerprint] = result
        _atomic_json(checkpoint_path, {"updated_at": _now(), "signature": signature, "tasks": checkpoint})

    analysis_json = output_dir / f"decision_intelligence_followup_{cutoff}_{run_id}.json"
    analysis_latest = output_dir / "decision_intelligence_followup_latest.json"
    observation_json = output_dir / f"decision_intelligence_observations_{cutoff}_{run_id}.json"
    observation_latest = output_dir / "decision_intelligence_observations_latest.json"
    summary = {
        "tasks_selected": len(tasks),
        "tasks_completed": sum(row.get("status") == "ok" for row in results),
        "tasks_failed": sum(row.get("status") == "error" for row in results),
        "tasks_reused": sum(bool(row.get("reused_from_checkpoint")) for row in results),
        "unique_tickers_analyzed": len(ticker_cache),
        "observations_written": len(observations),
        "unresolved": len(unresolved),
    }
    payload = {
        "schema_version": FOLLOWUP_SCHEMA_VERSION,
        "generated_at": _now(),
        "as_of_date": cutoff,
        "run_id": run_id,
        "source_calibration_json": str(calibration_file),
        "source_calibration_sha256": signature["source_calibration_sha256"],
        "summary": summary,
        "results": results,
        "unresolved": unresolved,
        "artifacts": {
            "followup_analysis_json": str(analysis_json),
            "followup_analysis_latest_json": str(analysis_latest),
            "observations_json": str(observation_json),
            "observations_latest_json": str(observation_latest),
            "checkpoint_json": str(checkpoint_path),
            "report_root": str(run_root),
        },
        "note": "Follow-up research only. No scorecard, threshold, proposal, position, or order was changed.",
    }
    observation_payload = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "generated_at": payload["generated_at"],
        "as_of_date": cutoff,
        "source_followup_analysis_json": str(analysis_json),
        "observations": observations,
        "unresolved": unresolved,
        "note": "Only source-linked numeric observations are calibration evidence.",
    }
    for path in (analysis_json, analysis_latest):
        _atomic_json(path, payload)
    for path in (observation_json, observation_latest):
        _atomic_json(path, observation_payload)
    return payload
