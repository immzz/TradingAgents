"""Batch TradingAgents research for decision-intelligence queue tasks.

This module is deliberately proposal-only.  It runs the existing analyst graph,
uses a separate structured synthesis call to fill the downstream scorecard, and
writes resumable research artifacts.  It never sizes or places an order.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.factory import create_llm_client


CONTRACT_VERSION = "decision_intelligence_v1"
DEFAULT_ANALYSTS = ("market", "social", "news", "fundamentals")
DEFAULT_MAX_TASKS = 25
DEFAULT_REPORT_CHARS = 18_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _text(value: object) -> str:
    return str(value or "").strip()


class EvidenceItem(BaseModel):
    evidence_id: str
    source_type: str
    title: str
    url: str
    published_at: str
    claim: str
    metric: str = ""
    current_value: float | None = None
    prior_value: float | None = None
    unit: str = ""
    direction: Literal["tailwind", "headwind", "neutral"]
    reliability_0_to_1: float = Field(ge=0, le=1)
    relevance_0_to_1: float = Field(ge=0, le=1)


class CausalChain(BaseModel):
    chain_id: str
    evidence_ids: list[str]
    operating_driver: str
    mechanism: str
    financial_metric: str
    impact_low_pct: float
    impact_base_pct: float
    impact_high_pct: float
    lag_days: int = Field(ge=0)
    confidence_0_to_1: float = Field(ge=0, le=1)
    falsifiers: list[str]


class Scenario(BaseModel):
    name: Literal["bear", "base", "bull"]
    probability_pct: float = Field(ge=0, le=100)
    revenue_growth_pct: float | None = None
    operating_margin_pct: float | None = None
    eps: float | None = None
    valuation_multiple: float | None = None
    implied_value: float | None = None
    return_pct: float | None = None
    assumptions: list[str]


class ScenarioModel(BaseModel):
    method: Literal[
        "earnings_multiple",
        "dcf",
        "sum_of_parts",
        "nav",
        "relative_value",
        "etf_total_return",
    ]
    forecast_horizon_days: int = Field(ge=1)
    forecast_horizon_unit: Literal["trading_sessions", "calendar_days"] = "trading_sessions"
    scenarios: list[Scenario]


class Sensitivity(BaseModel):
    variable: str
    low: float
    base: float
    high: float
    valuation_effect_low_pct: float
    valuation_effect_high_pct: float


class MonitoringTrigger(BaseModel):
    metric: str
    trigger: str
    threshold: float | None = None
    action: Literal["add", "hold", "reduce", "exit", "reassess"]
    evidence_source: str


class AnalystDecision(BaseModel):
    action: Literal["initiate", "add", "hold", "reduce", "exit", "avoid"]
    confidence_0_to_1: float = Field(ge=0, le=1)
    requested_position_scale_0_to_1: float = Field(ge=0, le=1)
    summary: str


class ScoreSet(BaseModel):
    business_quality: float = Field(ge=0, le=10)
    catalyst_strength: float = Field(ge=0, le=10)
    realization_path: float = Field(ge=0, le=10)
    fundamental_support: float = Field(ge=0, le=10)
    valuation_safety: float = Field(ge=0, le=10)
    sentiment_timing: float = Field(ge=0, le=10)
    risk_reward: float = Field(ge=0, le=10)
    tradingagents_confidence: float = Field(ge=0, le=10)


class StyleConfirmations(BaseModel):
    estimate_revision_support: bool
    valuation_disciplined: bool
    macro_or_cycle_support: bool
    inflection_catalyst: bool
    balance_sheet_runway: bool
    momentum_supportive_not_euphoric: bool
    media_crowding_review: bool


class MacroReadthrough(BaseModel):
    impact_matrix_readthrough: str
    global_event_readthrough: str
    supports_proposal: bool
    risk_vetoes: list[str]


class ResearchSynthesis(BaseModel):
    llm_decision: Literal["confirm", "watch", "veto"]
    thesis: str
    catalyst: str
    realization_path: str
    business_quality: str
    supply_demand: str
    fundamental_verdict: Literal["pass", "fail", "unknown"]
    valuation_verdict: Literal["cheap", "fair", "stretched", "extreme", "unknown"]
    sentiment_stage: Literal["early", "improving", "crowded", "euphoric", "broken", "washed_out", "unknown"]
    regime_fit: str
    macro_context: MacroReadthrough
    style_confirmations: StyleConfirmations
    media_crowding: str
    bull_case: str
    bear_case: str
    risk_vetoes: list[str]
    tradingagents_decision: str
    scores: ScoreSet
    evidence_ledger: list[EvidenceItem]
    causal_chains: list[CausalChain]
    scenario_model: ScenarioModel
    sensitivity: list[Sensitivity]
    disconfirming_evidence: list[str]
    monitoring_triggers: list[MonitoringTrigger]
    analyst_decision: AnalystDecision


def _task_decision(task: dict[str, Any]) -> str:
    value = _text(task.get("confirmation_decision")).lower()
    if value in {"confirm", "watch", "veto", "pending"}:
        return value
    review = task.get("confirmation_review") if isinstance(task.get("confirmation_review"), dict) else {}
    value = _text(review.get("decision")).lower()
    return value if value in {"confirm", "watch", "veto", "pending"} else "pending"


def _selected_tasks(
    queue: dict[str, Any],
    *,
    task_ids: set[str] | None,
    decisions: set[str],
    max_tasks: int | None,
) -> list[dict[str, Any]]:
    selected = []
    for task in queue.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = _text(task.get("task_id"))
        if not task_id or (task_ids is not None and task_id not in task_ids):
            continue
        if _task_decision(task) not in decisions:
            continue
        selected.append(task)
        if max_tasks is not None and max_tasks > 0 and len(selected) >= max_tasks:
            break
    return selected


def _reports_from_state(state: dict[str, Any], *, max_chars: int) -> dict[str, str]:
    fields = (
        "market_report",
        "sentiment_report",
        "news_report",
        "fundamentals_report",
        "investment_plan",
        "trader_investment_plan",
        "final_trade_decision",
    )
    reports = {}
    for key in fields:
        value = _text(state.get(key))
        reports[key] = value[:max_chars]
    for key, state_key in (("research_debate", "investment_debate_state"), ("risk_debate", "risk_debate_state")):
        value = state.get(state_key)
        reports[key] = _text((value or {}).get("history") if isinstance(value, dict) else value)[:max_chars]
    return reports


def build_synthesis_prompt(task: dict[str, Any], reports: dict[str, str], graph_decision: str) -> str:
    task_context = {
        key: task.get(key)
        for key in (
            "task_id",
            "as_of_date",
            "ticker",
            "strategy",
            "signal_kind",
            "mode",
            "decision",
            "target",
            "source_signal",
            "macro_overlay",
            "macro_impact",
            "current_snapshot",
            "news_context",
            "operating_gates",
            "required_questions",
        )
    }
    return f"""You are the final decision-intelligence synthesizer for a paper-only equity research workflow.

Analysis date: {task.get('as_of_date')}
Ticker: {task.get('ticker')}
Source strategy: {task.get('strategy')}

Use only facts present in TASK CONTEXT or TRADINGAGENTS REPORTS. Never invent a URL, filing, contract,
financial value, consensus estimate, or source date. A repeated/syndicated article is not independent
confirmation. Distinguish reported facts from your calculations and assumptions.

Build at least three atomic evidence rows from at least two independent origin domains, including at
least one primary company, regulatory, government, or authoritative industry source. Link at least two
causal chains to evidence IDs. Each chain must quantify low/base/high impact on an operating driver and
financial metric, include lag/confidence, and state falsifiers. Build exactly bear/base/bull scenarios;
probabilities must sum to 100. Set forecast_horizon_unit explicitly; use trading_sessions for the
standard 1-6 month equity horizon. Use current price from the task when computing implied values/returns.
Provide at least two sensitivity variables, one disconfirming fact, and two measurable monitoring
triggers. Set watch/veto and expose data gaps when the supplied evidence cannot support these fields.

The TradingAgents directional output is debate evidence only: {graph_decision}
The downstream deterministic validator will recompute returns, evidence quality, action, and position
scale. Do not assume that choosing confirm makes the card pass.

TASK CONTEXT:
{json.dumps(task_context, indent=2, ensure_ascii=False, default=str)}

TRADINGAGENTS REPORTS:
{json.dumps(reports, indent=2, ensure_ascii=False, default=str)}
"""


def _response_text(response: Any) -> str:
    """Extract text from a LangChain message without accepting tool payloads."""

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                parts.append(_text(item.get("text")))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _strict_json_object(text: str) -> dict[str, Any]:
    """Decode exactly one JSON object, tolerating only Markdown fence wrappers."""

    candidate = text.strip()
    if candidate.startswith("```") and candidate.endswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() in {"```", "```json"}:
            candidate = "\n".join(lines[1:-1]).strip()
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise TypeError(f"JSON synthesis root must be an object, got {type(value).__name__}")
    return value


def invoke_validated_output(structured_llm: Any, plain_llm: Any, prompt: str, schema: type[BaseModel]) -> BaseModel:
    """Invoke structured output, with a schema-validated JSON-text compatibility path.

    Some OpenAI OAuth/LangChain combinations return valid JSON in message content
    without populating the structured wrapper's ``parsed`` field.  The fallback is
    deliberately strict: it accepts one JSON object and validates the full Pydantic
    contract.  Free prose or partially shaped JSON still fails closed.
    """

    try:
        synthesized = structured_llm.invoke(prompt)
        if isinstance(synthesized, dict):
            synthesized = schema.model_validate(synthesized)
        if not isinstance(synthesized, schema):
            raise TypeError(f"structured synthesis returned {type(synthesized).__name__}")
        return synthesized
    except Exception as structured_error:  # noqa: BLE001 - compatibility retry is validated below
        json_schema = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        fallback_prompt = (
            f"{prompt}\n\n"
            "Return exactly one JSON object and no prose or Markdown. It must validate against this JSON Schema:\n"
            f"{json_schema}"
        )
        try:
            response = plain_llm.invoke(fallback_prompt)
            payload = _strict_json_object(_response_text(response))
            return schema.model_validate(payload)
        except Exception as fallback_error:  # noqa: BLE001 - retain both provider diagnostics
            raise RuntimeError(
                "structured synthesis failed and strict JSON fallback failed: "
                f"structured={type(structured_error).__name__}: {structured_error}; "
                f"fallback={type(fallback_error).__name__}: {fallback_error}"
            ) from fallback_error


def invoke_synthesis(structured_llm: Any, plain_llm: Any, prompt: str) -> ResearchSynthesis:
    """Validate a decision-intelligence synthesis against its complete schema."""

    return ResearchSynthesis.model_validate(
        invoke_validated_output(structured_llm, plain_llm, prompt, ResearchSynthesis)
    )


def merge_synthesis(task: dict[str, Any], synthesis: ResearchSynthesis) -> dict[str, Any]:
    card = deepcopy(task.get("scorecard_template") or {})
    output = synthesis.model_dump(mode="json")
    analysis = card.get("decision_analysis") if isinstance(card.get("decision_analysis"), dict) else {}
    snapshot = task.get("current_snapshot") if isinstance(task.get("current_snapshot"), dict) else {}
    analysis.update(
        {
            "contract_version": CONTRACT_VERSION,
            "analysis_as_of": task.get("as_of_date"),
            "current_price": snapshot.get("last_price"),
            "evidence_ledger": output.pop("evidence_ledger"),
            "causal_chains": output.pop("causal_chains"),
            "scenario_model": output.pop("scenario_model"),
            "sensitivity": output.pop("sensitivity"),
            "disconfirming_evidence": output.pop("disconfirming_evidence"),
            "monitoring_triggers": output.pop("monitoring_triggers"),
            "analyst_decision": output.pop("analyst_decision"),
        }
    )
    macro_update = output.pop("macro_context")
    style_update = output.pop("style_confirmations")
    card.update(output)
    card["analysis_contract_version"] = CONTRACT_VERSION
    card["llm_confirmation_task_id"] = task.get("task_id")
    card["symbol"] = task.get("ticker")
    card["macro_context"] = {**(card.get("macro_context") or {}), **macro_update}
    card["style_confirmations"] = {**(card.get("style_confirmations") or {}), **style_update}
    card["decision_analysis"] = analysis
    card["evidence"] = [
        {
            "title": row.get("title"),
            "url": row.get("url"),
            "date": row.get("published_at"),
            "notes": row.get("claim"),
            "evidence_id": row.get("evidence_id"),
        }
        for row in analysis.get("evidence_ledger") or []
        if isinstance(row, dict)
    ]
    return card


def _default_graph_factory(config: dict[str, Any], analysts: list[str], debug: bool) -> Callable[[], TradingAgentsGraph]:
    return lambda: TradingAgentsGraph(selected_analysts=analysts, config=config.copy(), debug=debug)


def _default_synthesis_llm(config: dict[str, Any]):
    kwargs = {
        "reasoning_effort": config.get("openai_reasoning_effort"),
        "thinking_level": config.get("google_thinking_level"),
        "effort": config.get("anthropic_effort"),
        "temperature": config.get("temperature"),
        "max_retries": config.get("llm_max_retries"),
    }
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    return create_llm_client(
        config["llm_provider"],
        config["deep_think_llm"],
        config.get("backend_url"),
        **kwargs,
    ).get_llm()


def run_queue(
    *,
    queue_path: str | Path,
    out_dir: str | Path,
    max_tasks: int | None = DEFAULT_MAX_TASKS,
    task_ids: list[str] | None = None,
    include_decisions: tuple[str, ...] = ("pending", "watch"),
    analysts: list[str] | None = None,
    config_overrides: dict[str, Any] | None = None,
    report_chars: int = DEFAULT_REPORT_CHARS,
    run_id: str | None = None,
    force: bool = False,
    debug: bool = False,
    graph_factory: Callable[[], Any] | None = None,
    synthesis_llm: Any | None = None,
    validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run batch research and write resumable filled-scorecard artifacts."""

    queue_file = Path(queue_path)
    queue = json.loads(queue_file.read_text(encoding="utf-8"))
    if not isinstance(queue, dict):
        raise ValueError("queue JSON root must be an object")
    queue_date = _text(queue.get("as_of_date"))
    if not queue_date:
        raise ValueError("queue is missing as_of_date")

    selected = _selected_tasks(
        queue,
        task_ids=set(task_ids) if task_ids else None,
        decisions={value.lower() for value in include_decisions},
        max_tasks=max_tasks,
    )
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    queue_hash = _hash(queue)

    config = DEFAULT_CONFIG.copy()
    if config_overrides:
        config.update({key: value for key, value in config_overrides.items() if value is not None})
    run_root = output_dir / "tradingagents_reports" / f"{queue_date}_{run_id}"
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
    if selected:
        synthesis_llm = synthesis_llm or _default_synthesis_llm(config)
        structured_llm = synthesis_llm.with_structured_output(ResearchSynthesis)

    signature = {
        "checkpoint_contract": "decision-intelligence-task.v1",
        "provider": config.get("llm_provider"),
        "quick_model": config.get("quick_think_llm"),
        "deep_model": config.get("deep_think_llm"),
        "analysts": analyst_list,
        "contract_version": CONTRACT_VERSION,
    }
    checkpoint_path = output_dir / "llm_confirmation_auto_checkpoint.json"
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.exists() and not force:
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                checkpoint = loaded.get("tasks") or {}
        except (OSError, json.JSONDecodeError):
            checkpoint = {}

    ticker_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    scorecard_by_task: dict[str, dict[str, Any]] = {}
    for task in queue.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = _text(task.get("task_id"))
        card = task.get("filled_scorecard")
        card_task_id = _text((card or {}).get("llm_confirmation_task_id")) if isinstance(card, dict) else ""
        if task_id and isinstance(card, dict) and card_task_id == task_id:
            scorecard_by_task[task_id] = deepcopy(card)
    preserved_scorecards = len(scorecard_by_task)
    for task in selected:
        task_id = _text(task.get("task_id"))
        ticker = _text(task.get("ticker")).upper()
        fingerprint = _hash({"task": task, "signature": signature})
        cached = checkpoint.get(fingerprint) if isinstance(checkpoint, dict) else None
        if isinstance(cached, dict) and cached.get("status") == "ok" and not force:
            reused = deepcopy(cached)
            reused["reused_from_checkpoint"] = True
            results.append(reused)
            if isinstance(reused.get("scorecard"), dict):
                scorecard_by_task[task_id] = reused["scorecard"]
            continue

        started = _now()
        try:
            if ticker not in ticker_cache:
                graph = graph_factory()
                state, graph_decision = graph.propagate(ticker, queue_date, asset_type="stock")
                report_dir = run_root / ticker
                report_path = graph.save_reports(state, ticker, save_path=report_dir)
                ticker_cache[ticker] = {
                    "state": state,
                    "decision": graph_decision,
                    "report_path": str(report_path),
                }
            research = ticker_cache[ticker]
            reports = _reports_from_state(research["state"], max_chars=report_chars)
            prompt = build_synthesis_prompt(task, reports, str(research["decision"]))
            if structured_llm is None:
                raise RuntimeError("structured synthesis LLM is unavailable")
            synthesized = invoke_synthesis(structured_llm, synthesis_llm, prompt)
            card = merge_synthesis(task, synthesized)
            validation = validator(card) if validator else {}
            result = {
                "task_id": task_id,
                "ticker": ticker,
                "strategy": task.get("strategy"),
                "status": "ok",
                "started_at": started,
                "finished_at": _now(),
                "task_fingerprint": fingerprint,
                "tradingagents_decision": research["decision"],
                "tradingagents_report_path": research["report_path"],
                "validation": validation,
                "scorecard": card,
                "reused_from_checkpoint": False,
            }
            scorecard_by_task[task_id] = card
        except Exception as exc:  # noqa: BLE001 - preserve partial batch progress
            result = {
                "task_id": task_id,
                "ticker": ticker,
                "strategy": task.get("strategy"),
                "status": "error",
                "started_at": started,
                "finished_at": _now(),
                "task_fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
                "reused_from_checkpoint": False,
            }
        results.append(result)
        checkpoint[fingerprint] = result
        _atomic_json(
            checkpoint_path,
            {"updated_at": _now(), "signature": signature, "tasks": checkpoint},
        )

    dated_run = output_dir / f"llm_confirmation_auto_analysis_{queue_date}_{run_id}.json"
    dated_scorecards = output_dir / f"llm_confirmation_auto_scorecards_{queue_date}_{run_id}.json"
    latest_run = output_dir / "llm_confirmation_auto_analysis_latest.json"
    latest_scorecards = output_dir / "llm_confirmation_auto_scorecards_latest.json"
    scorecards = list(scorecard_by_task.values())
    summary = {
        "tasks_selected": len(selected),
        "tasks_completed": sum(row.get("status") == "ok" for row in results),
        "tasks_failed": sum(row.get("status") == "error" for row in results),
        "tasks_reused": sum(bool(row.get("reused_from_checkpoint")) for row in results),
        "unique_tickers_analyzed": len(ticker_cache),
        "scorecards_written": len(scorecards),
        "scorecards_preserved": preserved_scorecards,
        "validation_ready": sum(bool((row.get("validation") or {}).get("ready")) for row in results),
    }
    payload = {
        "generated_at": _now(),
        "as_of_date": queue_date,
        "run_id": run_id,
        "source_queue_json": str(queue_file),
        "source_queue_sha256": queue_hash,
        "run_signature": signature,
        "summary": summary,
        "results": results,
        "artifacts": {
            "analysis_json": str(dated_run),
            "analysis_latest_json": str(latest_run),
            "scorecards_json": str(dated_scorecards),
            "scorecards_latest_json": str(latest_scorecards),
            "checkpoint_json": str(checkpoint_path),
            "report_root": str(run_root),
        },
        "note": "Automated research and scorecard synthesis only. No proposals were sized and no orders were placed.",
    }
    scorecard_payload = {
        "generated_at": payload["generated_at"],
        "as_of_date": queue_date,
        "source_queue_json": str(queue_file),
        "source_analysis_json": str(dated_run),
        "candidates": scorecards,
        "note": "Auto-filled research scorecards; deterministic review is still required before proposals.",
    }
    for path in (dated_run, latest_run):
        _atomic_json(path, payload)
    for path in (dated_scorecards, latest_scorecards):
        _atomic_json(path, scorecard_payload)
    return payload
