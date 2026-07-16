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
from tradingagents.dataflows.input_capture import capture_session, records_sha256, validate_records
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.factory import create_llm_client


CONTRACT_VERSION = "decision_intelligence_v1"
CHECKPOINT_CONTRACT = "decision-intelligence-task.v2"
SCORECARD_ARTIFACT_CONTRACT = "llm-confirmation-scorecards.v2"
OPERATION_CONTRACT = "confirmation-batch-operation.v1"
INPUT_POLICY_VERSION = "structured-confirmation-inputs.v2"
RESEARCH_CONTRACT_VERSION = "decision-intelligence-research.v2"
SYNTHESIS_CONTRACT_VERSION = "decision-intelligence-synthesis.v3"
RESEARCH_INPUT_CONTRACT = "research-input-manifest.v1"
ALLOWED_ANALYSTS = ("market", "fundamentals")
DEFAULT_ANALYSTS = ALLOWED_ANALYSTS
DEFAULT_MAX_TASKS = 25
DEFAULT_REPORT_CHARS = 18_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hash(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def _immutable_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"immutable research artifact differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def _task_fingerprint(task: dict[str, Any]) -> str:
    volatile = {"task_id", "safety_note", "decision_placeholder", "filled_scorecard"}
    normalized = {key: value for key, value in task.items() if key not in volatile}
    return _hash(
        {
            "task_contract": "confirmation-task-fingerprint.v1",
            "task": normalized,
            "scorecard_contract": (task.get("scorecard_template") or {}).get("analysis_contract_version")
            or task.get("analysis_contract_version"),
        }
    )


def _build_research_input_manifest(
    *,
    ticker: str,
    as_of_date: str,
    records: list[dict[str, Any]],
    signature: dict[str, Any],
    graph_config_sha256: str,
) -> dict[str, Any]:
    validate_records(records)
    stable_records = [{key: value for key, value in row.items() if key != "captured_at"} for row in records]
    stable = {
        "schema_version": RESEARCH_INPUT_CONTRACT,
        "ticker": ticker,
        "asset_type": "stock",
        "as_of_date": as_of_date,
        "tool_capture_contract": "research-tool-capture.v1",
        "tool_capture_sha256": records_sha256(records),
        "tool_records": stable_records,
        "input_categories": sorted({str(row.get("input_category")) for row in records}),
        "analyst_tool_allowlist": {
            "analysts": signature["analysts"],
            "tools": sorted({str(row.get("tool")) for row in records}),
            "raw_news_prose": False,
            "raw_social_prose": False,
        },
        "provider": signature.get("provider"),
        "quick_model": signature.get("quick_model"),
        "deep_model": signature.get("deep_model"),
        "graph_config_sha256": graph_config_sha256,
        "research_contract_version": RESEARCH_CONTRACT_VERSION,
        "safety_mode": "research_only_no_orders",
    }
    content_sha = _hash(stable)
    return {**stable, "generated_at": _now(), "content_sha256": content_sha}


def _research_fingerprint(manifest: dict[str, Any], signature: dict[str, Any]) -> str:
    return _hash(
        {
            "ticker": manifest["ticker"],
            "asset_type": manifest["asset_type"],
            "as_of_date": manifest["as_of_date"],
            "research_input_manifest_sha256": manifest["content_sha256"],
            "graph_config_sha256": manifest["graph_config_sha256"],
            "provider": signature.get("provider"),
            "quick_model": signature.get("quick_model"),
            "deep_model": signature.get("deep_model"),
            "analysts": signature["analysts"],
            "research_contract_version": RESEARCH_CONTRACT_VERSION,
        }
    )


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


def _reports_from_state(
    state: dict[str, Any], *, max_chars: int, include_unstructured_news: bool = True
) -> dict[str, str]:
    fields = [
        "market_report",
        "fundamentals_report",
        "investment_plan",
        "trader_investment_plan",
        "final_trade_decision",
    ]
    if include_unstructured_news:
        fields[1:1] = ["sentiment_report", "news_report"]
    reports = {}
    for key in fields:
        value = _text(state.get(key))
        reports[key] = value[:max_chars]
    for key, state_key in (("research_debate", "investment_debate_state"), ("risk_debate", "risk_debate_state")):
        value = state.get(state_key)
        reports[key] = _text((value or {}).get("history") if isinstance(value, dict) else value)[:max_chars]
    return reports


def _exact_json(path_value: object, sha_value: object, *, label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(path_value or ""))
    if not str(path_value or "").strip() or not path.is_file() or "latest" in path.name:
        raise ValueError(f"canonical {label} path is missing, mutable, or unreadable")
    if _file_hash(path) != str(sha_value or ""):
        raise ValueError(f"canonical {label} hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"canonical {label} root must be an object")
    return path, payload


def _validated_codex_news_context(task: dict[str, Any]) -> dict[str, Any]:
    """Expose only current-Codex interpreted structure, never raw source prose."""

    lineage = task.get("canonical_news_lineage")
    if not isinstance(lineage, dict) or not lineage.get("analysis_path"):
        return {"status": "unavailable", "reason": "canonical_news_lineage_missing"}
    analysis_path, analysis = _exact_json(
        lineage.get("analysis_path"), lineage.get("analysis_sha256"), label="news analysis"
    )
    operation_path, operation = _exact_json(
        lineage.get("operation_path"), lineage.get("operation_sha256"), label="news operation"
    )
    bundle_path, _bundle = _exact_json(
        lineage.get("bundle_path"), lineage.get("bundle_sha256"), label="news bundle"
    )
    authorship = analysis.get("authorship") if isinstance(analysis.get("authorship"), dict) else {}
    operation_lineage = operation.get("lineage") if isinstance(operation.get("lineage"), dict) else {}
    if not (
        analysis.get("schema_version") == "news_analysis.v3"
        and analysis.get("contract_version") == "codex-news-impact.v3"
        and analysis.get("provider") == "codex_automation"
        and authorship.get("mode") == "direct_skill_execution"
        and analysis.get("analysis_ready") is True
        and analysis.get("decision_ready") is True
        and analysis.get("status") == "ready"
        and str(analysis.get("trading_date") or "") == str(task.get("as_of_date") or "")
        and operation.get("schema_version") == "news_research_operation.v2"
        and operation.get("status") == "ok"
        and operation_lineage.get("analysis_sha256") == lineage.get("analysis_sha256")
        and operation_lineage.get("bundle_sha256") == lineage.get("bundle_sha256")
    ):
        raise ValueError("canonical Codex news analysis or final operation is not decision-ready")

    ticker = str(task.get("ticker") or "").upper()
    events = [
        row
        for row in analysis.get("events") or []
        if isinstance(row, dict) and ticker in {str(value).upper() for value in row.get("affected_tickers") or []}
    ]
    impacts = [
        row
        for row in analysis.get("ticker_impacts") or []
        if isinstance(row, dict) and str(row.get("ticker") or "").upper() == ticker
    ]
    article_ids = {
        str(article_id)
        for event in events
        for article_id in event.get("source_article_ids") or []
        if article_id
    }
    sources = []
    for row in analysis.get("source_articles") or []:
        if not isinstance(row, dict) or str(row.get("article_id") or "") not in article_ids:
            continue
        sources.append(
            {
                "article_id": row.get("article_id"),
                "url": row.get("url"),
                "published_at": row.get("published_at"),
                "observed_at": row.get("observed_at"),
                "available_at": row.get("available_at"),
                "retrieved_at": row.get("retrieved_at"),
                "origin_domain": row.get("origin_domain"),
                "origin_category": row.get("origin_category"),
                "quality_tier": row.get("quality_tier"),
                "provider_families": row.get("provider_families") or [],
            }
        )
    missing = sorted(article_ids - {str(row.get("article_id")) for row in sources})
    if missing:
        raise ValueError(f"canonical news events reference missing source article IDs: {','.join(missing)}")
    safe_events = []
    allowed_event_keys = {
        "event_id", "event_type", "affected_tickers", "analysis_layers", "direction", "confidence",
        "evidence_quality", "expected_price_impact_pct", "horizon_days", "sentiment_score",
        "sentiment_stage", "crowding_risk", "source_article_ids", "estimate_label", "title_zh",
        "summary_zh", "causal_chain_zh", "monitoring_triggers_zh", "risks_zh", "scenarios",
        "sentiment_rationale_zh",
    }
    for event in events:
        safe_events.append({key: event.get(key) for key in allowed_event_keys if key in event})
    context = {
        "status": "ready",
        "ticker": ticker,
        "analysis_contract_version": analysis.get("contract_version"),
        "analysis_model": analysis.get("model"),
        "authorship": {"provider": analysis.get("provider"), "mode": authorship.get("mode")},
        "analysis_path": str(analysis_path),
        "analysis_sha256": lineage.get("analysis_sha256"),
        "operation_path": str(operation_path),
        "operation_sha256": lineage.get("operation_sha256"),
        "bundle_path": str(bundle_path),
        "bundle_sha256": lineage.get("bundle_sha256"),
        "ticker_impacts": impacts,
        "events": safe_events,
        "source_metadata": sources,
        "raw_headlines_exposed": False,
        "raw_article_bodies_exposed": False,
        "raw_social_prose_exposed": False,
    }
    return {**context, "content_sha256": _hash(context)}


def build_synthesis_prompt(
    task: dict[str, Any],
    reports: dict[str, str],
    graph_decision: str,
    codex_news_context: dict[str, Any] | None = None,
) -> str:
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
            "operating_gates",
            "required_questions",
        )
    }
    allowed_reports = {
        key: reports.get(key)
        for key in (
            "market_report",
            "fundamentals_report",
            "investment_plan",
            "trader_investment_plan",
            "final_trade_decision",
            "research_debate",
            "risk_debate",
        )
        if reports.get(key)
    }
    return f"""You are the final decision-intelligence synthesizer for a paper-only equity research workflow.

Analysis date: {task.get('as_of_date')}
Ticker: {task.get('ticker')}
Source strategy: {task.get('strategy')}

Use only facts present in TASK CONTEXT or TRADINGAGENTS REPORTS. Never invent a URL, filing, contract,
financial value, consensus estimate, or source date. A repeated/syndicated article is not independent
confirmation. Distinguish reported facts from your calculations and assumptions.

Raw news headlines, article bodies, and social posts are intentionally excluded. Do not infer event
meaning, narrative impact, crowding, or sentiment from raw prose. Use only structured current-Codex
news fields already present in the task context; when they are absent, mark the corresponding fields
unknown and expose the evidence gap.

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
{json.dumps(allowed_reports, indent=2, ensure_ascii=False, default=str)}

VALIDATED CURRENT-CODEX NEWS CONTEXT (already interpreted; source prose excluded):
{json.dumps(codex_news_context or {"status": "unavailable"}, indent=2, ensure_ascii=False, default=str)}
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
    input_manifest_root: str | Path | None = None,
    research_root: str | Path | None = None,
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
    queue_hash = _file_hash(queue_file)

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
    unsupported_analysts = sorted(set(analyst_list) - set(ALLOWED_ANALYSTS))
    if unsupported_analysts:
        raise ValueError(
            "confirmation analysts may not consume raw news/social prose; "
            f"unsupported analysts: {', '.join(unsupported_analysts)}"
        )
    capture_required = graph_factory is None
    graph_factory = graph_factory or _default_graph_factory(config, analyst_list, debug)
    structured_llm = None
    if selected:
        synthesis_llm = synthesis_llm or _default_synthesis_llm(config)
        structured_llm = synthesis_llm.with_structured_output(ResearchSynthesis)

    signature = {
        "checkpoint_contract": CHECKPOINT_CONTRACT,
        "provider": config.get("llm_provider"),
        "quick_model": config.get("quick_think_llm"),
        "deep_model": config.get("deep_think_llm"),
        "analysts": analyst_list,
        "contract_version": CONTRACT_VERSION,
        "input_policy_version": INPUT_POLICY_VERSION,
        "research_contract_version": RESEARCH_CONTRACT_VERSION,
        "synthesis_contract_version": SYNTHESIS_CONTRACT_VERSION,
    }
    graph_config_sha256 = _hash(
        {
            key: config.get(key)
            for key in (
                "data_vendors",
                "tool_vendors",
                "max_debate_rounds",
                "max_risk_discuss_rounds",
                "output_language",
                "benchmark_ticker",
            )
        }
    )
    completed_analysis = output_dir / f"llm_confirmation_auto_analysis_{queue_date}_{run_id}.json"
    completed_operation = output_dir / f"llm_confirmation_auto_operation_{queue_date}_{run_id}.json"
    if completed_analysis.exists() or completed_operation.exists():
        if not (completed_analysis.exists() and completed_operation.exists()):
            raise ValueError("completed confirmation run has an incomplete immutable operation pair")
        existing_payload = json.loads(completed_analysis.read_text(encoding="utf-8"))
        existing_operation = json.loads(completed_operation.read_text(encoding="utf-8"))
        if (
            existing_payload.get("source_queue_sha256") != queue_hash
            or existing_payload.get("run_signature") != signature
            or existing_operation.get("source_analysis_sha256") != _file_hash(completed_analysis)
        ):
            raise ValueError("same run_id cannot be reused with different queue bytes or run signature")
        return existing_payload
    input_root = Path(input_manifest_root) if input_manifest_root else output_dir.parent / "llm_confirmation_inputs"
    persistent_research_root = Path(research_root) if research_root else output_dir.parent / "llm_confirmation_research"
    checkpoint_path = output_dir / "llm_confirmation_auto_checkpoint_v2.json"
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.exists() and not force:
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and loaded.get("signature") == signature:
                checkpoint = loaded.get("tasks") or {}
        except (OSError, json.JSONDecodeError):
            checkpoint = {}

    research_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    scorecard_by_task: dict[str, dict[str, Any]] = {}
    research_fingerprints: set[str] = set()
    graph_executions = 0
    research_resumes = 0
    for task in selected:
        task_id = _text(task.get("task_id"))
        ticker = _text(task.get("ticker")).upper()
        fingerprint = _task_fingerprint(task)
        planned_key = _hash(
            {"ticker": ticker, "as_of_date": queue_date, "signature": signature, "run_id": run_id}
        )
        started = _now()
        research_fingerprint = "unavailable"
        synthesis_fingerprint = "unavailable"
        try:
            if planned_key not in research_cache:
                resume_path = run_root / ticker / "research_result.json"
                if resume_path.exists() and not force:
                    research = json.loads(resume_path.read_text(encoding="utf-8"))
                    manifest_path = Path(str(research.get("research_input_manifest_path") or ""))
                    if (
                        research.get("schema_version") != "decision-intelligence-research-result.v2"
                        or not manifest_path.exists()
                        or research.get("research_input_manifest_sha256") != _file_hash(manifest_path)
                        or research.get("run_signature") != signature
                    ):
                        raise ValueError("same-run research resume artifact failed signature or input-manifest validation")
                    research_resumes += 1
                else:
                    graph = graph_factory()
                    report_dir = run_root / ticker
                    capture_journal = report_dir / "research_input_capture.jsonl"
                    with capture_session(journal_path=capture_journal) as captured:
                        state, graph_decision = graph.propagate(ticker, queue_date, asset_type="stock")
                    if capture_required and not captured.records:
                        raise ValueError("confirmation graph produced no captured structured tool inputs")
                    report_path = graph.save_reports(state, ticker, save_path=report_dir)
                    manifest = _build_research_input_manifest(
                        ticker=ticker,
                        as_of_date=queue_date,
                        records=captured.records,
                        signature=signature,
                        graph_config_sha256=graph_config_sha256,
                    )
                    manifest_path = input_root / queue_date / f"{manifest['content_sha256']}.json"
                    if manifest_path.exists():
                        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        if existing_manifest.get("content_sha256") != manifest["content_sha256"]:
                            raise ValueError("content-addressed research input manifest collision")
                        manifest = existing_manifest
                    else:
                        _immutable_json(manifest_path, manifest)
                    research_fingerprint = _research_fingerprint(manifest, signature)
                    research = {
                        "schema_version": "decision-intelligence-research-result.v2",
                        "generated_at": _now(),
                        "run_id": run_id,
                        "ticker": ticker,
                        "as_of_date": queue_date,
                        "research_fingerprint": research_fingerprint,
                        "research_input_manifest_path": str(manifest_path),
                        "research_input_manifest_sha256": _file_hash(manifest_path),
                        "run_signature": signature,
                        "tradingagents_decision": graph_decision,
                        "tradingagents_report_path": str(report_path),
                        "reports": _reports_from_state(
                            state,
                            max_chars=report_chars,
                            include_unstructured_news=False,
                        ),
                        "capture_journal_path": str(capture_journal),
                        "capture_journal_sha256": _file_hash(capture_journal) if capture_journal.exists() else None,
                    }
                    persistent_path = persistent_research_root / queue_date / research_fingerprint / "research.json"
                    if persistent_path.exists():
                        existing_research = json.loads(persistent_path.read_text(encoding="utf-8"))
                        if (
                            existing_research.get("research_input_manifest_sha256")
                            != research["research_input_manifest_sha256"]
                            or existing_research.get("run_signature") != signature
                        ):
                            raise ValueError("content-addressed research result collision")
                    else:
                        _immutable_json(persistent_path, research)
                    _immutable_json(resume_path, research)
                    graph_executions += 1
                research_cache[planned_key] = research
            research = research_cache[planned_key]
            research_fingerprint = str(research["research_fingerprint"])
            research_fingerprints.add(research_fingerprint)
            codex_news_context = _validated_codex_news_context(task)
            codex_news_context_sha256 = str(
                codex_news_context.get("content_sha256") or _hash(codex_news_context)
            )
            synthesis_fingerprint = _hash(
                {
                    "research_fingerprint": research_fingerprint,
                    "task_fingerprint": fingerprint,
                    "synthesis_model": signature.get("deep_model"),
                    "synthesis_contract_version": SYNTHESIS_CONTRACT_VERSION,
                    "codex_news_context_sha256": codex_news_context_sha256,
                }
            )
            cached = checkpoint.get(fingerprint) if isinstance(checkpoint, dict) else None
            cached_validation = cached.get("validation") if isinstance(cached, dict) else None
            if (
                isinstance(cached, dict)
                and cached.get("status") == "ok"
                and isinstance(cached_validation, dict)
                and cached_validation.get("ready") is True
                and isinstance(cached.get("scorecard"), dict)
                and cached.get("research_fingerprint") == research_fingerprint
                and cached.get("synthesis_fingerprint") == synthesis_fingerprint
                and cached.get("codex_news_context_sha256") == codex_news_context_sha256
                and cached.get("run_id") == run_id
                and not force
            ):
                reused = deepcopy(cached)
                reused["reused_from_checkpoint"] = True
                results.append(reused)
                scorecard_by_task[task_id] = reused["scorecard"]
                continue

            reports = research["reports"]
            prompt = build_synthesis_prompt(
                task,
                reports,
                str(research["tradingagents_decision"]),
                codex_news_context,
            )
            if structured_llm is None:
                raise RuntimeError("structured synthesis LLM is unavailable")
            synthesized = invoke_synthesis(structured_llm, synthesis_llm, prompt)
            card = merge_synthesis(task, synthesized)
            validation = validator(card) if validator else {
                "ready": False,
                "reasons": ["deterministic scorecard validator is required"],
            }
            validation_ready = isinstance(validation, dict) and validation.get("ready") is True
            result = {
                "task_id": task_id,
                "ticker": ticker,
                "strategy": task.get("strategy"),
                "status": "ok" if validation_ready else "invalid",
                "started_at": started,
                "finished_at": _now(),
                "task_fingerprint": fingerprint,
                "research_fingerprint": research_fingerprint,
                "synthesis_fingerprint": synthesis_fingerprint,
                "codex_news_context_sha256": codex_news_context_sha256,
                "codex_news_analysis_path": codex_news_context.get("analysis_path"),
                "codex_news_analysis_sha256": codex_news_context.get("analysis_sha256"),
                "codex_news_operation_path": codex_news_context.get("operation_path"),
                "codex_news_operation_sha256": codex_news_context.get("operation_sha256"),
                "raw_news_prose_exposed": False,
                "run_id": run_id,
                "research_input_manifest_path": research["research_input_manifest_path"],
                "research_input_manifest_sha256": research["research_input_manifest_sha256"],
                "tradingagents_decision": research["tradingagents_decision"],
                "tradingagents_report_path": research["tradingagents_report_path"],
                "validation": validation,
                "scorecard": card,
                "reused_from_checkpoint": False,
            }
            if validation_ready:
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
                "research_fingerprint": research_fingerprint,
                "synthesis_fingerprint": synthesis_fingerprint,
                "run_id": run_id,
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
    dated_operation = output_dir / f"llm_confirmation_auto_operation_{queue_date}_{run_id}.json"
    latest_run = output_dir / "llm_confirmation_auto_analysis_latest.json"
    latest_scorecards = output_dir / "llm_confirmation_auto_scorecards_latest.json"
    latest_operation = output_dir / "llm_confirmation_auto_operation_latest.json"
    scorecards = list(scorecard_by_task.values())
    tasks_validated = sum(row.get("status") == "ok" for row in results)
    tasks_invalid = sum(row.get("status") == "invalid" for row in results)
    tasks_failed = sum(row.get("status") == "error" for row in results)
    if not selected:
        batch_status = "blocked"
    elif tasks_invalid or tasks_failed:
        batch_status = "partial" if tasks_validated else "blocked"
    else:
        batch_status = "ok"
    summary = {
        "tasks_selected": len(selected),
        "tasks_attempted": len(results),
        "tasks_validated": tasks_validated,
        "tasks_completed": tasks_validated,
        "tasks_invalid": tasks_invalid,
        "tasks_failed": tasks_failed,
        "tasks_reused": sum(bool(row.get("reused_from_checkpoint")) for row in results),
        "unique_tickers_analyzed": graph_executions,
        "unique_research_fingerprints": len(research_fingerprints),
        "graph_executions": graph_executions,
        "same_run_research_resumes": research_resumes,
        "duplicate_graph_executions": max(0, graph_executions - len(research_fingerprints)),
        "research_cache_scope": "batch_and_same_run_capture_replay_only",
        "cross_run_cache_enabled": False,
        "scorecards_written": len(scorecards),
        "scorecards_preserved": 0,
        "validation_ready": tasks_validated,
    }
    payload = {
        "schema_version": "llm-confirmation-analysis.v2",
        "generated_at": _now(),
        "status": batch_status,
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
            "operation_json": str(dated_operation),
            "operation_latest_json": str(latest_operation),
            "checkpoint_json": str(checkpoint_path),
            "report_root": str(run_root),
        },
        "note": "Automated research and scorecard synthesis only. No proposals were sized and no orders were placed.",
    }
    scorecard_payload = {
        "schema_version": SCORECARD_ARTIFACT_CONTRACT,
        "generated_at": payload["generated_at"],
        "status": batch_status,
        "as_of_date": queue_date,
        "source_queue_json": str(queue_file),
        "source_queue_sha256": queue_hash,
        "source_analysis_json": str(dated_run),
        "operation_json": str(dated_operation),
        "run_signature": signature,
        "candidates": scorecards,
        "note": "Auto-filled research scorecards; deterministic review is still required before proposals.",
    }
    _immutable_json(dated_run, payload)
    _atomic_json(latest_run, payload)
    _immutable_json(dated_scorecards, scorecard_payload)
    _atomic_json(latest_scorecards, scorecard_payload)
    operation = {
        "schema_version": OPERATION_CONTRACT,
        "generated_at": payload["generated_at"],
        "status": batch_status,
        "decision_ready": batch_status == "ok",
        "as_of_date": queue_date,
        "run_id": run_id,
        "source_queue_json": str(queue_file),
        "source_queue_sha256": queue_hash,
        "source_analysis_json": str(dated_run),
        "source_analysis_sha256": _file_hash(dated_run),
        "scorecards_json": str(dated_scorecards),
        "scorecards_sha256": _file_hash(dated_scorecards),
        "run_signature": signature,
        "summary": summary,
        "task_validations": [
            {
                "task_id": row.get("task_id"),
                "task_fingerprint": row.get("task_fingerprint"),
                "research_fingerprint": row.get("research_fingerprint"),
                "synthesis_fingerprint": row.get("synthesis_fingerprint"),
                "research_input_manifest_path": row.get("research_input_manifest_path"),
                "research_input_manifest_sha256": row.get("research_input_manifest_sha256"),
                "status": row.get("status"),
                "validation_ready": bool((row.get("validation") or {}).get("ready")),
                "validation_sha256": _hash(row.get("validation") or {}),
            }
            for row in results
        ],
        "note": "Only status=ok operations with validation-ready tasks may enter deterministic review.",
    }
    _immutable_json(dated_operation, operation)
    _atomic_json(latest_operation, operation)
    return payload
