import hashlib
import json
from pathlib import Path

import pytest

from tradingagents.decision_intelligence_runner import (
    _validated_codex_news_context,
    CHECKPOINT_CONTRACT,
    CONTRACT_VERSION,
    OPERATION_CONTRACT,
    ResearchSynthesis,
    build_synthesis_prompt,
    invoke_synthesis,
    merge_synthesis,
    run_queue,
)


def _synthesis():
    return ResearchSynthesis.model_validate(
        {
            "llm_decision": "confirm",
            "thesis": "Backlog and signed demand support estimate revisions.",
            "catalyst": "Raised guidance and a signed multi-year contract.",
            "realization_path": "Shipments lift revenue and utilization lifts EPS.",
            "business_quality": "Profitable supplier with recurring customer demand.",
            "supply_demand": "Demand exceeds near-term qualified capacity.",
            "fundamental_verdict": "pass",
            "valuation_verdict": "fair",
            "sentiment_stage": "improving",
            "regime_fit": "Company catalyst offsets mixed beta.",
            "macro_context": {
                "impact_matrix_readthrough": "No direct blocked-theme hit.",
                "global_event_readthrough": "Trade policy is the main headwind.",
                "supports_proposal": True,
                "risk_vetoes": [],
            },
            "style_confirmations": {
                "estimate_revision_support": True,
                "valuation_disciplined": True,
                "macro_or_cycle_support": True,
                "inflection_catalyst": True,
                "balance_sheet_runway": True,
                "momentum_supportive_not_euphoric": True,
                "media_crowding_review": True,
            },
            "media_crowding": "Independent primary evidence supports the thesis.",
            "bull_case": "Contracts and mix exceed plan.",
            "bear_case": "Deployment slips and the multiple compresses.",
            "risk_vetoes": [],
            "tradingagents_decision": "TradingAgents debate supports monitoring and entry.",
            "scores": {
                "business_quality": 8,
                "catalyst_strength": 8,
                "realization_path": 8,
                "fundamental_support": 8,
                "valuation_safety": 7,
                "sentiment_timing": 7,
                "risk_reward": 8,
                "tradingagents_confidence": 7,
            },
            "evidence_ledger": [
                {
                    "evidence_id": "E1",
                    "source_type": "earnings_release",
                    "title": "Results",
                    "url": "https://ir.example.com/results",
                    "published_at": "2026-07-10",
                    "claim": "Backlog rose 25%.",
                    "metric": "backlog_growth",
                    "current_value": 25,
                    "prior_value": 10,
                    "unit": "%",
                    "direction": "tailwind",
                    "reliability_0_to_1": 0.95,
                    "relevance_0_to_1": 0.95,
                },
                {
                    "evidence_id": "E2",
                    "source_type": "industry_data",
                    "title": "Demand survey",
                    "url": "https://industry.example.org/survey",
                    "published_at": "2026-07-09",
                    "claim": "Customer units are expected to rise 12%.",
                    "metric": "unit_growth",
                    "current_value": 12,
                    "prior_value": 8,
                    "unit": "%",
                    "direction": "tailwind",
                    "reliability_0_to_1": 0.85,
                    "relevance_0_to_1": 0.9,
                },
                {
                    "evidence_id": "E3",
                    "source_type": "credible_news",
                    "title": "Contract",
                    "url": "https://news.example.net/contract",
                    "published_at": "2026-07-08",
                    "claim": "A multi-year contract was signed.",
                    "metric": "contract_value",
                    "current_value": None,
                    "prior_value": None,
                    "unit": "",
                    "direction": "tailwind",
                    "reliability_0_to_1": 0.8,
                    "relevance_0_to_1": 0.85,
                },
            ],
            "causal_chains": [
                {
                    "chain_id": "C1",
                    "evidence_ids": ["E1", "E3"],
                    "operating_driver": "units",
                    "mechanism": "Backlog converts to shipments.",
                    "financial_metric": "revenue",
                    "impact_low_pct": 4,
                    "impact_base_pct": 8,
                    "impact_high_pct": 12,
                    "lag_days": 126,
                    "confidence_0_to_1": 0.8,
                    "falsifiers": ["book-to-bill below 1"],
                },
                {
                    "chain_id": "C2",
                    "evidence_ids": ["E1", "E2"],
                    "operating_driver": "gross_margin",
                    "mechanism": "Utilization raises operating leverage.",
                    "financial_metric": "eps",
                    "impact_low_pct": 2,
                    "impact_base_pct": 6,
                    "impact_high_pct": 10,
                    "lag_days": 126,
                    "confidence_0_to_1": 0.7,
                    "falsifiers": ["gross-margin guidance cut"],
                },
            ],
            "scenario_model": {
                "method": "earnings_multiple",
                "forecast_horizon_days": 126,
                "scenarios": [
                    {"name": "bear", "probability_pct": 20, "implied_value": 80, "assumptions": ["ramp slips"]},
                    {"name": "base", "probability_pct": 50, "implied_value": 118, "assumptions": ["guide met"]},
                    {"name": "bull", "probability_pct": 30, "implied_value": 145, "assumptions": ["mix improves"]},
                ],
            },
            "sensitivity": [
                {"variable": "revenue growth", "low": 8, "base": 14, "high": 20, "valuation_effect_low_pct": -10, "valuation_effect_high_pct": 12},
                {"variable": "margin", "low": 18, "base": 21, "high": 24, "valuation_effect_low_pct": -8, "valuation_effect_high_pct": 10},
            ],
            "disconfirming_evidence": ["A smaller customer delayed deployment."],
            "monitoring_triggers": [
                {"metric": "book-to-bill", "trigger": "below", "threshold": 1, "action": "reduce", "evidence_source": "filing"},
                {"metric": "gross margin", "trigger": "below", "threshold": 20, "action": "reassess", "evidence_source": "results"},
            ],
            "analyst_decision": {
                "action": "add",
                "confidence_0_to_1": 0.8,
                "requested_position_scale_0_to_1": 0.7,
                "summary": "Evidence supports adding below the max position.",
            },
        }
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(task_id, strategy):
    return {
        "task_id": task_id,
        "as_of_date": "2026-07-10",
        "ticker": "ACME",
        "strategy": strategy,
        "confirmation_decision": "pending",
        "current_snapshot": {"last_price": 100},
        "scorecard_template": {
            "analysis_contract_version": CONTRACT_VERSION,
            "symbol": "ACME",
            "macro_context": {"ready": True},
            "style_confirmations": {},
            "decision_analysis": {"contract_version": CONTRACT_VERSION},
        },
    }


class FakeGraph:
    propagations = 0

    def propagate(self, ticker, analysis_date, asset_type="stock"):
        type(self).propagations += 1
        return (
            {
                "market_report": "Price context.",
                "sentiment_report": "Sentiment context.",
                "news_report": "News with sources.",
                "fundamentals_report": "Fundamental evidence.",
                "investment_plan": "Research plan.",
                "trader_investment_plan": "Trader plan.",
                "final_trade_decision": "Overweight.",
                "investment_debate_state": {"history": "Bull and bear."},
                "risk_debate_state": {"history": "Risk debate."},
            },
            "OVERWEIGHT",
        )

    def save_reports(self, state, ticker, save_path):
        return save_path / "complete_report.md"


class FakeStructured:
    def __init__(self):
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return _synthesis()


class FakeLLM:
    def __init__(self):
        self.structured = FakeStructured()

    def with_structured_output(self, schema):
        assert schema is ResearchSynthesis
        return self.structured


def test_synthesis_uses_strict_schema_validated_json_fallback():
    class BrokenStructured:
        def invoke(self, prompt):
            raise ValueError("missing parsed field")

    class JsonMessage:
        content = f"```json\n{_synthesis().model_dump_json()}\n```"

    class Plain:
        def invoke(self, prompt):
            assert '"evidence_ledger"' in prompt
            return JsonMessage()

    result = invoke_synthesis(BrokenStructured(), Plain(), "research prompt")

    assert result.llm_decision == "confirm"
    assert result.scenario_model.scenarios[0].name == "bear"


def test_validated_codex_news_injection_exposes_interpretation_but_not_raw_prose(tmp_path):
    bundle = tmp_path / "news_pipeline_2026-07-15_run.json"
    analysis = tmp_path / "news_analysis_2026-07-15_run.json"
    operation = tmp_path / "news_operation_2026-07-15_run.json"
    bundle.write_text('{"status":"ok"}', encoding="utf-8")
    bundle_sha = _sha(bundle)
    analysis_payload = {
        "schema_version": "news_analysis.v3",
        "contract_version": "codex-news-impact.v3",
        "provider": "codex_automation",
        "authorship": {"mode": "direct_skill_execution"},
        "analysis_ready": True,
        "decision_ready": True,
        "status": "ready",
        "trading_date": "2026-07-15",
        "model": "codex-test",
        "events": [
            {
                "event_id": "event-1",
                "affected_tickers": ["SMH"],
                "analysis_layers": ["supply_chain"],
                "direction": "positive",
                "summary_zh": "CODEX_INTERPRETED_SUMMARY",
                "source_article_ids": ["article-1"],
                "scenarios": [],
            }
        ],
        "ticker_impacts": [{"ticker": "SMH", "expected_price_impact_pct": 2.0}],
        "source_articles": [
            {
                "article_id": "article-1",
                "title": "RAW_HEADLINE_MUST_NOT_APPEAR",
                "summary": "RAW_BODY_MUST_NOT_APPEAR",
                "url": "https://example.test/primary",
                "published_at": "2026-07-15T12:00:00Z",
                "observed_at": "2026-07-15T12:01:00Z",
                "available_at": "2026-07-15T12:01:00Z",
                "retrieved_at": "2026-07-15T12:01:00Z",
                "origin_domain": "example.test",
                "origin_category": "company_primary",
                "quality_tier": "A",
                "provider_families": ["company"],
            }
        ],
    }
    analysis.write_text(json.dumps(analysis_payload), encoding="utf-8")
    analysis_sha = _sha(analysis)
    operation.write_text(
        json.dumps(
            {
                "schema_version": "news_research_operation.v2",
                "status": "ok",
                "lineage": {"analysis_sha256": analysis_sha, "bundle_sha256": bundle_sha},
            }
        ),
        encoding="utf-8",
    )
    task = _task("task-news", "Strategy A")
    task["as_of_date"] = "2026-07-15"
    task["ticker"] = "SMH"
    task["canonical_news_lineage"] = {
        "analysis_path": str(analysis),
        "analysis_sha256": analysis_sha,
        "operation_path": str(operation),
        "operation_sha256": _sha(operation),
        "bundle_path": str(bundle),
        "bundle_sha256": bundle_sha,
    }

    context = _validated_codex_news_context(task)
    prompt = build_synthesis_prompt(task, {}, "HOLD", context)

    assert context["status"] == "ready"
    assert context["events"][0]["summary_zh"] == "CODEX_INTERPRETED_SUMMARY"
    assert context["source_metadata"][0]["url"] == "https://example.test/primary"
    assert context["raw_headlines_exposed"] is False
    assert "RAW_HEADLINE_MUST_NOT_APPEAR" not in prompt
    assert "RAW_BODY_MUST_NOT_APPEAR" not in prompt
    assert "CODEX_INTERPRETED_SUMMARY" in prompt


def test_synthesis_json_fallback_rejects_free_prose():
    class BrokenStructured:
        def invoke(self, prompt):
            raise ValueError("missing parsed field")

    class Plain:
        def invoke(self, prompt):
            return "The answer is probably confirm."

    with pytest.raises(RuntimeError, match="strict JSON fallback failed"):
        invoke_synthesis(BrokenStructured(), Plain(), "research prompt")


def test_merge_synthesis_preserves_task_identity_and_builds_evidence_links():
    card = merge_synthesis(_task("task-1", "Strategy A"), _synthesis())

    assert card["llm_confirmation_task_id"] == "task-1"
    assert card["analysis_contract_version"] == CONTRACT_VERSION
    assert card["decision_analysis"]["evidence_ledger"][0]["evidence_id"] == "E1"
    assert card["evidence"][0]["url"] == "https://ir.example.com/results"
    assert card["macro_context"]["supports_proposal"] is True


def test_batch_reuses_by_ticker_but_disables_unsound_cross_run_cache(tmp_path):
    FakeGraph.propagations = 0
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps({"as_of_date": "2026-07-10", "tasks": [_task("task-1", "A"), _task("task-2", "B")]}),
        encoding="utf-8",
    )
    llm = FakeLLM()

    first = run_queue(
        queue_path=queue,
        out_dir=tmp_path / "out",
        run_id="run-1",
        graph_factory=FakeGraph,
        synthesis_llm=llm,
        validator=lambda card: {"ready": True},
    )

    assert first["summary"]["tasks_completed"] == 2
    assert first["summary"]["tasks_validated"] == 2
    assert first["summary"]["tasks_invalid"] == 0
    assert first["summary"]["unique_tickers_analyzed"] == 1
    assert FakeGraph.propagations == 1
    assert len(llm.structured.calls) == 2
    cards = json.loads((tmp_path / "out" / "llm_confirmation_auto_scorecards_latest.json").read_text())
    assert {card["llm_confirmation_task_id"] for card in cards["candidates"]} == {"task-1", "task-2"}

    refreshed_queue = json.loads(queue.read_text(encoding="utf-8"))
    refreshed_queue["generated_at"] = "2026-07-10T23:59:59Z"
    queue.write_text(json.dumps(refreshed_queue), encoding="utf-8")

    second = run_queue(
        queue_path=queue,
        out_dir=tmp_path / "out",
        run_id="run-2",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda card: {"ready": True},
    )

    assert second["summary"]["tasks_reused"] == 0
    assert second["summary"]["cross_run_cache_enabled"] is False
    assert second["summary"]["unique_tickers_analyzed"] == 1
    assert FakeGraph.propagations == 2
    assert first["source_queue_sha256"] != second["source_queue_sha256"]
    assert second["run_signature"]["checkpoint_contract"] == CHECKPOINT_CONTRACT
    operation = json.loads((tmp_path / "out" / second["artifacts"]["operation_json"]).read_text())
    assert operation["schema_version"] == OPERATION_CONTRACT
    assert operation["status"] == "ok"
    assert operation["decision_ready"] is True


def test_batch_preserves_task_error_and_continues(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"as_of_date": "2026-07-10", "tasks": [_task("task-1", "A")]}))

    class BrokenStructured:
        def invoke(self, prompt):
            raise RuntimeError("provider unavailable")

    class BrokenLLM:
        def with_structured_output(self, schema):
            return BrokenStructured()

    result = run_queue(
        queue_path=queue,
        out_dir=tmp_path / "out",
        run_id="broken",
        graph_factory=FakeGraph,
        synthesis_llm=BrokenLLM(),
    )

    assert result["summary"]["tasks_failed"] == 1
    assert result["summary"]["scorecards_written"] == 0
    assert "provider unavailable" in result["results"][0]["error"]
    assert result["status"] == "blocked"


def test_twenty_five_tasks_across_eleven_tickers_execute_at_most_eleven_graphs(tmp_path):
    FakeGraph.propagations = 0
    tickers = [f"T{i:02d}" for i in range(11)]
    tasks = []
    for index in range(25):
        task = _task(f"task-{index:02d}", f"strategy-{index:02d}")
        task["ticker"] = tickers[index % len(tickers)]
        task["scorecard_template"]["symbol"] = task["ticker"]
        tasks.append(task)
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"as_of_date": "2026-07-10", "tasks": tasks}), encoding="utf-8")

    result = run_queue(
        queue_path=queue,
        out_dir=tmp_path / "out",
        run_id="capacity-fixture",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda _card: {"ready": True},
    )

    assert result["summary"]["tasks_validated"] == 25
    assert result["summary"]["unique_research_fingerprints"] == 11
    assert result["summary"]["graph_executions"] == 11
    assert result["summary"]["duplicate_graph_executions"] == 0
    assert FakeGraph.propagations == 11


def test_interrupted_same_run_resumes_research_and_checkpoint_without_new_graph(tmp_path):
    FakeGraph.propagations = 0
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"as_of_date": "2026-07-10", "tasks": [_task("task-1", "A")]}), encoding="utf-8")
    out = tmp_path / "out"
    first = run_queue(
        queue_path=queue,
        out_dir=out,
        run_id="resume-run",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda _card: {"ready": True},
    )
    for key in ("analysis_json", "scorecards_json", "operation_json"):
        Path(first["artifacts"][key]).unlink()

    resumed = run_queue(
        queue_path=queue,
        out_dir=out,
        run_id="resume-run",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda _card: {"ready": True},
    )

    assert resumed["summary"]["tasks_reused"] == 1
    assert resumed["summary"]["same_run_research_resumes"] == 1
    assert resumed["summary"]["graph_executions"] == 0
    assert FakeGraph.propagations == 1


def test_batch_does_not_copy_unvalidated_existing_cards_into_v2_output(tmp_path):
    prior = {"llm_confirmation_task_id": "task-confirmed", "symbol": "OLD", "llm_decision": "confirm"}
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            {
                "as_of_date": "2026-07-10",
                "tasks": [
                    {
                        "task_id": "task-confirmed",
                        "ticker": "OLD",
                        "confirmation_decision": "confirm",
                        "filled_scorecard": prior,
                    },
                    _task("task-pending", "A"),
                ],
            }
        )
    )

    result = run_queue(
        queue_path=queue,
        out_dir=tmp_path / "out",
        run_id="incremental",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda card: {"ready": True},
    )

    cards = json.loads((tmp_path / "out" / "llm_confirmation_auto_scorecards_latest.json").read_text())
    assert result["summary"]["scorecards_preserved"] == 0
    assert {card["llm_confirmation_task_id"] for card in cards["candidates"]} == {"task-pending"}


def test_validation_false_is_invalid_and_never_enters_scorecard_artifact(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"as_of_date": "2026-07-10", "tasks": [_task("task-1", "A")]}))

    result = run_queue(
        queue_path=queue,
        out_dir=tmp_path / "out",
        run_id="invalid",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda card: {"ready": False, "reasons": ["primary source missing"]},
    )

    assert result["status"] == "blocked"
    assert result["summary"]["tasks_validated"] == 0
    assert result["summary"]["tasks_invalid"] == 1
    assert result["results"][0]["status"] == "invalid"
    cards = json.loads((tmp_path / "out" / "llm_confirmation_auto_scorecards_latest.json").read_text())
    assert cards["status"] == "blocked"
    assert cards["candidates"] == []
    operation = json.loads(Path(result["artifacts"]["operation_json"]).read_text())
    assert operation["status"] == "blocked"
    assert operation["decision_ready"] is False


def test_real_2026_07_14_smh_validation_fixture_is_invalid_under_v2(tmp_path):
    fixture_path = Path(__file__).parent / "fixtures" / "decision_intelligence_invalid_smh_2026-07-14.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    queue = tmp_path / "queue.json"
    task = _task(fixture["task_id"], "IC factor rotation momentum active top1")
    task["ticker"] = fixture["ticker"]
    task["scorecard_template"]["symbol"] = fixture["ticker"]
    queue.write_text(json.dumps({"as_of_date": "2026-07-14", "tasks": [task]}), encoding="utf-8")

    result = run_queue(
        queue_path=queue,
        out_dir=tmp_path / "out",
        run_id="smh-regression",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda card: fixture["validation"],
    )

    assert fixture["legacy_status"] == "ok"
    assert result["results"][0]["status"] == fixture["expected_v2_status"]
    assert result["summary"]["tasks_invalid"] == 1
    assert result["summary"]["scorecards_written"] == 0


def test_v1_checkpoint_and_validation_false_v2_cache_are_not_reused(tmp_path):
    FakeGraph.propagations = 0
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"as_of_date": "2026-07-10", "tasks": [_task("task-1", "A")]}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "llm_confirmation_auto_checkpoint.json").write_text(
        json.dumps({"signature": {"checkpoint_contract": "decision-intelligence-task.v1"}, "tasks": {"bad": {"status": "ok"}}})
    )

    first = run_queue(
        queue_path=queue,
        out_dir=out,
        run_id="first",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda card: {"ready": False, "reasons": ["invalid"]},
    )
    second = run_queue(
        queue_path=queue,
        out_dir=out,
        run_id="second",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda card: {"ready": True},
    )

    assert first["summary"]["tasks_invalid"] == 1
    assert second["summary"]["tasks_reused"] == 0
    assert second["summary"]["tasks_validated"] == 1
    assert FakeGraph.propagations == 2


def test_corrupt_v2_checkpoint_is_ignored_and_rebuilt_fail_closed(tmp_path):
    FakeGraph.propagations = 0
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"as_of_date": "2026-07-10", "tasks": [_task("task-1", "A")]}))
    out = tmp_path / "out"
    out.mkdir()
    (out / "llm_confirmation_auto_checkpoint_v2.json").write_text("{corrupt", encoding="utf-8")

    result = run_queue(
        queue_path=queue,
        out_dir=out,
        run_id="corrupt-checkpoint",
        graph_factory=FakeGraph,
        synthesis_llm=FakeLLM(),
        validator=lambda _card: {"ready": True},
    )

    assert result["summary"]["tasks_reused"] == 0
    assert result["summary"]["tasks_validated"] == 1
    assert FakeGraph.propagations == 1


def test_confirmation_prompt_excludes_raw_news_and_social_prose():
    task = _task("task-1", "A")
    task["news_context"] = {"headlines": ["RAW_HEADLINE_SENTINEL"]}
    prompt = build_synthesis_prompt(
        task,
        {
            "market_report": "structured price context",
            "fundamentals_report": "structured filing fields",
            "news_report": "RAW_NEWS_REPORT_SENTINEL",
            "sentiment_report": "RAW_SOCIAL_REPORT_SENTINEL",
        },
        "HOLD",
    )

    assert "RAW_HEADLINE_SENTINEL" not in prompt
    assert "RAW_NEWS_REPORT_SENTINEL" not in prompt
    assert "RAW_SOCIAL_REPORT_SENTINEL" not in prompt


def test_confirmation_rejects_news_and_social_analysts(tmp_path):
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps({"as_of_date": "2026-07-10", "tasks": [_task("task-1", "A")]}))

    with pytest.raises(ValueError, match="unsupported analysts: news, social"):
        run_queue(
            queue_path=queue,
            out_dir=tmp_path / "out",
            analysts=["market", "news", "social", "fundamentals"],
            graph_factory=FakeGraph,
            synthesis_llm=FakeLLM(),
            validator=lambda card: {"ready": True},
        )
