import json

import pytest

from tradingagents.decision_intelligence_runner import (
    CONTRACT_VERSION,
    ResearchSynthesis,
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


def test_batch_reuses_tradingagents_by_ticker_and_resumes_from_checkpoint(tmp_path):
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

    assert second["summary"]["tasks_reused"] == 2
    assert second["summary"]["unique_tickers_analyzed"] == 0
    assert FakeGraph.propagations == 1
    assert first["source_queue_sha256"] != second["source_queue_sha256"]
    assert second["run_signature"]["checkpoint_contract"] == "decision-intelligence-task.v1"


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


def test_batch_preserves_matching_existing_cards_for_incremental_review(tmp_path):
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
    assert result["summary"]["scorecards_preserved"] == 1
    assert {card["llm_confirmation_task_id"] for card in cards["candidates"]} == {
        "task-confirmed",
        "task-pending",
    }
