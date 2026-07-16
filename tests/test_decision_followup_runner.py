import json

from tradingagents.decision_followup_runner import (
    FollowupSynthesis,
    _validated_observations,
    followup_tasks,
    run_followup,
)


def _calibration():
    return {
        "rows": [
            {
                "task_id": "task-1",
                "ticker": "ACME",
                "strategy": "A",
                "prediction_date": "2026-01-02",
                "state": "open",
                "business_verification": {
                    "monitoring": [{"metric": "backlog growth", "observed": False, "threshold": 10}],
                    "causal_chains": [{"chain_id": "C1", "financial_metric": "revenue", "observed": False}],
                },
            },
            {
                "task_id": "task-2",
                "ticker": "ACME",
                "strategy": "B",
                "prediction_date": "2026-01-02",
                "state": "open",
                "business_verification": {
                    "monitoring": [{"metric": "gross margin", "observed": False}],
                    "causal_chains": [],
                },
            },
        ]
    }


def _synthesis(task_id="task-1", metric="backlog growth"):
    return FollowupSynthesis.model_validate(
        {
            "observations": [
                {
                    "task_id": task_id,
                    "ticker": "ACME",
                    "kind": "monitoring_trigger",
                    "status": "observed",
                    "metric": metric,
                    "value": 18,
                    "unit": "%",
                    "observed_at": "2026-02-02",
                    "published_at": "2026-02-02",
                    "source_url": "https://ir.example.com/results",
                    "source_type": "earnings_release",
                    "notes": "Backlog growth reported by the company.",
                }
            ],
            "unresolved": [],
            "summary": "One metric observed.",
        }
    )


class FakeGraph:
    calls = 0

    def propagate(self, ticker, analysis_date, asset_type="stock"):
        type(self).calls += 1
        return (
            {
                "news_report": "2026-02-02 company result https://ir.example.com/results with values.",
                "fundamentals_report": "Backlog data.",
            },
            "WATCH",
        )

    def save_reports(self, state, ticker, save_path):
        return save_path / "complete_report.md"


class FakeStructured:
    def __init__(self):
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return _synthesis("task-1" if "task-1" in prompt else "task-2", "backlog growth" if "task-1" in prompt else "gross margin")


class FakeLLM:
    def __init__(self):
        self.structured = FakeStructured()

    def with_structured_output(self, schema):
        assert schema is FollowupSynthesis
        return self.structured


def test_followup_tasks_select_only_missing_open_observations():
    tasks = followup_tasks(_calibration())
    assert [task["task_id"] for task in tasks] == ["task-1", "task-2"]
    assert tasks[0]["causal_chains"][0]["chain_id"] == "C1"


def test_validation_rejects_unrequested_or_future_observations():
    task = followup_tasks(_calibration(), max_tasks=1)[0]
    wrong_metric = _synthesis(metric="invented metric")
    valid, rejected = _validated_observations(
        wrong_metric,
        task,
        as_of_date="2026-02-02",
        report_text="2026-02-02 https://ir.example.com/results",
    )
    assert valid == []
    assert rejected[0]["rejection_reason"] == "metric was not requested"

    valid_source_shape = _synthesis()
    valid, rejected = _validated_observations(
        valid_source_shape,
        task,
        as_of_date="2026-02-02",
        report_text="2026-02-02 report without the claimed URL",
    )
    assert valid == []
    assert rejected[0]["rejection_reason"] == "source URL is not present in TradingAgents reports"


def test_followup_reuses_research_by_ticker_and_writes_observations(tmp_path):
    FakeGraph.calls = 0
    source = tmp_path / "calibration.json"
    source.write_text(json.dumps(_calibration()), encoding="utf-8")
    llm = FakeLLM()

    result = run_followup(
        calibration_path=source,
        out_dir=tmp_path / "out",
        as_of_date="2026-02-02",
        graph_factory=FakeGraph,
        synthesis_llm=llm,
        run_id="test",
    )

    assert result["summary"]["tasks_completed"] == 2
    assert result["summary"]["unique_tickers_analyzed"] == 1
    assert result["summary"]["observations_written"] == 2
    assert FakeGraph.calls == 1
    observations = json.loads((tmp_path / "out" / "decision_intelligence_observations_2026-02-02_test.json").read_text())
    assert observations["schema_version"] == "decision_intelligence_observations_v1"
    assert {row["task_id"] for row in observations["observations"]} == {"task-1", "task-2"}
