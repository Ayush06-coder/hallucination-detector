"""
Integration tests for agents/orchestrator.py.

Mocks each specialist agent function directly (not the LLM client),
so these test the orchestration logic itself: that run_all_agents
wires results together correctly, that a failing agent doesn't
crash the pipeline, and that aggregate_verdict/save_result/
build_result behave as expected.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.orchestrator import (
    run_all_agents,
    aggregate_verdict,
    build_result,
    save_result,
    run_detection,
)


FACT_TRUE = "VERDICT: TRUE\nREASON: Confirmed by evidence.\nEVIDENCE: Canberra is the capital."
CONSISTENCY_CONSISTENT = "VERDICT: CONSISTENT\nREASON: Matches question.\nKEY_CLAIM: Canberra is the capital."
CONFIDENCE_HIGH = "CONFIDENCE: 90\nREASON: Strong agreement across agents."
FINAL_TRUE = SimpleNamespace(
    content="FINAL_VERDICT: TRUE\nCONFIDENCE_SCORE: 92\nEXPLANATION: All agents agree the answer is correct."
)


# ============================================================
# run_all_agents
# ============================================================

class TestRunAllAgents:

    @patch("agents.orchestrator.confidence_agent", return_value=CONFIDENCE_HIGH)
    @patch("agents.orchestrator.consistency_agent", return_value=CONSISTENCY_CONSISTENT)
    @patch("agents.orchestrator.fact_check_agent", return_value=FACT_TRUE)
    def test_all_agents_succeed(self, mock_fact, mock_consistency, mock_confidence):
        results = run_all_agents("Q?", "A.")

        assert results["fact_check"]["verdict"] == "TRUE"
        assert results["consistency"]["verdict"] == "CONSISTENT"
        assert results["confidence"]["score"] == 90

    @patch("agents.orchestrator.confidence_agent", return_value=CONFIDENCE_HIGH)
    @patch("agents.orchestrator.consistency_agent", return_value=CONSISTENCY_CONSISTENT)
    @patch("agents.orchestrator.fact_check_agent", side_effect=Exception("search API down"))
    def test_pipeline_survives_one_agent_failing(self, mock_fact, mock_consistency, mock_confidence):
        """A single agent crashing must not take down the whole pipeline."""
        results = run_all_agents("Q?", "A.")

        assert results["fact_check"]["verdict"] == "UNCERTAIN"
        assert results["fact_check"]["reason"] == "Fact-checking failed."
        assert results["consistency"]["verdict"] == "CONSISTENT"
        assert results["confidence"]["score"] == 90

    @patch("agents.orchestrator.confidence_agent", side_effect=Exception("x"))
    @patch("agents.orchestrator.consistency_agent", side_effect=Exception("x"))
    @patch("agents.orchestrator.fact_check_agent", side_effect=Exception("x"))
    def test_pipeline_survives_all_agents_failing(self, mock_fact, mock_consistency, mock_confidence):
        """Total upstream outage should degrade to all-UNCERTAIN, not crash."""
        results = run_all_agents("Q?", "A.")

        assert results["fact_check"]["verdict"] == "UNCERTAIN"
        assert results["consistency"]["verdict"] == "UNCERTAIN"
        assert results["confidence"]["score"] == 0


# ============================================================
# aggregate_verdict
# ============================================================

class TestAggregateVerdict:

    @patch("agents.orchestrator.llm")
    def test_returns_parsed_verdict(self, mock_llm):
        mock_llm.invoke.return_value = FINAL_TRUE

        results = {
            "fact_check": {"verdict": "TRUE", "reason": "ok", "evidence": "ok", "raw": FACT_TRUE},
            "consistency": {"verdict": "CONSISTENT", "reason": "ok", "key_claim": "x", "raw": CONSISTENCY_CONSISTENT},
            "confidence": {"score": 90, "reason": "ok", "raw": CONFIDENCE_HIGH},
        }

        final = aggregate_verdict("Q?", "A.", results)

        assert final["verdict"] == "TRUE"
        assert final["confidence_score"] == 92

    @patch("agents.orchestrator.llm")
    def test_handles_missing_confidence_score_gracefully(self, mock_llm):
        """If confidence.get('score') is None (not int), the prompt build must not crash."""
        mock_llm.invoke.return_value = FINAL_TRUE

        results = {
            "fact_check": {"verdict": "UNCERTAIN", "reason": "x", "evidence": "x", "raw": ""},
            "consistency": {"verdict": "UNCERTAIN", "reason": "x", "key_claim": "", "raw": ""},
            "confidence": {"score": None, "reason": "x", "raw": ""},
        }

        final = aggregate_verdict("Q?", "A.", results)
        assert final["verdict"] in {"TRUE", "HALLUCINATED", "UNCERTAIN"}

    @patch("agents.orchestrator.llm")
    def test_final_judge_llm_failure_falls_back_to_uncertain(self, mock_llm):
        mock_llm.invoke.side_effect = Exception("groq down")

        results = {
            "fact_check": {"verdict": "TRUE", "reason": "x", "evidence": "x", "raw": ""},
            "consistency": {"verdict": "CONSISTENT", "reason": "x", "key_claim": "", "raw": ""},
            "confidence": {"score": 80, "reason": "x", "raw": ""},
        }

        final = aggregate_verdict("Q?", "A.", results)
        assert final["verdict"] == "UNCERTAIN"
        assert final["confidence_score"] == 0


# ============================================================
# build_result / save_result
# ============================================================

class TestBuildAndSaveResult:

    def test_build_result_shape(self):
        agent_results = {"fact_check": {}, "consistency": {}, "confidence": {}}
        final_result = {"verdict": "TRUE", "confidence_score": 90, "explanation": "x"}

        result = build_result("Q?", "A.", agent_results, final_result)

        assert result["question"] == "Q?"
        assert result["answer"] == "A."
        assert result["agents"] == agent_results
        assert result["final"] == final_result
        assert "timestamp" in result

    def test_save_result_writes_json_to_repo_data_results_dir(self, tmp_path, monkeypatch):
        """
        save_result must write into <project_root>/data/results
        regardless of the current working directory.
        """
        monkeypatch.chdir(tmp_path)

        result = {"question": "Q?", "answer": "A.", "final": {"verdict": "TRUE"}}
        filepath = save_result(result)

        assert os.path.exists(filepath)
        assert "data" + os.sep + "results" in filepath

        with open(filepath) as f:
            saved = json.load(f)
        assert saved["question"] == "Q?"

        os.remove(filepath)


# ============================================================
# run_detection (full pipeline, everything mocked)
# ============================================================

class TestRunDetectionFullPipeline:

    @patch("agents.orchestrator.reprompt_agent", return_value="CORRECTED_ANSWER: Canberra.")
    @patch("agents.orchestrator.llm")
    @patch("agents.orchestrator.confidence_agent", return_value="CONFIDENCE: 5\nREASON: contradicted")
    @patch("agents.orchestrator.consistency_agent", return_value="VERDICT: INCONSISTENT\nREASON: wrong\nKEY_CLAIM: Sydney")
    @patch("agents.orchestrator.fact_check_agent", return_value="VERDICT: FALSE\nREASON: wrong\nEVIDENCE: Canberra is correct")
    def test_hallucinated_answer_triggers_correction(
        self, mock_fact, mock_consistency, mock_confidence, mock_llm, mock_reprompt
    ):
        mock_llm.invoke.return_value = SimpleNamespace(
            content="FINAL_VERDICT: HALLUCINATED\nCONFIDENCE_SCORE: 95\nEXPLANATION: Contradicted by evidence."
        )

        result = run_detection(
            "What is the capital of Australia?",
            "The capital of Australia is Sydney.",
        )

        assert result["final"]["verdict"] == "HALLUCINATED"
        mock_reprompt.assert_called_once()
        assert "corrected_answer" in result
        assert "Canberra" in result["corrected_answer"]

        import glob
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for f in glob.glob(os.path.join(project_root, "data", "results", "result_*.json")):
            os.remove(f)

    @patch("agents.orchestrator.reprompt_agent")
    @patch("agents.orchestrator.llm")
    @patch("agents.orchestrator.confidence_agent", return_value=CONFIDENCE_HIGH)
    @patch("agents.orchestrator.consistency_agent", return_value=CONSISTENCY_CONSISTENT)
    @patch("agents.orchestrator.fact_check_agent", return_value=FACT_TRUE)
    def test_true_answer_does_not_trigger_correction(
        self, mock_fact, mock_consistency, mock_confidence, mock_llm, mock_reprompt
    ):
        mock_llm.invoke.return_value = FINAL_TRUE

        result = run_detection(
            "What is the capital of Australia?",
            "The capital of Australia is Canberra.",
        )

        assert result["final"]["verdict"] == "TRUE"
        mock_reprompt.assert_not_called()
        assert "corrected_answer" not in result

        import glob
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for f in glob.glob(os.path.join(project_root, "data", "results", "result_*.json")):
            os.remove(f)