"""
Tests for the individual agents (fact_check, consistency,
confidence, reprompt) with the Groq LLM call mocked.

No API key or network access is used — llm.invoke() is patched
to return a canned response, so these test the agent's prompt
construction, error handling, and return-value plumbing only.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def fake_llm_response(text):
    """Mimic the object returned by ChatGroq().invoke() — has a .content attr."""
    return SimpleNamespace(content=text)


# ============================================================
# fact_check_agent
# ============================================================

class TestFactCheckAgent:

    @patch("agents.fact_check_agent.search_web", return_value="Canberra is the capital of Australia.")
    @patch("agents.fact_check_agent.llm")
    def test_returns_llm_content_on_success(self, mock_llm, mock_search):
        from agents.fact_check_agent import fact_check_agent

        mock_llm.invoke.return_value = fake_llm_response(
            "VERDICT: FALSE\nREASON: Contradicted.\nEVIDENCE: Canberra is correct."
        )

        result = fact_check_agent(
            "What is the capital of Australia?",
            "The capital of Australia is Sydney.",
        )

        assert "VERDICT: FALSE" in result
        mock_search.assert_called_once()
        mock_llm.invoke.assert_called_once()

    @patch("agents.fact_check_agent.search_web", return_value="WEB_SEARCH_UNAVAILABLE: ...")
    @patch("agents.fact_check_agent.llm")
    def test_falls_back_gracefully_when_llm_raises(self, mock_llm, mock_search):
        from agents.fact_check_agent import fact_check_agent

        mock_llm.invoke.side_effect = Exception("groq API down")

        result = fact_check_agent("Any question?", "Any answer.")

        assert "UNCERTAIN" in result
        assert "could not be completed" in result.lower()


# ============================================================
# consistency_agent
# ============================================================

class TestConsistencyAgent:

    @patch("agents.consistency_agent.llm")
    def test_returns_llm_content_on_success(self, mock_llm):
        from agents.consistency_agent import consistency_agent

        mock_llm.invoke.return_value = fake_llm_response(
            "VERDICT: CONSISTENT\nREASON: No contradictions.\nKEY_CLAIM: Bell invented the telephone."
        )

        result = consistency_agent(
            "Who invented the telephone?",
            "Alexander Graham Bell invented the telephone in 1876.",
        )

        assert "CONSISTENT" in result

    @patch("agents.consistency_agent.llm")
    def test_falls_back_gracefully_when_llm_raises(self, mock_llm):
        from agents.consistency_agent import consistency_agent

        mock_llm.invoke.side_effect = Exception("timeout")

        result = consistency_agent("Any question?", "Any answer.")

        assert "UNCERTAIN" in result


# ============================================================
# confidence_agent
# ============================================================

class TestConfidenceAgent:

    @patch("agents.confidence_agent.llm")
    def test_returns_llm_content_on_success(self, mock_llm):
        from agents.confidence_agent import confidence_agent

        mock_llm.invoke.return_value = fake_llm_response(
            "CONFIDENCE: 10\nREASON: Fact-check contradicts the answer."
        )

        result = confidence_agent(
            "What is the capital of Australia?",
            "The capital of Australia is Sydney.",
            fact_check_result={"verdict": "FALSE", "reason": "wrong", "evidence": "Canberra"},
            consistency_result={"verdict": "INCONSISTENT", "reason": "wrong"},
        )

        assert "CONFIDENCE: 10" in result

    @patch("agents.confidence_agent.llm")
    def test_works_with_no_prior_results(self, mock_llm):
        """confidence_agent must not crash when called standalone (no fact/consistency yet)."""
        from agents.confidence_agent import confidence_agent

        mock_llm.invoke.return_value = fake_llm_response("CONFIDENCE: 50\nREASON: Unclear.")

        result = confidence_agent("Q?", "A.")
        assert "CONFIDENCE: 50" in result

    @patch("agents.confidence_agent.llm")
    def test_falls_back_gracefully_when_llm_raises(self, mock_llm):
        from agents.confidence_agent import confidence_agent

        mock_llm.invoke.side_effect = Exception("rate limited")

        result = confidence_agent("Q?", "A.")
        assert "CONFIDENCE: 0" in result


# ============================================================
# reprompt_agent
# ============================================================

class TestRepromptAgent:

    @patch("agents.reprompt_agent.llm")
    def test_returns_corrected_answer_on_success(self, mock_llm):
        from agents.reprompt_agent import reprompt_agent

        mock_llm.invoke.return_value = fake_llm_response(
            "CORRECTED_ANSWER: The capital of Australia is Canberra."
        )

        result = reprompt_agent(
            "What is the capital of Australia?",
            "The capital of Australia is Sydney.",
            "Canberra is the capital city of Australia.",
        )

        assert "Canberra" in result

    @patch("agents.reprompt_agent.llm")
    def test_falls_back_gracefully_when_llm_raises(self, mock_llm):
        from agents.reprompt_agent import reprompt_agent

        mock_llm.invoke.side_effect = Exception("groq down")

        result = reprompt_agent("Q?", "A.", "evidence")
        assert "CORRECTED_ANSWER:" in result
        assert "could not be generated" in result.lower()