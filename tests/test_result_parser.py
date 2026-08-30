"""
Unit tests for utils/result_parser.py

These test pure text-parsing logic and require no API key,
no network access, and no mocking.
"""

from utils.result_parser import (
    extract_value,
    normalize_verdict,
    parse_fact_check,
    parse_consistency,
    parse_confidence,
    parse_final_verdict,
)


# ============================================================
# extract_value
# ============================================================

def test_extract_value_basic():
    text = "VERDICT: TRUE\nREASON: Well established fact."
    assert extract_value(text, "VERDICT") == "TRUE"
    assert extract_value(text, "REASON") == "Well established fact."


def test_extract_value_case_insensitive_key():
    text = "verdict: FALSE"
    assert extract_value(text, "VERDICT") == "FALSE"


def test_extract_value_missing_key_returns_none():
    text = "REASON: no verdict line here"
    assert extract_value(text, "VERDICT") is None


def test_extract_value_empty_text_returns_none():
    assert extract_value("", "VERDICT") is None
    assert extract_value(None, "VERDICT") is None


def test_extract_value_strips_whitespace():
    text = "VERDICT:    TRUE   \n"
    assert extract_value(text, "VERDICT") == "TRUE"


# ============================================================
# normalize_verdict
# ============================================================

def test_normalize_verdict_exact_matches():
    assert normalize_verdict("TRUE") == "TRUE"
    assert normalize_verdict("FALSE") == "FALSE"
    assert normalize_verdict("UNCERTAIN") == "UNCERTAIN"
    assert normalize_verdict("CONSISTENT") == "CONSISTENT"
    assert normalize_verdict("INCONSISTENT") == "INCONSISTENT"
    assert normalize_verdict("HALLUCINATED") == "HALLUCINATED"


def test_normalize_verdict_lowercase_input():
    assert normalize_verdict("true") == "TRUE"
    assert normalize_verdict("false") == "FALSE"


def test_normalize_verdict_with_trailing_text():
    assert normalize_verdict("TRUE because evidence supports it") == "TRUE"


def test_normalize_verdict_none_or_empty_defaults_uncertain():
    assert normalize_verdict(None) == "UNCERTAIN"
    assert normalize_verdict("") == "UNCERTAIN"


def test_normalize_verdict_garbage_defaults_uncertain():
    assert normalize_verdict("BANANA") == "UNCERTAIN"


def test_normalize_verdict_no_false_positive_substring_match():
    assert normalize_verdict("INCONSISTENT") == "INCONSISTENT"
    assert normalize_verdict("INCONSISTENT because of X") == "INCONSISTENT"


# ============================================================
# parse_fact_check
# ============================================================

def test_parse_fact_check_full():
    text = (
        "VERDICT: FALSE\n"
        "REASON: Canberra is the capital, not Sydney.\n"
        "EVIDENCE: Canberra is the capital city of Australia."
    )
    result = parse_fact_check(text)
    assert result["verdict"] == "FALSE"
    assert result["reason"] == "Canberra is the capital, not Sydney."
    assert result["evidence"] == "Canberra is the capital city of Australia."
    assert result["raw"] == text


def test_parse_fact_check_missing_fields_get_defaults():
    result = parse_fact_check("VERDICT: TRUE")
    assert result["verdict"] == "TRUE"
    assert result["reason"] == "No reason provided."
    assert result["evidence"] == "No reliable external evidence available."


def test_parse_fact_check_empty_text():
    result = parse_fact_check("")
    assert result["verdict"] == "UNCERTAIN"
    assert result["raw"] == ""


# ============================================================
# parse_consistency
# ============================================================

def test_parse_consistency_full():
    text = (
        "VERDICT: INCONSISTENT\n"
        "REASON: Contradicts itself.\n"
        "KEY_CLAIM: Sydney is the capital."
    )
    result = parse_consistency(text)
    assert result["verdict"] == "INCONSISTENT"
    assert result["reason"] == "Contradicts itself."
    assert result["key_claim"] == "Sydney is the capital."


def test_parse_consistency_missing_key_claim_defaults_empty_string():
    result = parse_consistency("VERDICT: CONSISTENT\nREASON: Fine.")
    assert result["key_claim"] == ""


# ============================================================
# parse_confidence
# ============================================================

def test_parse_confidence_basic():
    text = "CONFIDENCE: 85\nREASON: Strong evidence."
    result = parse_confidence(text)
    assert result["score"] == 85
    assert result["reason"] == "Strong evidence."


def test_parse_confidence_clamps_above_100():
    text = "CONFIDENCE: 150\nREASON: overshoot"
    result = parse_confidence(text)
    assert result["score"] == 100


def test_parse_confidence_clamps_below_0():
    text = "CONFIDENCE: -10\nREASON: undershoot"
    result = parse_confidence(text)
    assert 0 <= result["score"] <= 100


def test_parse_confidence_no_number_returns_none():
    text = "CONFIDENCE: unknown\nREASON: could not determine"
    result = parse_confidence(text)
    assert result["score"] is None


def test_parse_confidence_missing_entirely():
    result = parse_confidence("")
    assert result["score"] is None
    assert result["reason"] == "No reason provided."


# ============================================================
# parse_final_verdict
# ============================================================

def test_parse_final_verdict_full():
    text = (
        "FINAL_VERDICT: HALLUCINATED\n"
        "CONFIDENCE_SCORE: 95\n"
        "EXPLANATION: The fact-check agent found clear contradicting evidence."
    )
    result = parse_final_verdict(text)
    assert result["verdict"] == "HALLUCINATED"
    assert result["confidence_score"] == 95
    assert "contradicting evidence" in result["explanation"]


def test_parse_final_verdict_unknown_verdict_defaults_uncertain():
    text = "FINAL_VERDICT: MAYBE\nCONFIDENCE_SCORE: 50\nEXPLANATION: unclear"
    result = parse_final_verdict(text)
    assert result["verdict"] == "UNCERTAIN"


def test_parse_final_verdict_missing_everything():
    result = parse_final_verdict("")
    assert result["verdict"] == "UNCERTAIN"
    assert result["confidence_score"] is None
    assert result["explanation"] == "No explanation provided."


def test_parse_final_verdict_clamps_score():
    text = "FINAL_VERDICT: TRUE\nCONFIDENCE_SCORE: 999\nEXPLANATION: x"
    result = parse_final_verdict(text)
    assert result["confidence_score"] == 100