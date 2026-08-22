import re


def extract_value(text, key):
    """
    Extract a single-line value from agent output.

    Example:
    VERDICT: TRUE
    -> TRUE
    """
    pattern = rf"{key}\s*:\s*(.+)"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip()

    return None


def parse_fact_check(text):
    return {
        "verdict": extract_value(text, "VERDICT"),
        "reason": extract_value(text, "REASON"),
        "evidence": extract_value(text, "EVIDENCE"),
        "raw": text
    }


def parse_consistency(text):
    return {
        "verdict": extract_value(text, "VERDICT"),
        "reason": extract_value(text, "REASON"),
        "key_claim": extract_value(text, "KEY_CLAIM"),
        "raw": text
    }


def parse_confidence(text):
    score = extract_value(text, "CONFIDENCE")

    try:
        score = int(score)
    except (TypeError, ValueError):
        score = None

    return {
        "score": score,
        "reason": extract_value(text, "REASON"),
        "raw": text
    }


def parse_final_verdict(text):
    confidence = extract_value(text, "CONFIDENCE_SCORE")

    try:
        confidence = int(confidence)
    except (TypeError, ValueError):
        confidence = None

    return {
        "verdict": extract_value(text, "FINAL_VERDICT"),
        "confidence_score": confidence,
        "explanation": extract_value(text, "EXPLANATION"),
        "raw": text
    }