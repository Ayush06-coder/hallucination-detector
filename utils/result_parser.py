import re


def extract_value(text, key):
    """
    Extract a single-line value from agent output.

    Example:
        VERDICT: TRUE
    ->  TRUE
    """

    if not text:
        return None

    pattern = rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE | re.MULTILINE
    )

    if match:
        return match.group(1).strip()

    return None


def normalize_verdict(value):
    if not value:
        return "UNCERTAIN"

    value = value.strip().upper()

    if "HALLUCINATED" in value:
        return "HALLUCINATED"

    if "INCONSISTENT" in value:
        return "INCONSISTENT"

    if value.startswith("TRUE"):
        return "TRUE"

    if value.startswith("FALSE"):
        return "FALSE"

    if value.startswith("CONSISTENT"):
        return "CONSISTENT"

    if value.startswith("UNCERTAIN"):
        return "UNCERTAIN"

    return value


def parse_fact_check(text):
    return {
        "verdict": normalize_verdict(
            extract_value(text, "VERDICT")
        ),
        "reason": extract_value(text, "REASON"),
        "evidence": extract_value(text, "EVIDENCE"),
        "raw": text
    }


def parse_consistency(text):
    return {
        "verdict": normalize_verdict(
            extract_value(text, "VERDICT")
        ),
        "reason": extract_value(text, "REASON"),
        "key_claim": extract_value(text, "KEY_CLAIM"),
        "raw": text
    }


def parse_confidence(text):
    score = extract_value(text, "CONFIDENCE")

    try:
        match = re.search(r"\d+", score or "")
        score = int(match.group()) if match else None

        if score is not None:
            score = max(0, min(100, score))

    except (TypeError, ValueError):
        score = None

    return {
        "score": score,
        "reason": extract_value(text, "REASON"),
        "raw": text
    }


def parse_final_verdict(text):
    confidence = extract_value(
        text,
        "CONFIDENCE_SCORE"
    )

    try:
        match = re.search(r"\d+", confidence or "")
        confidence = int(match.group()) if match else None

        if confidence is not None:
            confidence = max(0, min(100, confidence))

    except (TypeError, ValueError):
        confidence = None

    verdict = extract_value(
        text,
        "FINAL_VERDICT"
    )

    if verdict:
        verdict = verdict.strip().upper()

        if verdict.startswith("TRUE"):
            verdict = "TRUE"
        elif verdict.startswith("HALLUCINATED"):
            verdict = "HALLUCINATED"
        elif verdict.startswith("UNCERTAIN"):
            verdict = "UNCERTAIN"

    else:
        verdict = "UNCERTAIN"

    return {
        "verdict": verdict,
        "confidence_score": confidence,
        "explanation": extract_value(
            text,
            "EXPLANATION"
        ),
        "raw": text
    }