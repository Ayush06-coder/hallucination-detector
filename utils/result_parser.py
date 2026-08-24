import re


# ============================================================
# GENERIC VALUE EXTRACTION
# ============================================================

def extract_value(text, key):
    """
    Extract a single-line value from agent output.

    Example:
        VERDICT: TRUE

    Returns:
        TRUE
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


# ============================================================
# VERDICT NORMALIZATION
# ============================================================

def normalize_verdict(value):
    """
    Normalize specialist-agent verdicts.

    Supported values:
        TRUE
        FALSE
        UNCERTAIN
        CONSISTENT
        INCONSISTENT
        HALLUCINATED
    """

    if not value:
        return "UNCERTAIN"

    value = value.strip().upper()

    # Exact / prefix-based matching rather than
    # unrestricted substring matching.

    if value == "TRUE" or value.startswith("TRUE "):
        return "TRUE"

    if value == "FALSE" or value.startswith("FALSE "):
        return "FALSE"

    if value == "UNCERTAIN" or value.startswith("UNCERTAIN "):
        return "UNCERTAIN"

    if value == "CONSISTENT" or value.startswith("CONSISTENT "):
        return "CONSISTENT"

    if value == "INCONSISTENT" or value.startswith("INCONSISTENT "):
        return "INCONSISTENT"

    if value == "HALLUCINATED" or value.startswith("HALLUCINATED "):
        return "HALLUCINATED"

    return "UNCERTAIN"


# ============================================================
# FACT-CHECK PARSER
# ============================================================

def parse_fact_check(text):
    return {
        "verdict": normalize_verdict(
            extract_value(text, "VERDICT")
        ),
        "reason": (
            extract_value(text, "REASON")
            or "No reason provided."
        ),
        "evidence": (
            extract_value(text, "EVIDENCE")
            or "No reliable external evidence available."
        ),
        "raw": text or ""
    }


# ============================================================
# CONSISTENCY PARSER
# ============================================================

def parse_consistency(text):
    return {
        "verdict": normalize_verdict(
            extract_value(text, "VERDICT")
        ),
        "reason": (
            extract_value(text, "REASON")
            or "No reason provided."
        ),
        "key_claim": (
            extract_value(text, "KEY_CLAIM")
            or ""
        ),
        "raw": text or ""
    }


# ============================================================
# CONFIDENCE PARSER
# ============================================================

def parse_confidence(text):

    raw_score = extract_value(
        text,
        "CONFIDENCE"
    )

    try:

        match = re.search(
            r"\d+",
            raw_score or ""
        )

        score = (
            int(match.group())
            if match
            else None
        )

        if score is not None:
            score = max(
                0,
                min(100, score)
            )

    except (TypeError, ValueError):

        score = None

    return {
        "score": score,
        "reason": (
            extract_value(text, "REASON")
            or "No reason provided."
        ),
        "raw": text or ""
    }


# ============================================================
# FINAL VERDICT PARSER
# ============================================================

def parse_final_verdict(text):

    # --------------------------------------------------------
    # CONFIDENCE SCORE
    # --------------------------------------------------------

    raw_confidence = extract_value(
        text,
        "CONFIDENCE_SCORE"
    )

    try:

        match = re.search(
            r"\d+",
            raw_confidence or ""
        )

        confidence = (
            int(match.group())
            if match
            else None
        )

        if confidence is not None:
            confidence = max(
                0,
                min(100, confidence)
            )

    except (TypeError, ValueError):

        confidence = None

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    raw_verdict = extract_value(
        text,
        "FINAL_VERDICT"
    )

    if raw_verdict:

        raw_verdict = (
            raw_verdict
            .strip()
            .upper()
        )

        if (
            raw_verdict == "TRUE"
            or raw_verdict.startswith("TRUE ")
        ):
            verdict = "TRUE"

        elif (
            raw_verdict == "HALLUCINATED"
            or raw_verdict.startswith("HALLUCINATED ")
        ):
            verdict = "HALLUCINATED"

        elif (
            raw_verdict == "UNCERTAIN"
            or raw_verdict.startswith("UNCERTAIN ")
        ):
            verdict = "UNCERTAIN"

        else:
            verdict = "UNCERTAIN"

    else:

        verdict = "UNCERTAIN"

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return {
        "verdict": verdict,
        "confidence_score": confidence,
        "explanation": (
            extract_value(
                text,
                "EXPLANATION"
            )
            or "No explanation provided."
        ),
        "raw": text or ""
    }