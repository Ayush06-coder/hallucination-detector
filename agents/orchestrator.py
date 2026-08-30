import json
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from agents.fact_check_agent import fact_check_agent
from agents.consistency_agent import consistency_agent
from agents.confidence_agent import confidence_agent
from agents.reprompt_agent import reprompt_agent

from utils.get_input import get_user_input
from utils.result_parser import (
    parse_fact_check,
    parse_consistency,
    parse_confidence,
    parse_final_verdict,
)

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# SAFE AGENT EXECUTION
# ============================================================

def run_safe_agent(
    agent_function,
    question,
    llm_response,
    name,
    *extra_args
):
    """
    Run an agent safely.

    Supports agents that require additional
    results from previous agents.
    """

    try:
        return agent_function(
            question,
            llm_response,
            *extra_args
        )

    except Exception as e:
        print(f"❌ {name} failed: {e}")
        return None


# ============================================================
# RUN ALL SPECIALIST AGENTS
# ============================================================

def run_all_agents(question, llm_response):

    print("\n🚀 Orchestrator starting...\n")

    # --------------------------------------------------------
    # FACT CHECK
    # --------------------------------------------------------

    fact_raw = run_safe_agent(
        fact_check_agent,
        question,
        llm_response,
        "Fact-Check Agent"
    )

    if fact_raw:
        fact_result = parse_fact_check(fact_raw)
    else:
        fact_result = {
            "verdict": "UNCERTAIN",
            "reason": "Fact-checking failed.",
            "evidence": "No reliable external evidence available.",
            "raw": ""
        }

    # --------------------------------------------------------
    # CONSISTENCY
    # --------------------------------------------------------

    consistency_raw = run_safe_agent(
        consistency_agent,
        question,
        llm_response,
        "Consistency Agent"
    )

    if consistency_raw:
        consistency_result = parse_consistency(
            consistency_raw
        )
    else:
        consistency_result = {
            "verdict": "UNCERTAIN",
            "reason": "Consistency analysis failed.",
            "key_claim": "",
            "raw": ""
        }

    # -------------------------
    # CONFIDENCE
    # -------------------------

    confidence_raw = run_safe_agent(
        confidence_agent,
        question,
        llm_response,
        "Confidence Agent",
        fact_result,
        consistency_result
    )

    if confidence_raw:
        confidence_result = parse_confidence(
            confidence_raw
        )
    else:
        confidence_result = {
            "score": 0,
            "reason": "Confidence analysis failed.",
            "raw": ""
        }

    results = {
        "fact_check": fact_result,
        "consistency": consistency_result,
        "confidence": confidence_result
    }

    return results


# ============================================================
# FINAL VERDICT
# ============================================================

def aggregate_verdict(
    question,
    llm_response,
    results
):

    print("\n⚖️ Aggregating final verdict...\n")

    fact_check = results["fact_check"]
    consistency = results["consistency"]
    confidence = results["confidence"]

    # Safely obtain confidence score
    confidence_score = confidence.get("score")

    if not isinstance(confidence_score, int):
        confidence_score = 0

    prompt = f"""
You are the FINAL JUDGE of a multi-agent
hallucination detection system.

Your job is to determine whether the ORIGINAL
LLM ANSWER is factually reliable.

==================================================
QUESTION
==================================================

{question}

==================================================
LLM ANSWER
==================================================

{llm_response}

==================================================
FACT-CHECK AGENT
==================================================

Verdict:
{fact_check.get("verdict")}

Reason:
{fact_check.get("reason")}

Evidence:
{fact_check.get("evidence")}

Raw result:
{fact_check.get("raw")}

==================================================
CONSISTENCY AGENT
==================================================

Verdict:
{consistency.get("verdict")}

Reason:
{consistency.get("reason")}

Key claim:
{consistency.get("key_claim")}

Raw result:
{consistency.get("raw")}

==================================================
CONFIDENCE AGENT
==================================================

Confidence score:
{confidence_score}/100

Reason:
{confidence.get("reason")}

Raw result:
{confidence.get("raw")}

==================================================
DECISION RULES
==================================================

Follow these rules carefully.

RULE 1 — FACTUAL EVIDENCE HAS THE HIGHEST PRIORITY

The Fact-Check Agent has access to external evidence.

If the Fact-Check Agent identifies credible evidence
that directly contradicts the LLM answer, treat the
answer as HALLUCINATED unless there is strong evidence
that the fact-check result itself is unreliable.

If the Fact-Check Agent confirms the answer with
credible evidence, this strongly supports TRUE.

--------------------------------------------------

RULE 2 — CONSISTENCY IS SECONDARY EVIDENCE

Use the Consistency Agent to determine whether the
answer logically and factually agrees with the question.

CONSISTENT supports TRUE.

INCONSISTENT supports HALLUCINATED.

However, consistency alone must NOT override strong
external evidence.

--------------------------------------------------

RULE 3 — CONFIDENCE IS SUPPORTING EVIDENCE

The Confidence Agent's score is NOT the final verdict.

A high confidence score does NOT mean the original
answer is true.

For example:

A wrong answer such as:
"The capital of Australia is Sydney."

may receive a confidence score of 95 because the claim
is clear and easy to evaluate.

Therefore:

CONFIDENCE = certainty of the evaluation,
NOT proof that the original answer is correct.

Use the confidence score only to determine how certain
you should be about the FINAL VERDICT.

--------------------------------------------------

RULE 4 — STRONG AGREEMENT

Examples:

Fact-Check = FALSE
Consistency = INCONSISTENT

=> HALLUCINATED

Fact-Check = TRUE
Consistency = CONSISTENT

=> TRUE

--------------------------------------------------

RULE 5 — DISAGREEMENT

If agents strongly disagree and external evidence is
insufficient or ambiguous, use:

UNCERTAIN

Do not invent evidence to force a verdict.

--------------------------------------------------

RULE 6 — FINAL CONFIDENCE

The final confidence score must represent YOUR confidence
in the final verdict.

It must NOT simply copy the Confidence Agent's score.

For example:

Fact-Check = FALSE
Consistency = INCONSISTENT
Confidence Agent = 95

Final verdict:
HALLUCINATED

Final confidence:
approximately 90-100

But if:

Fact-Check = UNCERTAIN
Consistency = CONSISTENT
Confidence Agent = 60

Final verdict:
UNCERTAIN

Final confidence:
approximately 50-70

==================================================
POSSIBLE VERDICTS
==================================================

TRUE
HALLUCINATED
UNCERTAIN

==================================================
OUTPUT FORMAT
==================================================

Reply ONLY in this exact format:

FINAL_VERDICT: [TRUE/HALLUCINATED/UNCERTAIN]
CONFIDENCE_SCORE: [integer 0-100]
EXPLANATION: [2-3 concise sentences]
"""

    try:

        result = llm.invoke(prompt)

        return parse_final_verdict(
            result.content
        )

    except Exception as e:

        print(f"❌ Final Judge failed: {e}")

        return {
            "verdict": "UNCERTAIN",
            "confidence_score": 0,
            "explanation": (
                "The final judge could not complete "
                "the verification."
            ),
            "raw": str(e)
        }


# ============================================================
# BUILD RESULT
# ============================================================

def build_result(
    question,
    llm_response,
    agent_results,
    final_result
):

    return {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": llm_response,
        "agents": agent_results,
        "final": final_result
    }


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    results_dir = os.path.join(
        project_root,
        "data",
        "results"
    )

    os.makedirs(
        results_dir,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    filename = f"result_{timestamp}.json"

    filepath = os.path.join(
        results_dir,
        filename
    )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False
        )

    return filepath


# ============================================================
# COMPLETE DETECTION PIPELINE
# ============================================================

def run_detection(
    question,
    llm_response
):

    # --------------------------------------------------------
    # RUN SPECIALIST AGENTS
    # --------------------------------------------------------

    results = run_all_agents(
        question,
        llm_response
    )

    # --------------------------------------------------------
    # FINAL JUDGE
    # --------------------------------------------------------

    final = aggregate_verdict(
        question,
        llm_response,
        results
    )

    # --------------------------------------------------------
    # BUILD COMPLETE RESULT
    # --------------------------------------------------------

    complete_result = build_result(
        question,
        llm_response,
        results,
        final
    )

    # --------------------------------------------------------
    # SAVE INITIAL RESULT
    # --------------------------------------------------------

    filepath = save_result(
        complete_result
    )

    print("\n" + "=" * 60)

    print(
        json.dumps(
            complete_result,
            indent=2,
            ensure_ascii=False
        )
    )

    print("=" * 60)

    print(
        f"\n💾 JSON saved to: {filepath}"
    )

    # --------------------------------------------------------
    # CORRECTION AGENT
    # --------------------------------------------------------

    if final.get("verdict") == "HALLUCINATED":

        try:

            corrected = reprompt_agent(
                question,
                llm_response,
                results["fact_check"]["raw"]
            )

            complete_result[
                "corrected_answer"
            ] = corrected

            # Save again with correction
            save_result(
                complete_result
            )

        except Exception as e:

            print(
                f"⚠️ Correction generation failed: {e}"
            )

            complete_result[
                "corrected_answer"
            ] = None

    return complete_result


# ============================================================
# TERMINAL MODE
# ============================================================

if __name__ == "__main__":

    question, llm_response = get_user_input()

    result = run_detection(
        question,
        llm_response
    )

    print("\nFINAL RESULT:")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )