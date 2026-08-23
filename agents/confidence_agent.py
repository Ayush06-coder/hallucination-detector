from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()


llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0
)


def confidence_agent(
    question,
    llm_response,
    fact_check_result=None,
    consistency_result=None
):
    print("📊 Confidence Agent running...")

    fact_check_result = fact_check_result or {}
    consistency_result = consistency_result or {}

    fact_verdict = fact_check_result.get(
        "verdict",
        "UNCERTAIN"
    )

    fact_reason = fact_check_result.get(
        "reason",
        ""
    )

    fact_evidence = fact_check_result.get(
        "evidence",
        ""
    )

    consistency_verdict = consistency_result.get(
        "verdict",
        "UNCERTAIN"
    )

    consistency_reason = consistency_result.get(
        "reason",
        ""
    )

    prompt = f"""
You are the confidence-scoring specialist in a
multi-agent hallucination detection system.

Your job is to estimate how confident we should be
that the ORIGINAL AI-GENERATED ANSWER is factually
correct after considering the independent verification
agents.

QUESTION:
{question}

AI-GENERATED ANSWER:
{llm_response}

FACT-CHECK AGENT:
Verdict: {fact_verdict}
Reason: {fact_reason}
Evidence: {fact_evidence}

CONSISTENCY AGENT:
Verdict: {consistency_verdict}
Reason: {consistency_reason}

IMPORTANT RULES:

1. This score represents confidence in the
   FACTUAL CORRECTNESS of the original answer.

2. Give the greatest weight to reliable external
   evidence from the Fact-Check Agent.

3. If Fact-Check is FALSE and provides clear
   contradictory evidence, confidence that the
   original answer is correct should be LOW.

4. If Fact-Check is TRUE and Consistency is
   CONSISTENT, confidence that the original answer
   is correct should normally be HIGH.

5. If Fact-Check is UNCERTAIN or evidence is weak,
   do not give an artificially high score.

6. If Fact-Check and Consistency disagree,
   reduce confidence.

7. Do not confuse confidence in the AI answer with
   confidence in the detector's final verdict.

8. Do not invent evidence.

Use the following general interpretation:

90-100:
Very strong evidence that the original answer
is factually correct.

70-89:
Strong evidence that the original answer
is probably correct.

40-69:
Mixed, incomplete, or ambiguous evidence.

10-39:
Strong reason to doubt the original answer.

0-9:
Very strong evidence that the original answer
is factually incorrect.

Reply ONLY in this exact format:

CONFIDENCE: [integer 0-100]
REASON: [one concise explanation]
"""

    try:

        result = llm.invoke(prompt)

        return result.content.strip()

    except Exception as e:

        print(
            f"❌ Confidence Agent failed: {e}"
        )

        return """
CONFIDENCE: 0
REASON: Confidence analysis could not be completed.
"""


if __name__ == "__main__":

    question = (
        "What is the capital of Australia?"
    )

    llm_response = (
        "The capital of Australia is Sydney."
    )

    fact_check_result = {
        "verdict": "FALSE",
        "reason": (
            "The evidence states that Canberra, "
            "not Sydney, is the capital of Australia."
        ),
        "evidence": (
            "Canberra is the capital city of Australia."
        )
    }

    consistency_result = {
        "verdict": "INCONSISTENT",
        "reason": (
            "The answer incorrectly states "
            "that Sydney is the capital."
        )
    }

    result = confidence_agent(
        question,
        llm_response,
        fact_check_result,
        consistency_result
    )

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)