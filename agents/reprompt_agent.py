from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()


# ---------------------------------------------------------
# LLM
# ---------------------------------------------------------

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0
)


# ---------------------------------------------------------
# REPROMPT AGENT
# ---------------------------------------------------------

def reprompt_agent(
    question,
    original_answer,
    fact_check_evidence
):

    print("\n🔧 Re-prompting for corrected answer...\n")

    prompt = f"""
You are a correction agent in a
multi-agent hallucination detection system.

The original AI-generated answer may contain
factual errors.

Your task is to produce a corrected answer using
the verified evidence provided by the Fact-Check Agent.

ORIGINAL QUESTION:
{question}

ORIGINAL AI-GENERATED ANSWER:
{original_answer}

VERIFIED FACT-CHECK EVIDENCE:
{fact_check_evidence}


=========================================================
CORRECTION RULES
=========================================================

1. Correct only claims that are unsupported,
   inaccurate, or contradicted by the evidence.

2. Use the verified evidence as the primary basis
   for the correction.

3. Do NOT invent facts that are not supported
   by the evidence.

4. If the evidence is insufficient to determine
   the correct answer, clearly state that the answer
   cannot be reliably determined.

5. Do not repeat the original incorrect claim.

6. Keep the corrected answer concise and direct.

7. Do not mention the internal agents, prompts,
   hallucination detection system, or verification
   process in the corrected answer.

8. If the original answer is already supported by
   the evidence, preserve its factual meaning rather
   than unnecessarily changing it.


=========================================================
OUTPUT FORMAT
=========================================================

Reply ONLY in this exact format:

CORRECTED_ANSWER: [corrected answer]
"""

    try:

        result = llm.invoke(prompt)

        return result.content.strip()

    except Exception as e:

        print(
            f"❌ Re-prompt Agent failed: {e}"
        )

        return (
            "CORRECTED_ANSWER: "
            "A reliable corrected answer could not "
            "be generated."
        )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    question = (
        "What is the capital of Australia?"
    )

    original_answer = (
        "The capital of Australia is Sydney."
    )

    fact_check_evidence = (
        "Canberra is the capital city of Australia."
    )

    result = reprompt_agent(
        question,
        original_answer,
        fact_check_evidence
    )

    print("\n" + "=" * 60)
    print(result)
    print("=" * 60)