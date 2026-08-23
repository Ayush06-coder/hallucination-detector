from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0
)


def consistency_agent(question, llm_response):
    print("🔄 Consistency Agent running...")

    prompt = f"""
You are a consistency-verification specialist in a
multi-agent hallucination detection system.

Question:
{question}

AI-Generated Answer:
{llm_response}

Analyze the answer carefully.

Check:

1. Does the answer directly answer the question?
2. What factual claims does it make?
3. Are the claims internally consistent?
4. Are there contradictions?
5. Are there impossible or obviously suspicious statements?
6. Does the answer contain unnecessary claims that could be incorrect?

IMPORTANT:

This agent should evaluate logical and internal consistency.
Do not pretend to have external evidence.

Use:

CONSISTENT
- if the answer directly answers the question and contains
  no obvious contradiction.

INCONSISTENT
- if the answer contains contradictory, impossible,
  or clearly incorrect internal claims.

UNCERTAIN
- if there is not enough information to determine consistency.

Reply ONLY in this format:

VERDICT: [CONSISTENT/INCONSISTENT/UNCERTAIN]
REASON: [one concise explanation]
KEY_CLAIM: [the most important claim evaluated]
"""

    try:
        result = llm.invoke(prompt)
        return result.content

    except Exception as e:
        print(f"❌ Consistency Agent failed: {e}")

        return """
VERDICT: UNCERTAIN
REASON: Consistency analysis could not be completed.
KEY_CLAIM: Unable to determine.
"""


if __name__ == "__main__":
    question = "Who invented the telephone?"
    llm_response = "Alexander Graham Bell invented the telephone in 1876."

    result = consistency_agent(question, llm_response)
    print(result)