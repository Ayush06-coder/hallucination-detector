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
You are a consistency-verification specialist in a multi-agent
hallucination detection system.

Your task is to independently analyze whether the provided answer
is logically and factually consistent with the question.

Question:
{question}

Answer:
{llm_response}

Analyze:
1. What factual claims does the answer make?
2. Do those claims directly answer the question?
3. Are there contradictions, impossible statements, or suspicious claims?
4. Does the answer contain internal inconsistencies?
5. Based on your analysis, determine whether the answer is consistent.

Reply ONLY in this format:

VERDICT: [CONSISTENT/INCONSISTENT/UNCERTAIN]
REASON: [one concise explanation]
KEY_CLAIM: [the most important claim you evaluated]
"""

    result = llm.invoke(prompt)
    return result.content


if __name__ == "__main__":
    question = "Who invented the telephone?"
    llm_response = "Alexander Graham Bell invented the telephone in 1876."

    result = consistency_agent(question, llm_response)
    print(result)