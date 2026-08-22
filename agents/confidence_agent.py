from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
    temperature=0
)


def confidence_agent(question, llm_response):
    print("📊 Confidence Agent running...")

    prompt = f"""
You are a confidence-scoring specialist in a hallucination detection system.

Question:
{question}

Answer:
{llm_response}

Evaluate how confident we should be that the answer is factually correct.

Consider:
- Whether the claim is well-established
- Whether the answer contains specific factual claims
- Whether the claims appear verifiable
- Whether uncertainty or ambiguity exists
- Whether the answer contains suspicious precision

Give a score from 0 to 100.

Reply ONLY in this format:

CONFIDENCE: [integer 0-100]
REASON: [one concise explanation]
"""

    result = llm.invoke(prompt)
    return result.content


if __name__ == "__main__":
    question = "Who invented the telephone?"
    llm_response = "Alexander Graham Bell invented the telephone in 1876."

    result = confidence_agent(question, llm_response)
    print(result)