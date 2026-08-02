from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

def confidence_agent(question, llm_response):
    print("📊 Confidence Agent running...")

    prompt = f"""
    You are a confidence-scoring agent.

    Question: {question}
    Answer given: {llm_response}

    Rate how confident you are that this answer is factually correct, on a scale of 0-100.
    Consider: is this a well-established fact, a contested topic, or something uncertain?

    Reply in this exact format:
    CONFIDENCE: [number between 0-100]
    REASON: [one sentence explanation of why you gave this score]
    """
    result = llm.invoke(prompt)
    return result.content

if __name__ == "__main__":
    question = "Who invented the telephone?"
    llm_response = "Alexander Graham Bell invented the telephone in 1876."
    result = confidence_agent(question, llm_response)
    print(result)