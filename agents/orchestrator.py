from agents.fact_check_agent import fact_check_agent
from agents.consistency_agent import consistency_agent
from agents.confidence_agent import confidence_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from utils.get_input import get_user_input

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

def run_all_agents(question, llm_response):
    print("\n🚀 Orchestrator starting...\n")

    fact_result = fact_check_agent(question, llm_response)
    consistency_result = consistency_agent(question, llm_response)
    confidence_result = confidence_agent(question, llm_response)

    return {
        "fact_check": fact_result,
        "consistency": consistency_result,
        "confidence": confidence_result
    }

def aggregate_verdict(question, llm_response, results):
    print("\n⚖️ Aggregating final verdict...\n")

    prompt = f"""
    You are the final judge combining results from three agents.

    Question: {question}
    LLM Response: {llm_response}

    Fact-Check Agent said: {results['fact_check']}
    Consistency Agent said: {results['consistency']}
    Confidence Agent said: {results['confidence']}

    Based on ALL three agents combined, give a FINAL verdict.
    Reply in this exact format:
    FINAL_VERDICT: [TRUE/HALLUCINATED/UNCERTAIN]
    CONFIDENCE_SCORE: [number between 0-100]
    EXPLANATION: [2-3 sentence summary combining all agent findings]
    """
    result = llm.invoke(prompt)
    return result.content

if __name__ == "__main__":
    question, llm_response = get_user_input()

    results = run_all_agents(question, llm_response)
    final = aggregate_verdict(question, llm_response, results)

    print("=" * 50)
    print(final)
    print("=" * 50)