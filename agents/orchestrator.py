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


def run_all_agents(question, llm_response):
    print("\n🚀 Orchestrator starting...\n")

    fact_raw = fact_check_agent(question, llm_response)
    consistency_raw = consistency_agent(question, llm_response)
    confidence_raw = confidence_agent(question, llm_response)

    results = {
        "fact_check": parse_fact_check(fact_raw),
        "consistency": parse_consistency(consistency_raw),
        "confidence": parse_confidence(confidence_raw)
    }

    return results


def aggregate_verdict(question, llm_response, results):
    print("\n⚖️ Aggregating final verdict...\n")

    prompt = f"""
You are the final judge in a multi-agent hallucination detection system.

Question:
{question}

LLM Response:
{llm_response}

FACT-CHECK AGENT:
{json.dumps(results["fact_check"], indent=2)}

CONSISTENCY AGENT:
{json.dumps(results["consistency"], indent=2)}

CONFIDENCE AGENT:
{json.dumps(results["confidence"], indent=2)}

Combine all findings.

Important:
- Do not blindly trust any single agent.
- Give more weight to concrete external evidence.
- Consider contradictions between agents.
- Distinguish TRUE, HALLUCINATED, and UNCERTAIN.
- The confidence score should represent your confidence in the final verdict.

Reply ONLY in this format:

FINAL_VERDICT: [TRUE/HALLUCINATED/UNCERTAIN]
CONFIDENCE_SCORE: [integer 0-100]
EXPLANATION: [2-3 concise sentences]
"""

    result = llm.invoke(prompt)

    return parse_final_verdict(result.content)


def build_result(question, llm_response, agent_results, final_result):
    return {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": llm_response,
        "agents": agent_results,
        "final": final_result
    }


def save_result(result):
    os.makedirs("data/results", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result_{timestamp}.json"

    filepath = os.path.join("data", "results", filename)

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    return filepath


def run_detection(question, llm_response):
    results = run_all_agents(question, llm_response)

    final = aggregate_verdict(
        question,
        llm_response,
        results
    )

    complete_result = build_result(
        question,
        llm_response,
        results,
        final
    )

    filepath = save_result(complete_result)

    print("\n" + "=" * 60)
    print(json.dumps(complete_result, indent=2, ensure_ascii=False))
    print("=" * 60)

    print(f"\n💾 JSON saved to: {filepath}")

    if final.get("verdict") == "HALLUCINATED":
        corrected = reprompt_agent(
            question,
            llm_response,
            results["fact_check"]["raw"]
        )

        complete_result["corrected_answer"] = corrected

        save_result(complete_result)

    return complete_result


if __name__ == "__main__":
    question, llm_response = get_user_input()

    result = run_detection(
        question,
        llm_response
    )

    print("\nFINAL RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))