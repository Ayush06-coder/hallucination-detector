from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

def consistency_agent(question, llm_response):
    print("🔄 Consistency Agent running...")

    rephrase_prompt = f"""
    Rephrase this question in a different way, keeping the same meaning:
    "{question}"
    Only return the rephrased question, nothing else.
    """
    rephrased_question = llm.invoke(rephrase_prompt).content.strip()

    second_answer_prompt = f"Answer this question concisely: {rephrased_question}"
    second_answer = llm.invoke(second_answer_prompt).content.strip()

    compare_prompt = f"""
    You are a consistency-checking agent.

    Original question: {question}
    Original answer: {llm_response}

    Rephrased question: {rephrased_question}
    New answer: {second_answer}

    Do both answers agree on the same facts, or do they contradict each other?
    Reply in this exact format:
    VERDICT: [CONSISTENT/INCONSISTENT]
    REASON: [one sentence explanation]
    """
    result = llm.invoke(compare_prompt)
    return result.content

if __name__ == "__main__":
    question = "Who invented the telephone?"
    llm_response = "Alexander Graham Bell invented the telephone in 1876."
    result = consistency_agent(question, llm_response)
    print(result)