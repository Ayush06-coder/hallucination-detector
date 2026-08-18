from langchain_groq import ChatGroq
from ddgs import DDGS
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

def search_web(query):
    with DDGS() as ddgs:
        results = []
        for r in ddgs.text(query, max_results=3):
            results.append(r['body'])
        return " ".join(results)

def fact_check_agent(question, llm_response):
    print("🔍 Fact-Check Agent running...")
    
    search_query = f"fact check: {question}"
    evidence = search_web(search_query)
    
    prompt = f"""
    You are a fact-checking agent.
    
    Original Question: {question}
    LLM Response to check: {llm_response}
    Evidence from web search: {evidence}
    
    Based on the evidence, is the LLM response accurate?
    Reply in this exact format:
    VERDICT: [TRUE/FALSE/UNCERTAIN]
    REASON: [one sentence explanation]
    EVIDENCE: [key fact from search that supports your verdict]
    """
    
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    question = "Who invented the telephone?"
    llm_response = "Alexander Graham Bell invented the telephone in 1876."
    result = fact_check_agent(question, llm_response)
    print(result)