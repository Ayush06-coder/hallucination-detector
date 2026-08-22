from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b"
)

def reprompt_agent(question, original_answer, fact_check_evidence):
    print("\n🔧 Re-prompting for corrected answer...\n")

    prompt = f"""
    You previously answered a question, but it may contain inaccurate information.

    Original question: {question}
    Your previous answer: {original_answer}
    Verified evidence found: {fact_check_evidence}

    Based on the verified evidence, provide a corrected, accurate answer.
    Keep it concise, 1-2 sentences.
    Reply in this exact format:
    CORRECTED_ANSWER: [your corrected answer]
    """
    result = llm.invoke(prompt)
    return result.content