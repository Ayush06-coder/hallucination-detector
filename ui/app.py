import chainlit as cl
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from utils.test_questions import TEST_QUESTIONS
from utils.logger import log_result
from agents.orchestrator import run_all_agents, aggregate_verdict
from agents.reprompt_agent import reprompt_agent

@cl.set_starters
async def set_starters():
    samples = random.sample(TEST_QUESTIONS, 4)
    return [
        cl.Starter(
            label=s["question"][:40] + ("..." if len(s["question"]) > 40 else ""),
            message=f"Question: {s['question']}\nAnswer: {s['llm_response']}"
        )
        for s in samples
    ]

@cl.on_chat_start
async def start():
    cl.user_session.set("stage", "question")
    await cl.Message(
        content="🔍 **Welcome to the Hallucination Detector**\n\nI'll check an AI-generated answer for factual accuracy using 3 specialist agents.\n\n**First, what's the question that was asked?**"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    stage = cl.user_session.get("stage")
    text = message.content.strip()

    # Handle starter clicks — combined "Question: ... Answer: ..." format
    if "Question:" in text and "Answer:" in text:
        question = text.split("Question:")[1].split("Answer:")[0].strip()
        llm_response = text.split("Answer:")[1].strip()
        await run_detection(question, llm_response)
        cl.user_session.set("stage", "question")
        return

    if stage == "question":
        cl.user_session.set("question", text)
        cl.user_session.set("stage", "answer")
        await cl.Message(content="Got it ✅\n\n**Now paste the LLM's answer that you want me to check:**").send()
        return

    if stage == "answer":
        question = cl.user_session.get("question")
        llm_response = text
        await run_detection(question, llm_response)
        cl.user_session.set("stage", "question")
        return


async def run_detection(question, llm_response):
    await cl.Message(content=f"🔎 Checking:\n> **Q:** {question}\n> **A:** {llm_response}").send()

    async with cl.Step(name="Fact-Check Agent", type="tool") as step:
        step.output = "Searching the web for evidence..."

    async with cl.Step(name="Consistency Agent", type="tool") as step:
        step.output = "Re-asking the question differently..."

    async with cl.Step(name="Confidence Agent", type="tool") as step:
        step.output = "Scoring certainty of the claim..."

    results = run_all_agents(question, llm_response)
    final = aggregate_verdict(question, llm_response, results)

    elements = [
        cl.Text(name="Fact-Check Agent", content=results["fact_check"], display="inline"),
        cl.Text(name="Consistency Agent", content=results["consistency"], display="inline"),
        cl.Text(name="Confidence Agent", content=results["confidence"], display="inline"),
    ]

    if "HALLUCINATED" in final:
        header = "## ❌ Verdict: Likely hallucinated"
    elif "UNCERTAIN" in final:
        header = "## ⚠️ Verdict: Uncertain"
    else:
        header = "## ✅ Verdict: Likely accurate"

    await cl.Message(content=f"{header}\n\n{final}", elements=elements).send()

    log_result(question, llm_response, final)

    if "HALLUCINATED" in final:
        corrected = reprompt_agent(question, llm_response, results['fact_check'])
        await cl.Message(content=f"### 📝 Corrected answer\n{corrected}").send()

    await cl.Message(content="---\n**Ask another question whenever you're ready.**").send()