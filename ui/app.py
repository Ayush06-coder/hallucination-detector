import chainlit as cl
import sys
import os
import random
import re

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.test_questions import TEST_QUESTIONS
from utils.logger import log_result
from agents.orchestrator import (
    run_all_agents,
    aggregate_verdict,
    infer_question_from_answer,
)
from agents.reprompt_agent import reprompt_agent


# ============================================================
# STARTER CARDS
# ============================================================

@cl.set_starters
async def set_starters():
    samples = random.sample(
        TEST_QUESTIONS,
        min(4, len(TEST_QUESTIONS))
    )

    return [
        cl.Starter(
            label=s["question"][:45]
            + ("..." if len(s["question"]) > 45 else ""),
            message=(
                f"__SAMPLE_CASE__\n"
                f"Question: {s['question']}\n"
                f"Answer: {s['llm_response']}"
            ),
        )
        for s in samples
    ]


# ============================================================
# CHAT START
# ============================================================

@cl.on_chat_start
async def start():

    cl.user_session.set("stage", "answer")

    await cl.Message(
        content=(
            "# 🔍 Hallucination Detector\n\n"
            "### Multi-Agent AI Verification Engine\n\n"
            "Paste an **AI-generated answer** below and I'll independently "
            "verify it using three specialist agents.\n\n"
            "---\n\n"
            "### How it works\n\n"
            "🔎 **Fact-Check Agent** — searches for external evidence\n\n"
            "🔄 **Consistency Agent** — independently tests the claim\n\n"
            "🎯 **Confidence Agent** — evaluates certainty\n\n"
            "⚖️ **Final Judge** — combines all findings\n\n"
            "---\n\n"
            "### 🚀 Quick start\n"
            "Paste an LLM answer, or choose one of the sample cases above."
        )
    ).send()


# ============================================================
# MESSAGE HANDLER
# ============================================================

@cl.on_message
async def main(message: cl.Message):

    text = message.content.strip()

    if not text:
        await cl.Message(
            content="⚠️ Please paste an LLM-generated answer first."
        ).send()
        return

    # --------------------------------------------------------
    # SAMPLE CARD
    # --------------------------------------------------------

    if text.startswith("__SAMPLE_CASE__"):

        question_match = re.search(
            r"Question:\s*(.*?)\nAnswer:",
            text,
            re.DOTALL,
        )

        answer_match = re.search(
            r"Answer:\s*(.*)",
            text,
            re.DOTALL,
        )

        if question_match and answer_match:

            question = question_match.group(1).strip()
            llm_response = answer_match.group(1).strip()

            await run_detection(
                question,
                llm_response,
                sample=True,
            )

            return

    # --------------------------------------------------------
    # NORMAL ANSWER-ONLY MODE
    # --------------------------------------------------------

    await run_detection(
        question=None,
        llm_response=text,
        sample=False,
    )


# ============================================================
# DETECTION PIPELINE
# ============================================================

async def run_detection(question, llm_response, sample=False):

    # --------------------------------------------------------
    # Infer question when user only supplied an answer
    # --------------------------------------------------------

    if not question:

        async with cl.Step(
            name="Claim Context",
            type="llm",
        ) as step:

            step.output = (
                "Identifying the factual question and claims "
                "contained in the answer..."
            )

            question = infer_question_from_answer(llm_response)

    # --------------------------------------------------------
    # Show input summary
    # --------------------------------------------------------

    source_label = (
        "Sample case"
        if sample
        else "Custom LLM answer"
    )

    await cl.Message(
        content=(
            f"## 🧪 {source_label}\n\n"
            f"**Question context:**\n> {question}\n\n"
            f"**Answer being verified:**\n> {llm_response}"
        )
    ).send()

    # --------------------------------------------------------
    # Actual multi-agent execution
    # --------------------------------------------------------

    try:

        async with cl.Step(
            name="🔎 Fact-Check Agent",
            type="tool",
        ) as step:

            step.output = "Searching the web for independent evidence..."

            results = {}

            # Fact check first
            from agents.fact_check_agent import fact_check_agent

            results["fact_check"] = fact_check_agent(
                question,
                llm_response,
            )

        async with cl.Step(
            name="🔄 Consistency Agent",
            type="tool",
        ) as step:

            step.output = (
                "Rephrasing the question and independently "
                "checking whether the answer remains consistent..."
            )

            from agents.consistency_agent import consistency_agent

            results["consistency"] = consistency_agent(
                question,
                llm_response,
            )

        async with cl.Step(
            name="🎯 Confidence Agent",
            type="tool",
        ) as step:

            step.output = (
                "Evaluating confidence in the factual correctness..."
            )

            from agents.confidence_agent import confidence_agent

            results["confidence"] = confidence_agent(
                question,
                llm_response,
            )

    except Exception as e:

        await cl.Message(
            content=(
                "## ⚠️ Agent Pipeline Error\n\n"
                f"`{type(e).__name__}: {str(e)}`\n\n"
                "The verification pipeline stopped before producing "
                "a final verdict."
            )
        ).send()

        print(f"\n❌ Agent pipeline error: {e}")
        return

    # --------------------------------------------------------
    # Final aggregation
    # --------------------------------------------------------

    async with cl.Step(
        name="⚖️ Final Judge",
        type="llm",
    ) as step:

        step.output = (
            "Combining evidence from all three specialist agents..."
        )

        final = aggregate_verdict(
            question,
            llm_response,
            results,
        )

    # --------------------------------------------------------
    # Parse verdict
    # --------------------------------------------------------

    final_upper = final.upper()

    if "HALLUCINATED" in final_upper:
        verdict = "HALLUCINATED"
        icon = "🔴"
        title = "Likely Hallucinated"

    elif "UNCERTAIN" in final_upper:
        verdict = "UNCERTAIN"
        icon = "🟡"
        title = "Uncertain"

    else:
        verdict = "TRUE"
        icon = "🟢"
        title = "Likely Accurate"

    # --------------------------------------------------------
    # Result message
    # --------------------------------------------------------

    await cl.Message(
        content=(
            f"# {icon} {title}\n\n"
            f"### Final Assessment\n\n"
            f"{final}\n\n"
            "---\n\n"
            "## 🤖 Agent Findings\n\n"
            "### 🔎 Fact-Check Agent\n"
            f"{results['fact_check']}\n\n"
            "---\n\n"
            "### 🔄 Consistency Agent\n"
            f"{results['consistency']}\n\n"
            "---\n\n"
            "### 🎯 Confidence Agent\n"
            f"{results['confidence']}"
        )
    ).send()

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    log_result(
        question,
        llm_response,
        final,
    )

    # --------------------------------------------------------
    # Corrected answer
    # --------------------------------------------------------

    if verdict == "HALLUCINATED":

        async with cl.Step(
            name="📝 Correction Agent",
            type="llm",
        ) as step:

            step.output = (
                "Generating a corrected answer using the "
                "fact-checking evidence..."
            )

            corrected = reprompt_agent(
                question,
                llm_response,
                results["fact_check"],
            )

        await cl.Message(
            content=(
                "# ✨ Suggested Correction\n\n"
                f"{corrected}"
            )
        ).send()

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    await cl.Message(
        content=(
            "---\n\n"
            "### 🔄 Ready for another check?\n\n"
            "Paste another LLM answer or choose a sample case."
        )
    ).send()