import gradio as gr

from agents.orchestrator import run_detection


# ============================================================
# SAMPLE CASES
# ============================================================

SAMPLE_CASES = [
    {
        "title": "Berlin Wall",
        "category": "HISTORY",
        "question": "What year did the Berlin Wall fall?",
        "answer": "The Berlin Wall fell in 1987.",
        "style": "history",
    },
    {
        "title": "Human Heart",
        "category": "SCIENCE",
        "question": "How many chambers does the human heart have?",
        "answer": "The human heart has 5 chambers.",
        "style": "science",
    },
    {
        "title": "Mount Everest",
        "category": "GEOGRAPHY",
        "question": "What is the height of Mount Everest?",
        "answer": "Mount Everest is 10,000 meters tall.",
        "style": "geography",
    },
    {
        "title": "Sun Temperature",
        "category": "SPACE",
        "question": "What is the temperature at the surface of the Sun?",
        "answer": "The Sun's surface temperature is 10,000 °C.",
        "style": "space",
    },
    {
        "title": "Python Creator",
        "category": "TECHNOLOGY",
        "question": "Who created the Python programming language?",
        "answer": "Python was created by Guido van Rossum in 1989.",
        "style": "technology",
    },
]


# ============================================================
# BACKEND FUNCTION
# ============================================================

def detect_hallucination(question, answer):

    if not answer or not answer.strip():
        return (
            "## ⚠️ No Answer Provided\n\n"
            "Please enter an AI-generated answer.",
            "",
            "",
            "",
            ""
        )

    if not question or not question.strip():
        question = "What factual claims are made in this answer?"

    try:

        result = run_detection(
            question.strip(),
            answer.strip()
        )

        final = result.get("final", {})
        agents = result.get("agents", {})

        verdict = final.get("verdict", "UNKNOWN")
        confidence = final.get("confidence_score", "N/A")
        explanation = final.get(
            "explanation",
            "No explanation available."
        )

        # ----------------------------------------------------
        # VERDICT
        # ----------------------------------------------------

        if verdict == "HALLUCINATED":
            status = "🔴 HALLUCINATED"
            verdict_class = "danger"

        elif verdict == "UNCERTAIN":
            status = "🟡 UNCERTAIN"
            verdict_class = "warning"

        elif verdict == "TRUE":
            status = "🟢 TRUE"
            verdict_class = "success"

        else:
            status = f"⚪ {verdict}"
            verdict_class = "neutral"

        # ----------------------------------------------------
        # AGENT RESULTS
        # ----------------------------------------------------

        fact_check = agents.get("fact_check", {})
        consistency = agents.get("consistency", {})
        confidence_agent = agents.get("confidence", {})

        fact_result = (
            f"### 🔎 Fact-Check Agent\n\n"
            f"**Verdict**\n\n"
            f"`{fact_check.get('verdict', 'N/A')}`\n\n"
            f"**Reason**\n\n"
            f"{fact_check.get('reason', 'N/A')}\n\n"
            f"**Evidence**\n\n"
            f"{fact_check.get('evidence', 'N/A')}"
        )

        consistency_result = (
            f"### 🔄 Consistency Agent\n\n"
            f"**Verdict**\n\n"
            f"`{consistency.get('verdict', 'N/A')}`\n\n"
            f"**Reason**\n\n"
            f"{consistency.get('reason', 'N/A')}\n\n"
            f"**Key Claim**\n\n"
            f"{consistency.get('key_claim', 'N/A')}"
        )

        confidence_result = (
            f"### 🎯 Confidence Agent\n\n"
            f"**Score**\n\n"
            f"`{confidence_agent.get('score', 'N/A')} / 5`\n\n"
            f"**Reason**\n\n"
            f"{confidence_agent.get('reason', 'N/A')}"
        )

        # ----------------------------------------------------
        # CORRECTED ANSWER
        # ----------------------------------------------------

        corrected_answer = result.get(
            "corrected_answer",
            "No correction was generated."
        )

        # ----------------------------------------------------
        # FINAL RESULT
        # ----------------------------------------------------

        final_result = (
            f'<div class="verdict-card {verdict_class}">'
            f'<div class="verdict-top">'
            f'<div>'
            f'<div class="small-label">FINAL VERDICT</div>'
            f'<div class="verdict-title">{status}</div>'
            f'</div>'
            f'<div class="confidence-box">'
            f'<div class="small-label">CONFIDENCE</div>'
            f'<div class="confidence-value">{confidence}%</div>'
            f'</div>'
            f'</div>'
            f'<div class="result-divider"></div>'
            f'<div class="small-label">EXPLANATION</div>'
            f'<div class="explanation">{explanation}</div>'
            f'<div class="small-label correction-label">'
            f'SUGGESTED CORRECTION'
            f'</div>'
            f'<div class="correction">{corrected_answer}</div>'
            f'</div>'
        )

        return (
            final_result,
            fact_result,
            consistency_result,
            confidence_result,
            corrected_answer
        )

    except Exception as e:

        error = (
            "## ❌ Pipeline Error\n\n"
            f"`{type(e).__name__}: {str(e)}`"
        )

        return error, "", "", "", ""


# ============================================================
# SAMPLE LOADER
# ============================================================

def load_sample(index):

    sample = SAMPLE_CASES[index]

    return (
        sample["question"],
        sample["answer"]
    )


# ============================================================
# CSS
# ============================================================

CSS = """

/* ============================================================
   GLOBAL
   ============================================================ */

body {
    background:
        radial-gradient(
            circle at 50% -20%,
            rgba(99, 102, 241, 0.12),
            transparent 45%
        ),
        #090c13 !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 24px 50px !important;
}


/* ============================================================
   HEADER
   ============================================================ */

.hero {
    text-align: center;
    padding: 25px 10px 20px;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin-bottom: 6px;
    background: linear-gradient(
        90deg,
        #a78bfa,
        #60a5fa,
        #f472b6
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    font-size: 17px;
    color: #cbd5e1;
    margin-bottom: 7px;
}

.hero-description {
    font-size: 13px;
    color: #8993a5;
}


/* ============================================================
   PIPELINE
   ============================================================ */

.pipeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin: 12px auto 26px;
    padding: 13px;
    border: 1px solid #252b38;
    border-radius: 14px;
    background: rgba(15, 19, 29, 0.8);
}

.pipeline-item {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #d7dce5;
    font-size: 13px;
    font-weight: 600;
    padding: 7px 10px;
}

.pipeline-icon {
    width: 31px;
    height: 31px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 9px;
    background: #1c2330;
    font-size: 16px;
}

.pipeline-arrow {
    color: #596273;
    font-size: 18px;
}


/* ============================================================
   SECTION CONTAINERS
   ============================================================ */

.section {
    border: 1px solid #252b38;
    border-radius: 15px;
    background: rgba(14, 18, 27, 0.86);
    padding: 20px;
    margin-bottom: 18px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
}

.section-subtitle {
    font-size: 12px;
    color: #7e8798;
    margin-top: 3px;
}


/* ============================================================
   SAMPLE CARDS
   ============================================================ */

.sample-card {
    min-height: 145px !important;
    border: 1px solid #2a3140 !important;
    border-radius: 12px !important;
    background: linear-gradient(
        145deg,
        #141a25,
        #0e131d
    ) !important;
    transition: all 0.2s ease !important;
    text-align: left !important;
}

.sample-card:hover {
    border-color: #6366f1 !important;
    transform: translateY(-2px);
    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.25);
}

.sample-card button {
    height: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
}

.sample-history {
    border-top: 2px solid #22c55e !important;
}

.sample-science {
    border-top: 2px solid #38bdf8 !important;
}

.sample-geography {
    border-top: 2px solid #a855f7 !important;
}

.sample-space {
    border-top: 2px solid #facc15 !important;
}

.sample-technology {
    border-top: 2px solid #ef4444 !important;
}


/* ============================================================
   INPUT AREA
   ============================================================ */

.input-panel {
    border: 1px solid #2a3140;
    border-radius: 13px;
    padding: 18px;
    background: #10151f;
}

.input-title {
    font-size: 15px;
    font-weight: 700;
    color: #e5e7eb;
    margin-bottom: 12px;
}

textarea,
input {
    background: #111722 !important;
    border: 1px solid #303746 !important;
    border-radius: 9px !important;
}

textarea:focus,
input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.12) !important;
}


/* ============================================================
   VERIFY BUTTON
   ============================================================ */

.verify-button {
    background: linear-gradient(
        90deg,
        #4f46e5,
        #7c3aed,
        #db2777
    ) !important;
    border: none !important;
    border-radius: 9px !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    height: 48px !important;
    margin-top: 10px !important;
}

.verify-button:hover {
    filter: brightness(1.1);
}


/* ============================================================
   RESULT CARD
   ============================================================ */

.verdict-card {
    border: 1px solid #303746;
    border-radius: 13px;
    padding: 22px;
    background: #10151f;
    margin-bottom: 15px;
}

.verdict-card.danger {
    border-left: 4px solid #ef4444;
}

.verdict-card.warning {
    border-left: 4px solid #f59e0b;
}

.verdict-card.success {
    border-left: 4px solid #22c55e;
}

.verdict-card.neutral {
    border-left: 4px solid #64748b;
}

.verdict-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.small-label {
    font-size: 10px;
    letter-spacing: 1px;
    font-weight: 700;
    color: #7f899a;
}

.verdict-title {
    font-size: 28px;
    font-weight: 800;
    margin-top: 4px;
}

.confidence-box {
    text-align: right;
}

.confidence-value {
    font-size: 25px;
    font-weight: 800;
    color: #60a5fa;
    margin-top: 3px;
}

.result-divider {
    height: 1px;
    background: #282f3c;
    margin: 18px 0;
}

.explanation {
    color: #cbd5e1;
    line-height: 1.6;
    margin-top: 7px;
}

.correction-label {
    margin-top: 18px;
}

.correction {
    margin-top: 7px;
    color: #d1fae5;
    line-height: 1.6;
}


/* ============================================================
   AGENT CARDS
   ============================================================ */

.agent-card {
    border: 1px solid #292f3c;
    border-radius: 12px;
    padding: 17px;
    min-height: 230px;
    background: #10151f;
}

.agent-card h3 {
    margin-top: 0 !important;
    font-size: 15px !important;
}

.agent-card p {
    color: #aab3c2;
    line-height: 1.55;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer {
    text-align: center;
    color: #626c7c;
    font-size: 11px;
    padding: 18px;
}

"""


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="Hallucination Detector",
    theme=gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate"
    ),
    css=CSS
) as app:

    # ========================================================
    # HEADER
    # ========================================================

    gr.HTML(
        """
        <div class="hero">

            <div class="hero-title">
                🔍 Hallucination Detector
            </div>

            <div class="hero-subtitle">
                Multi-Agent AI Verification Engine
            </div>

            <div class="hero-description">
                Detect factual hallucinations using independent
                verification agents and evidence-based analysis.
            </div>

        </div>
        """
    )


    # ========================================================
    # PIPELINE
    # ========================================================

    gr.HTML(
        """
        <div class="pipeline">

            <div class="pipeline-item">
                <div class="pipeline-icon">🔎</div>
                Fact-Check
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                <div class="pipeline-icon">🔄</div>
                Consistency
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                <div class="pipeline-icon">🎯</div>
                Confidence
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                <div class="pipeline-icon">⚖️</div>
                Final Judge
            </div>

            <div class="pipeline-arrow">→</div>

            <div class="pipeline-item">
                <div class="pipeline-icon">📝</div>
                Correction
            </div>

        </div>
        """
    )


    # ========================================================
    # SAMPLE QUESTIONS
    # ========================================================

    with gr.Column(elem_classes="section"):

        gr.HTML(
            """
            <div class="section-title">
                🧪 Try a Sample Case
            </div>

            <div class="section-subtitle">
                Click any case to automatically load a question
                and AI-generated answer.
            </div>
            """
        )

        with gr.Row():

            sample_buttons = []

            for i, sample in enumerate(SAMPLE_CASES):

                button = gr.Button(
                    value=(
                        f"📚 {sample['title']}\n"
                        f"{sample['category']}\n\n"
                        f"{sample['question']}"
                    ),
                    elem_classes=[
                        "sample-card",
                        f"sample-{sample['style']}"
                    ]
                )

                sample_buttons.append(button)


    # ========================================================
    # INPUT SECTION
    # ========================================================

    with gr.Column(elem_classes="section"):

        gr.HTML(
            """
            <div class="input-title">
                ✏️ Enter Your Own Test
            </div>
            """
        )

        with gr.Row():

            question = gr.Textbox(
                label="Question (Optional)",
                placeholder=(
                    "e.g. What year did the Berlin Wall fall?"
                ),
                lines=3,
                scale=1
            )

            answer = gr.Textbox(
                label="AI-Generated Answer",
                placeholder=(
                    "Paste the AI-generated answer here..."
                ),
                lines=6,
                scale=1
            )

        check_button = gr.Button(
            "🔍  Verify Answer",
            variant="primary",
            elem_classes="verify-button"
        )

        gr.Markdown(
            "<div style='text-align:center; color:#626c7c; "
            "font-size:11px;'>"
            "🔒 Your input is processed locally through the "
            "verification pipeline."
            "</div>"
        )


    # ========================================================
    # RESULTS
    # ========================================================

    with gr.Column(elem_classes="section"):

        gr.HTML(
            """
            <div class="section-title">
                📊 Verification Results
            </div>
            """
        )

        final_result = gr.HTML(
            value="""
            <div class="verdict-card neutral">
                <div class="small-label">
                    FINAL VERDICT
                </div>

                <div style="
                    font-size:22px;
                    font-weight:700;
                    margin-top:8px;
                    color:#64748b;
                ">
                    Waiting for analysis...
                </div>

                <div style="
                    margin-top:10px;
                    color:#7f899a;
                ">
                    Enter an answer above and click
                    <b>Verify Answer</b>.
                </div>
            </div>
            """
        )

        with gr.Row():

            with gr.Column(elem_classes="agent-card"):

                fact_output = gr.Markdown(
                    "### 🔎 Fact-Check Agent\n\n"
                    "Waiting for analysis..."
                )

            with gr.Column(elem_classes="agent-card"):

                consistency_output = gr.Markdown(
                    "### 🔄 Consistency Agent\n\n"
                    "Waiting for analysis..."
                )

            with gr.Column(elem_classes="agent-card"):

                confidence_output = gr.Markdown(
                    "### 🎯 Confidence Agent\n\n"
                    "Waiting for analysis..."
                )


        corrected_answer = gr.Markdown(
            "### ✨ Suggested Correction\n\n"
            "A correction will appear here when a hallucination "
            "is detected."
        )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        """
        <div class="footer">
            Multi-Agent Hallucination Detection
            &nbsp;•&nbsp;
            Built with Python + Gradio
            &nbsp;•&nbsp;
            Stay skeptical, stay informed.
        </div>
        """
    )


    # ========================================================
    # SAMPLE BUTTON ACTIONS
    # ========================================================

    for i, button in enumerate(sample_buttons):

        button.click(
            fn=lambda index=i: load_sample(index),
            inputs=None,
            outputs=[question, answer]
        )


    # ========================================================
    # VERIFY ACTION
    # ========================================================

    check_button.click(
        fn=detect_hallucination,
        inputs=[
            question,
            answer
        ],
        outputs=[
            final_result,
            fact_output,
            consistency_output,
            confidence_output,
            corrected_answer
        ]
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.launch()