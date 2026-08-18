import json
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "history.json")

def log_result(question, llm_response, final_verdict):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "llm_response": llm_response,
        "verdict": final_verdict
    }

    history = []
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []

    history.append(entry)

    with open(LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)