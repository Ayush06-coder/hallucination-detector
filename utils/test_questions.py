import json
import os

def load_test_questions():
    path = os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(path, "r") as f:
        return json.load(f)

TEST_QUESTIONS = load_test_questions()