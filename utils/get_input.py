from utils.test_questions import TEST_QUESTIONS
import random


def get_user_input():
    print(
        "Enter your question "
        "(or press Enter to use a random test question):"
    )

    question = input("Question: ").strip()

    # ---------------------------------------------------------
    # RANDOM TEST QUESTION
    # ---------------------------------------------------------

    if question == "":
        sample = random.choice(TEST_QUESTIONS)

        print(
            f"\nUsing default question: "
            f"{sample['question']}"
        )

        print(
            f"Using default LLM response: "
            f"{sample['llm_response']}\n"
        )

        return (
            sample["question"],
            sample["llm_response"]
        )

    # ---------------------------------------------------------
    # CUSTOM QUESTION
    # ---------------------------------------------------------

    while True:

        llm_response = input(
            "LLM Response to check: "
        ).strip()

        if llm_response:
            break

        print(
            "⚠️ LLM response cannot be empty. "
            "Please enter an answer to check."
        )

    return question, llm_response