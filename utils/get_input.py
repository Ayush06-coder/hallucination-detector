from utils.test_questions import TEST_QUESTIONS
import random

def get_user_input():
    print("Enter your question (or press Enter to use a random test question):")
    question = input("Question: ").strip()

    if question == "":
        sample = random.choice(TEST_QUESTIONS)
        print(f"\nUsing default question: {sample['question']}")
        print(f"Using default LLM response: {sample['llm_response']}\n")
        return sample['question'], sample['llm_response']

    llm_response = input("LLM Response to check: ").strip()
    return question, llm_response