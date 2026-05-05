"""
Generate synthetic training data for the learned router.

Each example is a (prompt, model) pair representing what model
should handle that type of request. This simulates what real
logged traffic would look like after thousands of requests.

Diversity matters: clean, similar examples produce an overfit model.
Real traffic is messy — short questions, typos, ambiguous phrasing.
The examples here try to capture that variety.
"""

import json
import random
import os

# fmt: off
EXAMPLES = {
    "gpt-4": [
        # conceptual questions
        "What are the ethical implications of AI?",
        "Explain the difference between supervised and unsupervised learning",
        "What is the capital of France?",
        "How does inflation affect purchasing power?",
        "Can you explain quantum entanglement?",
        "What caused World War 1?",
        "How does the stock market work?",
        "What is the meaning of life?",
        "Explain the theory of relativity",
        "What are the pros and cons of remote work?",
        "How does the human immune system work?",
        "What is machine learning?",
        "Explain blockchain technology",
        "What is the difference between a virus and a bacteria?",
        "How do vaccines work?",
        "What is climate change?",
        "Explain natural language processing",
        "What is the Turing test?",
        "How does GPS work?",
        "What is the difference between RAM and ROM?",
        "Can you recommend a book on stoicism?",
        "What is the Socratic method?",
        "How does photosynthesis work?",
        "What is a black hole?",
        "What is a neural network?",
        "How does the internet work?",
        "What is the difference between ML and AI?",
        "What is the big bang theory?",
        # short/ambiguous questions that aren't coding or summarization
        "Why do stars twinkle?",
        "What is democracy?",
        "Why is the sky blue?",
        "What is a republic?",
        "How does sleep work?",
        "Why do we dream?",
        "What is consciousness?",
        "How do airplanes fly?",
        "What is gravity?",
        "Why is water wet?",
        "What is the difference between weather and climate?",
        "How does memory work in the brain?",
        "What is evolution?",
        "Why do leaves change color?",
        "How do magnets work?",
        "What is dark matter?",
        "Why do we age?",
        "How does the economy work?",
        "What is inflation?",
        "Why is gold valuable?",
        # conversational
        "Can you help me understand this concept?",
        "I'm confused about how recursion works",
        "Can you explain this to me like I'm five?",
        "What should I know about investing?",
        "I want to learn about history",
        "Tell me something interesting about space",
        "What's the difference between a CEO and a CTO?",
        "How do I get better at critical thinking?",
        "What makes a good leader?",
        "How do I deal with stress?",
    ],
    "mistralai/Mistral-7B-Instruct-v0.3": [
        # coding tasks
        "Write a Python function to reverse a string",
        "Write a SQL query to find duplicate rows",
        "How do I center a div in CSS?",
        "Write a bash script to backup files",
        "Implement a binary search algorithm",
        "Write a regex to validate email addresses",
        "How do I read a file line by line in Python?",
        "Write a function to flatten a nested list",
        "Write a JavaScript function to sort an array",
        "How do I merge two dictionaries in Python?",
        "Write a Docker compose file for a web app",
        "Implement a linked list in Python",
        "Write a unit test for this function",
        "How do I handle exceptions in Python?",
        "Write a function to check if a number is prime",
        "How do I reverse a linked list?",
        "Write a Python class for a bank account",
        "How do I connect to a database in Python?",
        "Write a function to find the longest substring",
        "Write a quicksort implementation",
        "How do I use async/await in Python?",
        "Write a function to validate a password",
        "How do I parse JSON in Python?",
        "Write a REST API endpoint in FastAPI",
        "Fix this bug in my Python code",
        "How do I implement a hash map?",
        "Write a function to detect palindromes",
        "How do I do a left join in SQL?",
        "Write a recursive fibonacci function",
        "How do I use decorators in Python?",
        # summarization tasks
        "Summarize this article about climate change",
        "Give me a summary of the key points",
        "Condense this report into bullet points",
        "What are the main takeaways from this text?",
        "Summarize the key findings of this paper",
        "Give me the main points of this document",
        "Make this shorter",
        "TL;DR this for me",
        "Can you summarize what I just wrote?",
        "Extract the key ideas from this paragraph",
        "Shorten this email",
        "Summarize this meeting transcript",
        "What are the highlights from this article?",
        "Give me a one-paragraph summary",
        "Condense this into three bullet points",
    ],
}
# fmt: on


def generate():
    data = []
    for model, prompts in EXAMPLES.items():
        for prompt in prompts:
            data.append({"prompt": prompt, "model": model})

    random.shuffle(data)

    os.makedirs("data", exist_ok=True)
    output_path = "data/training_data.json"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data)} training examples → {output_path}")
    for model in EXAMPLES:
        count = sum(1 for d in data if d["model"] == model)
        print(f"  {model}: {count} examples")


if __name__ == "__main__":
    generate()
