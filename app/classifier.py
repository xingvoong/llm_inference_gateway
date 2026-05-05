from transformers import pipeline

# Labels the classifier detects
TASK_LABELS = ["summarization", "question answering", "code generation", "general chat"]

# Loaded once at startup, not on every request
_classifier = None


def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
        )
    return _classifier


def classify_prompt(prompt: str) -> str:
    """
    Returns the most likely task type for a given prompt.
    One of: summarization, question answering, code generation, general chat
    """
    result = get_classifier()(prompt, candidate_labels=TASK_LABELS)
    return result["labels"][0]
