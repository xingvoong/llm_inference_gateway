from app.providers import BaseProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.huggingface_provider import HuggingFaceProvider
from app.classifier import classify_prompt

# Model tiers — swap these for real models when ready
FAST_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"   # small, fast, cheap
DEFAULT_MODEL = "gpt-4"                               # balanced default
BEST_MODEL = "gpt-4"                                  # high priority requests

LOW_COST_THRESHOLD = 0.01
SHORT_PROMPT_LENGTH = 100


def route_request(prompt: str, priority: str = None, max_cost: float = None) -> tuple[BaseProvider, str]:
    # Rule 1: high priority always gets the best model
    if priority == "high":
        return OpenAIProvider(model=BEST_MODEL), BEST_MODEL

    # Rule 2: strict budget gets the cheapest model
    if max_cost is not None and max_cost < LOW_COST_THRESHOLD:
        return HuggingFaceProvider(model=FAST_MODEL), FAST_MODEL

    # Rule 3: classify the prompt to detect task type
    # Phase 1 used len(prompt) < 100 — this is smarter
    task = classify_prompt(prompt)

    if task == "code generation":
        return HuggingFaceProvider(model=FAST_MODEL), FAST_MODEL

    if task == "summarization":
        return HuggingFaceProvider(model=FAST_MODEL), FAST_MODEL

    # question answering and general chat → default model
    return OpenAIProvider(model=DEFAULT_MODEL), DEFAULT_MODEL
