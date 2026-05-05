from app.providers import BaseProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.huggingface_provider import HuggingFaceProvider
from app.learned_router import predict_model, is_trained_model_available
from app.classifier import classify_prompt

FAST_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_MODEL = "gpt-4"
BEST_MODEL = "gpt-4"

LOW_COST_THRESHOLD = 0.01


def _get_provider(model_name: str) -> BaseProvider:
    if "gpt" in model_name:
        return OpenAIProvider(model=model_name)
    return HuggingFaceProvider(model=model_name)


def route_request(prompt: str, priority: str = None, max_cost: float = None) -> tuple[BaseProvider, str, str]:
    # Rule 1: high priority always gets the best model
    if priority == "high":
        return OpenAIProvider(model=BEST_MODEL), BEST_MODEL, "priority==high"

    # Rule 2: strict budget gets the cheapest model
    if max_cost is not None and max_cost < LOW_COST_THRESHOLD:
        return HuggingFaceProvider(model=FAST_MODEL), FAST_MODEL, "max_cost<0.01"

    # Rule 3: use learned router if trained model exists, else fall back to zero-shot
    if is_trained_model_available():
        model_name = predict_model(prompt)
        return _get_provider(model_name), model_name, "learned_router"
    else:
        task = classify_prompt(prompt)
        if task in ("code generation", "summarization"):
            return HuggingFaceProvider(model=FAST_MODEL), FAST_MODEL, f"zero_shot:{task}"
        return OpenAIProvider(model=DEFAULT_MODEL), DEFAULT_MODEL, f"zero_shot:{task}"
