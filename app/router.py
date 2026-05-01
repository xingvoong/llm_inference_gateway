from app.providers import BaseProvider
from app.providers.openai_provider import OpenAIProvider
from app.providers.huggingface_provider import HuggingFaceProvider

CHEAPEST_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_MODEL = "gpt-4"
SMALL_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LOW_COST_THRESHOLD = 0.01
SHORT_PROMPT_LENGTH = 100


def route_request(prompt: str, priority: str = None, max_cost: float = None) -> tuple[BaseProvider, str]:
    if priority == "high":
        return OpenAIProvider(model="gpt-4"), "gpt-4"

    if max_cost is not None and max_cost < LOW_COST_THRESHOLD:
        return HuggingFaceProvider(model=CHEAPEST_MODEL), CHEAPEST_MODEL

    if len(prompt) < SHORT_PROMPT_LENGTH:
        return HuggingFaceProvider(model=SMALL_MODEL), SMALL_MODEL

    return OpenAIProvider(model=DEFAULT_MODEL), DEFAULT_MODEL
