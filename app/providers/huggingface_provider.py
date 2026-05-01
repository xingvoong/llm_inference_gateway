from app.providers import BaseProvider


class HuggingFaceProvider(BaseProvider):
    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        # TODO: replace with real HuggingFace API call in Phase 2
        # from huggingface_hub import InferenceClient
        # client = InferenceClient(model=self.model)
        # return client.text_generation(prompt, max_new_tokens=500)
        return f"[mock huggingface/{self.model}] response to: {prompt[:50]}"
