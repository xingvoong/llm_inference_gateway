from app.providers import BaseProvider


class OpenAIProvider(BaseProvider):
    def __init__(self, model: str = "gpt-4"):
        self.model = model

    def generate_response(self, prompt: str) -> str:
        # TODO: replace with real OpenAI API call in Phase 2
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.chat.completions.create(
        #     model=self.model,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        # return response.choices[0].message.content
        return f"[mock openai/{self.model}] response to: {prompt[:50]}"
