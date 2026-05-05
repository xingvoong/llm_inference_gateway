import os
import requests
from dotenv import load_dotenv
from app.providers import BaseProvider

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider(BaseProvider):
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def generate_response(self, prompt: str) -> str:
        response = requests.post(
            OPENROUTER_API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        if not response.ok:
            raise Exception(f"OpenRouter error {response.status_code}: {response.text}")
        return response.json()["choices"][0]["message"]["content"]
