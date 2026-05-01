from abc import ABC, abstractmethod


class BaseProvider(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass
