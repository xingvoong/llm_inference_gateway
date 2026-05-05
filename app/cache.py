"""
Exact-match response cache.

Keyed on (prompt, model_name). If the same prompt hits the same model
again, we return the stored response instead of calling the provider.
"""

_cache: dict[tuple[str, str], str] = {}


def get(prompt: str, model_name: str) -> str | None:
    return _cache.get((prompt, model_name))


def set(prompt: str, model_name: str, response: str) -> None:
    _cache[(prompt, model_name)] = response


def size() -> int:
    return len(_cache)


def clear() -> None:
    _cache.clear()
