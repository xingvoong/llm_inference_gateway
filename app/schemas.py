from pydantic import BaseModel
from typing import Optional
from enum import Enum


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ChatRequest(BaseModel):
    prompt: str
    priority: Optional[Priority] = None
    max_cost: Optional[float] = None


class ChatResponse(BaseModel):
    response: str
    model_used: str


class BatchChatRequest(BaseModel):
    prompts: list[str]
    priority: Optional[Priority] = None
    max_cost: Optional[float] = None


class BatchChatResponse(BaseModel):
    results: list[ChatResponse]
