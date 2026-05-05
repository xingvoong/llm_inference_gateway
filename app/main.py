import time
from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse, BatchChatRequest, BatchChatResponse
from app.router import route_request
from app.logger import init_db, log_request
from app import cache

app = FastAPI()
init_db()


def _handle_single(prompt: str, priority: str = None, max_cost: float = None) -> ChatResponse:
    provider, model_name, routing_reason = route_request(
        prompt=prompt,
        priority=priority,
        max_cost=max_cost,
    )

    cached = cache.get(prompt, model_name)
    if cached is not None:
        log_request(
            prompt=prompt,
            selected_model=model_name,
            routing_reason=f"cache_hit:{routing_reason}",
            latency_ms=0.0,
            response_length=len(cached),
        )
        return ChatResponse(response=cached, model_used=model_name)

    start = time.time()
    response = provider.generate_response(prompt)
    latency_ms = (time.time() - start) * 1000

    cache.set(prompt, model_name, response)

    log_request(
        prompt=prompt,
        selected_model=model_name,
        routing_reason=routing_reason,
        latency_ms=latency_ms,
        response_length=len(response),
    )

    return ChatResponse(response=response, model_used=model_name)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    return _handle_single(
        prompt=request.prompt,
        priority=request.priority,
        max_cost=request.max_cost,
    )


@app.post("/chat/batch", response_model=BatchChatResponse)
def chat_batch(request: BatchChatRequest):
    results = [
        _handle_single(
            prompt=prompt,
            priority=request.priority,
            max_cost=request.max_cost,
        )
        for prompt in request.prompts
    ]
    return BatchChatResponse(results=results)
