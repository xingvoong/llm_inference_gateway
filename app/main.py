import time
from fastapi import FastAPI
from app.schemas import ChatRequest, ChatResponse
from app.router import route_request
from app.logger import init_db, log_request

app = FastAPI()
init_db()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    provider, model_name = route_request(
        prompt=request.prompt,
        priority=request.priority,
        max_cost=request.max_cost,
    )

    start = time.time()
    response = provider.generate_response(request.prompt)
    latency_ms = (time.time() - start) * 1000

    log_request(
        prompt=request.prompt,
        selected_model=model_name,
        latency_ms=latency_ms,
        response_length=len(response),
    )

    return ChatResponse(response=response, model_used=model_name)
