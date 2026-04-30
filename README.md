# LLM Inference Gateway

A FastAPI service that routes prompts to different LLM providers using rule-based logic, and logs every request for future ML training.

---

## Why this exists

Not every prompt needs GPT-4. A routing layer lets you pick the right model for the job — balancing cost, speed, and quality. The logging system captures every decision, which becomes training data for replacing the rules with a learned router later.

This project explores a few problems that show up in any serious inference platform:

- **API design matters.** The shape of `POST /chat` — what fields it exposes, what it hides — determines how easy the system is to build on. Good inference APIs feel obvious in hindsight.
- **Providers are interchangeable at the interface, not in practice.** Abstracting OpenAI and HuggingFace behind a common interface is straightforward. Handling their different failure modes, rate limits, and latency profiles is where the real work is.
- **Cost is a first-class concern.** `max_cost` isn't a nice-to-have. At scale, routing decisions are billing decisions. The abstraction has to be designed with that in mind from the start.
- **The gateway is the leverage point.** New models, new features, new providers — they all land here first. The cleaner this layer, the easier everything downstream becomes.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           LLM Inference Gateway      │
                        │                                       │
  POST /chat  ───────►  │  ┌──────────┐     ┌──────────────┐  │
                        │  │  FastAPI  │────►│    Router    │  │
  {                     │  │  /chat   │     │              │  │
    prompt,             │  └──────────┘     │  priority?   │  │
    priority,           │                   │  prompt len? │  │
    max_cost            │                   │  max_cost?   │  │
  }                     │                   └──────┬───────┘  │
                        │                          │           │
                        │            ┌─────────────┼──────┐   │
                        │            ▼             ▼      ▼   │
                        │      ┌─────────┐  ┌─────────┐       │
                        │      │ OpenAI  │  │   HF    │  ...  │
                        │      │Provider │  │Provider │       │
                        │      └────┬────┘  └────┬────┘       │
                        │           │             │            │
                        │           └──────┬──────┘            │
                        │                  ▼                   │
                        │           ┌────────────┐             │
                        │           │   Logger   │             │
                        │           │  (SQLite)  │             │
                        │           └────────────┘             │
                        └─────────────────────────────────────┘

                                          │
                              ┌───────────┴───────────┐
                              ▼                       ▼
                        ┌──────────┐           ┌──────────────┐
                        │  OpenAI  │           │  HuggingFace │
                        │   API    │           │     API      │
                        └──────────┘           └──────────────┘
```

**Request flow:**

1. Client sends `POST /chat` with `prompt`, `priority`, and optional `max_cost`
2. FastAPI validates the request against the schema
3. Router applies rules and selects a provider
4. Selected provider calls the upstream API (or returns a mock)
5. Logger writes the request, model choice, and latency to SQLite
6. Response returns to the client

**Key design decisions:**

- Providers share a common interface — swapping one out doesn't touch the router
- Routing logic is isolated in its own module — rules change without touching the API layer
- Logger is fire-and-forget — it never blocks the response path
- `max_cost` and `priority` are optional — the gateway degrades gracefully to a default

---

## Project Plan

### Phase 1 — Core API (current)
- [ ] `POST /chat` endpoint with `prompt`, `priority`, `max_cost` fields
- [ ] Request/response schemas with validation
- [ ] Rule-based router (`priority`, prompt length, cost thresholds)
- [ ] Provider abstraction with `generate_response(prompt)`
- [ ] OpenAI provider (mocked if no API key)
- [ ] HuggingFace provider (mocked if no API key)
- [ ] SQLite logger (timestamp, prompt, model, latency, response length)
- [ ] Metrics script (avg latency per model, usage count per model)

### Phase 2 — Real Integrations (next)
- [ ] Wire up real OpenAI API key
- [ ] Wire up real HuggingFace inference API
- [ ] Add provider error handling and fallback logic
- [ ] Extend routing rules (token count, domain detection)

### Phase 3 — Learned Router (future)
- [ ] Use logged request data to train a routing classifier
- [ ] Replace rule-based router with ML model
- [ ] A/B test rule-based vs. learned router

---

## Project Structure

```
llm_inference_gateway/
├── app/
│   ├── main.py               # FastAPI entry point
│   ├── router.py             # Routing logic
│   ├── schemas.py            # Request/response models
│   ├── logger.py             # SQLite request logger
│   └── providers/
│       ├── __init__.py
│       ├── openai_provider.py
│       └── huggingface_provider.py
├── metrics.py                # Compute avg latency and model usage
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload
```

---

## Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformers in one sentence", "priority": "high"}'
```

---

## Routing Rules

| Condition | Model |
|---|---|
| `priority == "high"` | GPT-4 |
| `len(prompt) < 100` | Mistral (small, fast) |
| `max_cost` is low (< 0.01) | Cheapest available |
| Default | GPT-4 |

---

## Logs

Every request is logged to `logs/requests.db` (SQLite).

Fields: `timestamp`, `prompt`, `selected_model`, `latency_ms`, `response_length`

---

## Metrics

```bash
python metrics.py
```

Outputs avg latency per model and total request count per model.
