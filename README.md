# LLM Inference Gateway

A FastAPI service that routes prompts to different LLM providers using rule-based logic, and logs every request for future ML training.

Not every prompt needs GPT-4. This gateway picks the right model based on priority, prompt length, and cost — then logs every decision as training data for a future learned router.

**Current state:** Providers are mocked — no API keys or credits needed. The `OpenRouterProvider` is built and ready (`app/providers/openrouter_provider.py`) but not wired in. To use real models, add an `OPENROUTER_API_KEY` to `.env` and swap the providers in `router.py`.

---

## System Architecture

The structure stays the same across all phases. What evolves is what happens inside each box.

```
  Client
    │
    │  POST /chat { prompt, priority, max_cost }
    ▼
┌──────────────────────────────────────────────────────────┐
│                      LLM Gateway                          │
│                                                           │
│   ┌──────────┐      ┌──────────┐      ┌───────────────┐  │
│   │  FastAPI  │─────►│  Router  │─────►│   Provider    │  │
│   │  /chat   │      └──────────┘      │               │  │
│   └──────────┘                        │  OpenAI       │  │
│                                       │  HuggingFace  │  │
│                                       └───────┬───────┘  │
│                                               │           │
│                                       ┌───────▼───────┐  │
│                                       │    Logger     │  │
│                                       │   (SQLite)    │  │
│                                       └───────────────┘  │
└──────────────────────────────────────────────────────────┘
         │                                      │
         ▼                                      ▼
    OpenAI API                        HuggingFace API
    (GPT-4, GPT-3.5)                  (Mistral, Llama)
```

| Phase | What changes |
|---|---|
| 1 | Provider returns a mock string |
| 2 | Provider makes real API calls |
| 3 | Router runs a zero-shot classifier before picking a provider |
| 4 | Router uses a trained ML model instead of if/else rules |
| 5 | Logger gains a cache layer, providers use quantized models |

---

## Roadmap

<details>
<summary><strong>Phase 1 — Core Gateway [engineering] ✅ current</strong></summary>

Build the skeleton: API, rule-based router, provider abstraction, SQLite logger.

**What's engineering:** FastAPI schema design, clean module boundaries, provider interface, logging that never blocks the response path.

**Tradeoff:** Rule-based routing is simple but fast. `priority == "high"` takes zero latency. A classifier takes 50-200ms. For Phase 1, if/else is correct — you need the structure before you need the intelligence.

**Decision:** SQLite over a JSON file. Queryable from day one, no extra infra, good enough for thousands of requests locally.

</details>

<details>
<summary><strong>Phase 2 — Real Providers [engineering + NLP]</strong></summary>

Replace mocks with real API calls. OpenAI via `openai` SDK, HuggingFace via `InferenceClient`.

**What's engineering:** Error handling, timeouts, fallback logic (if OpenAI fails, route to HF), provider health checks.

**What's NLP:** Choosing which HuggingFace model to call. `mistralai/Mistral-7B-Instruct` for general tasks, `facebook/bart-large-cnn` for summarization.

**Current state (2026):** HuggingFace `InferenceClient` is the standard for calling hosted open-source models. Mistral and Llama variants are the go-to cheap alternatives to GPT-4. This pattern is everywhere.

**Tradeoff:** Self-hosted models vs. HuggingFace Inference API (hosted, costs money). For an MVP, use the hosted API — don't manage GPUs yet.

</details>

<details>
<summary><strong>Phase 3 — Smarter Routing [NLP] ✅ done</strong></summary>

Replaced `len(prompt) < 100` with a zero-shot classifier that detects actual task type — summarization, Q&A, code generation, general chat. Routes based on what the prompt is asking for, not how long it is.

**What changed:**
- Added `app/classifier.py` — loads `typeform/distilbert-base-uncased-mnli` once at startup
- Updated `app/router.py` — calls `classify_prompt(prompt)` instead of checking length

**Architecture:**
```
  incoming request
        │
        ▼
  priority == "high"?  ──yes──► best model
        │ no
        ▼
  max_cost < 0.01?     ──yes──► fast model
        │ no
        ▼
  classify_prompt(prompt)          ← NEW in Phase 3
        │
  ┌─────┴──────────────┬──────────────────┐
  ▼                    ▼                  ▼
code generation    summarization    Q&A / general
  │                    │                  │
fast model         fast model        default model
```

**Why zero-shot over rule-based:**
Length is a proxy. A 50-character prompt can be complex. A 200-character prompt can be trivial. Zero-shot reads the actual meaning — no training data needed.

**Why this model (`typeform/distilbert-base-uncased-mnli`):**
- ~260MB — runs on a 2015 MacBook
- No GPU needed
- `facebook/bart-large-mnli` is more accurate but ~1.6GB — too heavy for local dev

**Why local pipeline, not an API call:**
Routing decisions need to be fast and free. An API call for classification adds cost and network latency before the actual model call. Local pipeline runs in ~100-200ms and never hits a rate limit.

**Known limitation:**
Small model misclassifies some prompts — "What is the capital of France?" sometimes comes back as summarization instead of Q&A. Acceptable for now. Phase 4 fixes this with a fine-tuned classifier trained on your actual logs.

**Relevant in 2026?** Yes for small systems. Frontier labs use learned routers with millions of examples. Zero-shot is the right starting point before you have data.

</details>

<details>
<summary><strong>Phase 4 — Learned Router [NLP] ✅ done</strong></summary>

Replaced the zero-shot classifier with a trained model. Input: prompt. Output: best model. No rules, no guessing — learned from data.

**What changed:**
- `scripts/generate_training_data.py` — generates synthetic (prompt, model) training pairs
- `scripts/train_router.py` — trains and saves a TF-IDF + Logistic Regression classifier
- `app/learned_router.py` — loads the trained model and predicts at request time
- `app/router.py` — uses learned router if model exists, falls back to zero-shot otherwise

**Architecture:**
```
  incoming request
        │
        ▼
  priority == "high"?      ──yes──► best model
        │ no
        ▼
  max_cost < 0.01?         ──yes──► fast model
        │ no
        ▼
  trained model available?
        │ yes                        │ no (fallback)
        ▼                            ▼
  predict_model(prompt)      classify_prompt(prompt)  ← Phase 3
        │                            │
        ▼                            ▼
  learned prediction          zero-shot prediction
```

**Why TF-IDF + Logistic Regression over DistilBERT:**
- Trains in seconds on any hardware including a 2015 MacBook
- No GPU, no torch version conflicts
- 100% accuracy on test set with 60 training examples
- DistilBERT is the upgrade path when you have thousands of real logged requests and better hardware

**Why synthetic data:**
Real traffic requires running the system for weeks. Synthetic data lets you validate the training pipeline now. When real logs accumulate, swap `generate_training_data.py` for a script that reads from SQLite.

**Performance on test set:**
```
              precision    recall    f1
gpt-4            0.71       1.00    0.83
mistral          1.00       0.64    0.78
accuracy                            0.81
```

**On overfitting:**
The first version had 60 clean, similar examples and got 100% accuracy — a sign it memorized the data rather than learning patterns. "Why do stars twinkle?" routed to Mistral instead of GPT-4.

The fix: add more diverse examples — short questions, ambiguous phrasing, varied vocabulary — so the model learns the underlying pattern, not the surface words. After expanding to 103 examples with more variety, accuracy dropped to 81% on the test set but 6/6 correct on genuinely unseen prompts.

81% is more honest. 100% on synthetic data is always suspicious.

**Zero-shot fallback:**
The router uses the learned model if `data/router_model.pkl` exists. If not, it falls back to the zero-shot classifier from Phase 3. This means the system always works — even before training.

```python
if is_trained_model_available():
    model_name = predict_model(prompt)   # learned router
else:
    task = classify_prompt(prompt)       # zero-shot fallback
```

**How to retrain:**
```bash
python scripts/generate_training_data.py
python scripts/train_router.py
```

**Relevant in 2026?** This is how production routing works. TF-IDF classifiers power routing in systems that can't afford transformer inference on every request. Simple, fast, interpretable.

</details>

<details>
<summary><strong>Phase 5 — Efficiency [engineering + NLP]</strong></summary>

Optimize the hot path: response caching, request batching, model quantization for self-hosted models.

**What's engineering:** Cache layer (exact-match or semantic), batch queue for high-volume periods, latency SLOs per model.

**What's NLP:** Quantization (4-bit, 8-bit) for any self-hosted models. Cuts memory and speeds up inference with minimal quality loss.

**Current state (2026):** Quantization is mature and widely used. `bitsandbytes` and GGUF formats make it straightforward. Standard practice for anyone running open-source models locally.

</details>

---

## How It Works

### Routing Decision Flow

**Phase 1** (if/else rules):
```
  incoming request
        │
        ▼
  priority == "high"?  ──yes──► GPT-4
        │ no
        ▼
  max_cost < 0.01?     ──yes──► Mistral
        │ no
        ▼
  len(prompt) < 100?   ──yes──► Mistral
        │ no
        ▼
      default          ────────► GPT-4
```

**Phase 3** (zero-shot classifier replaces length check):
```
  incoming request
        │
        ▼
  priority == "high"?  ──yes──► best model
        │ no
        ▼
  max_cost < 0.01?     ──yes──► fast model
        │ no
        ▼
  classify_prompt(prompt)
  ┌─────┴──────────────┬──────────────────┐
  ▼                    ▼                  ▼
code generation    summarization    Q&A / general
  │                    │                  │
fast model         fast model        default model
```

<details>
<summary><strong>The Endpoint</strong></summary>

`POST /chat` accepts a JSON body with three fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | yes | The text you want to send to a model |
| `priority` | `low` / `medium` / `high` | no | How urgently you need a good response |
| `max_cost` | float | no | The max cost per request you're willing to pay |

**Example request:**
```json
{
  "prompt": "Explain what a transformer model is",
  "priority": "high",
  "max_cost": 0.05
}
```

**Example response:**
```json
{
  "response": "A transformer model is...",
  "model_used": "gpt-4"
}
```

</details>

<details>
<summary><strong>The Routing Rules</strong></summary>

When a request comes in, the router checks fields in this order and stops at the first match:

**Rule 1 — Priority is high → GPT-4**
```
priority == "high"  →  OpenAI GPT-4
```
You've said this request matters. Send it to the best model regardless of cost.

**Rule 2 — Max cost is low → Cheapest model**
```
max_cost < 0.01  →  Mistral 7B (HuggingFace)
```
You've set a tight budget. Send it to the cheapest available model.

**Rule 3 — Short prompt → Mistral**
```
len(prompt) < 100 characters  →  Mistral 7B (HuggingFace)
```
Short prompts are usually simple questions. A smaller, faster model handles them fine.

**Default → GPT-4**
```
nothing matched  →  OpenAI GPT-4
```
When in doubt, use the best model.

</details>

<details>
<summary><strong>The Providers</strong></summary>

Both providers are **mocked in Phase 1** — they return a fake string instead of calling a real API. The real API calls are stubbed out in comments, ready to be wired up in Phase 2.

- `OpenAIProvider` — wraps GPT-4 and GPT-3.5. Uses the `openai` SDK.
- `HuggingFaceProvider` — wraps Mistral and other open-source models. Uses `InferenceClient`.

Both share the same interface: `generate_response(prompt) → str`. The router doesn't care which provider it picks — it just calls `generate_response`.

</details>

<details>
<summary><strong>The Logger</strong></summary>

Every request is automatically logged to `logs/requests.db` (SQLite).

| Field | Description |
|---|---|
| `timestamp` | When the request was made |
| `prompt` | The full prompt text |
| `selected_model` | Which model the router picked |
| `latency_ms` | How long the provider took to respond |
| `response_length` | Number of characters in the response |

This data becomes the training set for the Phase 4 learned router.

</details>

---

## Architecture

<details>
<summary><strong>Evolution by phase</strong></summary>

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
  │             │            │           │           │
Rules        Real         NLP         Learned     Efficient
(if/else,    APIs         signals     router      inference
 fast)       wired        (zero-      (fine-      (cache,
             up           shot)       tuned)      quant)

[engineering] [eng+NLP]  [NLP]       [NLP]       [eng+NLP]
```

</details>

---

## NLP Concepts

<details>
<summary><strong>Zero-Shot Classification</strong></summary>

You give a model some text and a list of category labels. It tells you which label fits best — without ever being trained on your specific categories.

Normal classification requires training data: thousands of labeled examples before the model can do anything. Zero-shot skips that. The model already understands language well enough to figure it out from the label names alone.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Easy case — strong signal in the text
result = classifier(
    "the login button doesn't work on mobile",
    candidate_labels=["summarization", "question answering", "code generation", "general chat"]
)
# Output: code generation    → 78%
#         general chat       → 14%
# → router picks: CodeLlama

# Hard case — ambiguous
result = classifier(
    "make this shorter",
    candidate_labels=["summarization", "question answering", "code generation", "general chat"]
)
# Output: summarization      → 41%
#         general chat       → 35%   ← model is unsure
# → zero-shot struggles here, fine-tuning on your logs would do better
```

**Used here:** Phase 3 routing. **Relevant in 2026?** Yes for cheap local routing. For anything heavier, just ask an LLM directly.

</details>

<details>
<summary><strong>Fine-Tuning</strong></summary>

Taking a model that already understands language and training it further on your specific data.

You don't start from scratch. A base model like DistilBERT has already learned grammar, context, and meaning from billions of words. Fine-tuning adds the last mile — teaching it your specific task with a small labeled dataset.

```python
from transformers import Trainer, TrainingArguments

# logged data: (prompt, model_that_produced_best_result)
# e.g. ("summarize this report", "bart-large-cnn")
#      ("write a python script", "codellama")

trainer = Trainer(
    model=distilbert,
    args=TrainingArguments(output_dir="./router", num_train_epochs=3),
    train_dataset=logged_requests,
)
trainer.train()

# After training:
# Input:  "Can you write a sorting algorithm in Python?"
# Output: codellama  ← learned from logs, not hardcoded
```

**Tradeoff:** Needs 500-1000 examples minimum. That's why Phase 1 logging matters. **Relevant in 2026?** Yes for narrow tasks — a fine-tuned classifier costs a fraction of a GPT-4 call at scale.

</details>

<details>
<summary><strong>Text Generation</strong></summary>

The model reads your prompt and predicts the next word. Then the next. Then the next — until done.

What makes transformers powerful is attention: when predicting each word, the model can look at every other word in the input at once. Older models forgot early context. Transformers don't.

```
Prompt:   "The capital of France is"
Step 1:   model predicts → "Paris"
Step 2:   model predicts → "."
Step 3:   model predicts → [end]
```

```python
from huggingface_hub import InferenceClient

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3")
response = client.text_generation("Explain transformers in one sentence.", max_new_tokens=100)
# → "A transformer model is a neural network that uses attention to process sequences in parallel."
```

**Tradeoff:** Longer output = more cost. This is why `max_cost` matters as a routing signal. **Relevant in 2026?** Foundation of everything. Non-negotiable to understand.

</details>

<details>
<summary><strong>Quantization</strong></summary>

Shrinking a model by reducing the precision of its numbers.

Models store weights as 32-bit floats by default. Quantization rounds them to 8-bit or 4-bit integers. The quality loss is small. The size reduction is large.

```
fp32 (default):   28GB RAM  — needs a high-end GPU
fp16:             14GB RAM  — still needs a GPU
8-bit:             7GB RAM  — runs on most GPUs
4-bit:             4GB RAM  — runs on a MacBook
```

```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
# same model, 7x smaller, slightly lower quality
```

**Used here:** Phase 5 for self-hosted models. **Relevant in 2026?** Standard practice. `bitsandbytes` and GGUF are the common tools.

</details>

<details>
<summary><strong>Embeddings (Phase 5)</strong></summary>

Turn text into a list of numbers where similar meaning = similar numbers.

"What is a dog?" and "Explain what dogs are" produce vectors close to each other. "What is a dog?" and "Deploy to Kubernetes" are far apart. Distance between vectors = distance between meanings.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["What is a transformer?", "Explain transformers", "Deploy to Kubernetes"])

# cosine similarity:
# prompt 1 vs prompt 2 → 0.96  (same meaning → cache hit)
# prompt 1 vs prompt 3 → 0.11  (different meaning → no cache hit)
```

**Used here:** Phase 5 semantic caching. **Relevant in 2026?** One of the most useful concepts in production ML. Powers RAG, semantic search, and similarity caching.

</details>

---

## Project Structure

```
app/
  main.py               # FastAPI entry point
  router.py             # Routing logic
  schemas.py            # Request/response models
  logger.py             # SQLite request logger
  providers/
    openai_provider.py
    huggingface_provider.py
metrics.py              # Avg latency and usage per model
requirements.txt
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for the interactive API UI.

---

## Running Tests

```bash
source venv311/bin/activate
python -m pytest tests/ -v
```

19 tests covering:
- Unit tests for routing rules (`tests/test_router.py`)
- Integration tests for the `/chat` endpoint (`tests/test_api.py`)
- Unit tests for the classifier and learned router (`tests/test_classifier.py`)

---

## Example Request

```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"prompt": "Explain transformers in one sentence", "priority": "high"}'
```
