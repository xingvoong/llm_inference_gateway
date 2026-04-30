# LLM Inference Gateway

A FastAPI service that routes prompts to different LLM providers using rule-based logic, and logs every request for future ML training.

Not every prompt needs GPT-4. This gateway picks the right model based on priority, prompt length, and cost — then logs every decision as training data for a future learned router.

---

## Architecture

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

---

## Routing Decision Flow

```
  incoming request
        │
        ▼
  priority == "high"?  ──yes──► GPT-4
        │ no
        ▼
  max_cost < 0.01?     ──yes──► Cheapest model
        │ no
        ▼
  len(prompt) < 100?   ──yes──► Mistral (fast, cheap)
        │ no
        ▼
      default          ────────► GPT-4
```

**Phase 3 upgrade:** replace `len(prompt) < 100` with a zero-shot classifier:

```
  incoming request
        │
        ▼
  priority == "high"?  ──yes──► GPT-4
        │ no
        ▼
  max_cost < 0.01?     ──yes──► Cheapest model
        │ no
        ▼
  classify(prompt)
  ┌─────┴──────────────┬────────────────┬──────────┐
  ▼                    ▼                ▼          ▼
summarization        Q&A            code gen    general
  │                    │                │          │
bart-large-cnn    deepset/roberta    CodeLlama   GPT-4
```

---

## Evolution by Phase

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
  │             │            │           │           │
Rules        Real         NLP         Learned     Efficient
(if/else,    APIs         signals     router      inference
 fast)       wired        (zero-      (fine-      (cache,
             up           shot)       tuned)      quant)

[engineering] [eng+NLP]  [NLP]       [NLP]       [eng+NLP]
```

---

## NLP Concepts Used in This Project

### Zero-Shot Classification
**What it is:** Classify text into categories without any training data. You give the model a prompt and a list of labels — it returns a probability for each label.

**Zero-shot learning** means the model can do a task it was never explicitly trained on. Normal classification requires labeled examples — "here are 1000 emails labeled spam/not spam, learn to tell them apart." Zero-shot skips that entirely. You just describe the task in plain language and the model figures it out.

That's possible because large models trained on massive text already understand what words and concepts mean. "Summarization" isn't a mystery to a model that's read the entire internet. Zero-shot exploits that existing knowledge instead of building from scratch.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "Can you summarize this article for me?",
    candidate_labels=["summarization", "question answering", "code generation"]
)
# Output: summarization — 94%
```

No training. No labeled data. Just the model's existing understanding of language.

**How it works under the hood:** A model trained on natural language inference (NLI) learns to judge whether a hypothesis follows from a premise. Zero-shot classification repurposes this — "this text is about summarization" becomes the hypothesis, your prompt becomes the premise.

**Where it's used here:** Phase 3 routing. Instead of `len(prompt) < 100`, we classify the prompt as summarization, Q&A, code, or general — then route to the best model for that task.

**Tradeoff:** Less accurate than a fine-tuned model. A classifier trained on your specific data will always beat zero-shot on that same data. But zero-shot costs nothing to set up and works immediately — which is why it's useful for routing in Phase 3 before you have enough logged data to fine-tune in Phase 4.

**Relevant in 2026?** Yes, but narrowly. For lightweight routing decisions where you don't want to call a large model, a local zero-shot classifier is still practical. For anything more complex, you'd just prompt an LLM to classify it directly.

---

### Text Classification / Fine-tuning
**What it is:** Training a model to assign a label to a piece of text. Fine-tuning takes a pre-trained model and continues training it on your specific task and data.

**The key idea:** you don't train from scratch. A model like DistilBERT has already read hundreds of billions of words — it knows grammar, context, and meaning. Fine-tuning just teaches it the last mile: "for this project, these kinds of prompts should go to this model." You're steering existing knowledge, not building new knowledge.

Think of it like hiring an experienced engineer and giving them a one-week onboarding. You're not teaching them to code — you're teaching them your specific codebase.

```python
from transformers import Trainer, TrainingArguments

# logged data: (prompt, model_that_worked_best)
training_data = load_from_sqlite("logs/requests.db")

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./router-model", num_train_epochs=3),
    train_dataset=training_data,
)
trainer.train()
```

**Where it's used here:** Phase 4 learned router. Fine-tune a small classifier on logged `(prompt, chosen_model)` pairs. The model learns which prompts route well to which providers — replacing the if/else rules entirely.

**Tradeoff:** You need data first. Fine-tuning on 50 examples produces a bad model. You need 500-1000 minimum. That's why Phase 1 logging matters — every request is a future training example.

**Relevant in 2026?** Yes, for narrow tasks. Fine-tuning a small classifier is far cheaper than calling GPT-4 for every routing decision. At high request volume, that cost difference compounds fast. Companies running inference at scale do this.

---

### Text Generation (Transformers)
**What it is:** Generating text one token at a time given an input prompt. The model predicts the most likely next token, appends it, and repeats until done.

**The key idea:** a token isn't a word — it's a chunk of text, usually 3-4 characters. "transformer" might be split into ["trans", "former"]. The model operates on tokens, not words. It predicts one at a time, in order, left to right.

What makes transformers different from older models is **attention**. Every token can look at every other token in the input simultaneously. Older models (RNNs) processed tokens sequentially and forgot early context. Attention solves that — the model knows the full context when predicting each token.

```
Input:  "The capital of France is"
Token 1 prediction: "Paris"  ← attends to all input tokens at once
```

**Where it's used here:** The HuggingFace provider calls a hosted generation model (Mistral, Llama) via `InferenceClient`. The model takes your prompt and returns generated text token by token.

**Tradeoff:** Generation is expensive. Each token requires a full forward pass through the model. Longer outputs = more compute = more cost = more latency. This is why `max_cost` as a routing signal matters — generation length directly impacts cost.

**Relevant in 2026?** This is the foundation of everything. GPT-4, Claude, Mistral, Llama — all transformer-based text generation models. Understanding this is non-negotiable.

---

### Quantization
**What it is:** Reducing the precision of a model's weights to make it smaller and faster. A standard model stores weights as 32-bit floats. Quantization compresses them to 8-bit or 4-bit integers.

**The key idea:** most of a model's precision is wasted. The difference between 0.37291847 and 0.373 is negligible for inference quality, but the storage difference is significant. Quantization trades a tiny amount of accuracy for a large reduction in memory and compute.

The practical impact: a 7B parameter model at full precision requires ~28GB of RAM. At 4-bit quantization, that drops to ~4GB — enough to run on a MacBook.

```
Full precision (fp32):  28GB RAM,  slow
Half precision (fp16):  14GB RAM,  faster
8-bit quantization:      7GB RAM,  fast
4-bit quantization:      4GB RAM,  fastest  ← common for local inference
```

**Where it's used here:** Phase 5, for self-hosted models. Quantize before deploying to cut memory and latency on cheaper hardware.

**Tradeoff:** Some quality loss, especially at 4-bit. For routing classifiers and lightweight tasks, it's negligible. For frontier-quality generation, you might notice it.

**Relevant in 2026?** Very. `bitsandbytes` and GGUF (via `llama.cpp`) are standard tools. Anyone running open-source models locally uses quantization. It's not optional at most hardware budgets.

---

### Embeddings (future)
**What it is:** A numerical representation of text as a dense vector of floats. The key property: semantically similar text produces similar vectors. You can measure meaning by measuring distance.

**The key idea:** words aren't numbers, but models need numbers. Embeddings are how you turn language into math. A sentence gets mapped to a point in high-dimensional space (e.g. 768 dimensions). "What is a dog?" and "Can you explain dogs?" end up close to each other. "What is a dog?" and "Deploy to production" end up far apart.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["What is a dog?", "Can you explain dogs?"])

# cosine similarity ≈ 0.97 — nearly identical meaning
```

**Where it could be used here:** Semantic caching in Phase 5. Instead of caching only exact-match prompts, cache by semantic similarity. "What is a transformer?" and "Explain transformers to me" would hit the same cache entry — saving a full model call.

**Tradeoff:** Embedding every request adds latency and compute. Only worth it at high request volume where cache hits are frequent enough to justify the overhead.

**Relevant in 2026?** Yes — one of the most practically useful NLP concepts right now. Embeddings are the foundation of RAG (retrieval-augmented generation), semantic search, recommendation systems, and similarity-based caching. If you work with LLMs in production, you will use embeddings.

---

## Routing Rules

| Condition | Model | Phase |
|---|---|---|
| `priority == "high"` | GPT-4 | 1 |
| `len(prompt) < 100` | Mistral | 1 |
| `max_cost < 0.01` | Cheapest available | 1 |
| `classify(prompt) == "summarization"` | bart-large-cnn | 3 |
| `classify(prompt) == "Q&A"` | deepset/roberta-base-squad2 | 3 |
| Learned classifier output | Best model per training data | 4 |
| Default | GPT-4 | 1 |

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

---

## Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain transformers in one sentence", "priority": "high"}'
```

---

## Roadmap

### Phase 1 — Core Gateway `[engineering]` *(current)*
Build the skeleton: API, rule-based router, provider abstraction, SQLite logger.

**What's engineering:** FastAPI schema design, clean module boundaries, provider interface, async logging that never blocks the response path.

**Tradeoff:** Rule-based routing is simple but fast. `priority == "high"` takes zero latency. A classifier takes 50-200ms. For Phase 1, if/else is correct — you need the structure before you need the intelligence.

**Decision:** SQLite over a JSON file. Queryable from day one, no extra infra, good enough for thousands of requests locally.

---

### Phase 2 — Real Providers `[engineering + NLP]` *(next)*
Replace mocks with real API calls. OpenAI via `openai` SDK, HuggingFace via `InferenceClient`.

**What's engineering:** Error handling, timeouts, fallback logic (if OpenAI fails, route to HF), provider health checks.

**What's NLP:** Choosing which HuggingFace model to call. `mistralai/Mistral-7B-Instruct` for general tasks, `facebook/bart-large-cnn` for summarization.

**Current state (2026):** HuggingFace `InferenceClient` is the standard for calling hosted open-source models. Mistral and Llama variants are the go-to cheap alternatives to GPT-4. This pattern is everywhere.

**Tradeoff:** Self-hosted models (run on your machine) vs. HuggingFace Inference API (hosted, costs money). For an MVP, use the hosted API — don't manage GPUs yet.

---

### Phase 3 — Smarter Routing `[NLP]`
Replace `len(prompt) < 100` with a zero-shot classifier that detects task type — summarization, Q&A, generation, code. Route based on what the prompt is actually asking for.

**What's NLP:** Zero-shot classification pipeline (`facebook/bart-large-mnli`). No fine-tuning, no training data. Give it a prompt and a list of labels, it returns probabilities.

**Current state (2026):** Zero-shot classification is still practical for low-stakes routing decisions. It's not how frontier labs route — they use learned routers trained on millions of examples — but it's a legitimate pattern for smaller systems and a clean stepping stone to Phase 4.

**Tradeoff:** Adds 100-200ms latency per request for the classification step. Worth it if it meaningfully improves model selection. Not worth it if your routing labels are too coarse to matter. Measure first.

**Decision:** Run the classifier as a local pipeline, not an API call. Keeps routing latency predictable and free.

---

### Phase 4 — Learned Router `[NLP]`
Train a classifier on your logged requests. Input: prompt. Output: best model. Replace the rules entirely.

**What's NLP:** Fine-tuning a small transformer (DistilBERT or similar) on `(prompt, chosen_model, latency, cost)` tuples from your SQLite logs. The logs you're building in Phase 1 are the training data.

**Current state (2026):** This is how serious routing systems work. Companies like Martian and Unify have built businesses on learned routers. Fine-tuning a small classifier is cheaper and faster than it was two years ago. DistilBERT fine-tuning runs on a laptop.

**Tradeoff:** You need enough logged data before this is worth training. 500-1000 labeled examples minimum. That's why Phase 1 logging matters — you're collecting the dataset now.

---

### Phase 5 — Efficiency `[engineering + NLP]`
Optimize the hot path: response caching, request batching, model quantization for self-hosted models.

**What's engineering:** Cache layer (exact-match or semantic), batch queue for high-volume periods, latency SLOs per model.

**What's NLP:** Quantization (4-bit, 8-bit) for any self-hosted models. Cuts memory and speeds up inference with minimal quality loss.

**Current state (2026):** Quantization is mature and widely used. `bitsandbytes` and GGUF formats make it straightforward. This is standard practice for anyone running open-source models locally or on cheaper hardware.
