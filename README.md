# LLM Inference Gateway

A FastAPI service that routes prompts to different LLM providers using rule-based and ML-based logic. Every request is logged to SQLite — that log is the training data for the next version of the router.

Not every prompt needs GPT-4. This gateway picks the right model based on priority, cost, and what the prompt is actually asking for.

---

## What This Is

A from-scratch inference gateway built in 5 phases. Starts with `if/else` rules, ends with a trained ML classifier making routing decisions. Built entirely free — providers are mocked, no API keys needed.

**Technical scope:**
- REST API with FastAPI
- Rule-based and ML-based routing
- NLP classification (zero-shot + TF-IDF/LR trained classifier)
- SQLite logging with DB migration
- In-memory response cache
- Batch endpoint
- Full observability via `metrics.py`
- 28 unit + integration tests

---

## Technical Strengths

**Provider abstraction.** `OpenAIProvider` and `HuggingFaceProvider` both implement `generate_response(prompt) → str`. The router doesn't know or care which it's talking to. Swapping real APIs in requires one line.

**Layered routing.** Rules run in priority order — hard constraints first (priority flag, cost cap), learned model second, zero-shot fallback third. The system always works even before any training.

```python
if priority == "high":          # Rule 1: hard constraint
    return gpt4
if max_cost < 0.01:             # Rule 2: hard constraint
    return mistral
if is_trained_model_available():
    return predict_model(prompt)  # Rule 3: learned
else:
    return classify_prompt(prompt)  # Rule 4: zero-shot fallback
```

**No dead ends.** Learned router not trained yet? Falls back to zero-shot. Zero-shot uncertain? Returns default model. The system degrades gracefully at every layer.

**Cache keyed on (prompt, model).** Not just prompt. Same prompt routed to two different models gets two separate cache entries. Cache hits are logged with a `cache_hit:` prefix so metrics break them out.

**DB migration built in.** `init_db()` checks existing column names via `PRAGMA table_info` and `ALTER TABLE`s missing columns. Existing databases don't break when the schema changes.

**Synthetic data with diversity controls.** First version of training data got 100% accuracy — a red flag. Added diverse phrasing, short questions, ambiguous prompts. Accuracy dropped to 81% and real-world predictions improved. 81% is more honest than 100%.

---

## Architecture

```
  Client
    │
    │  POST /chat { prompt, priority, max_cost }
    │  POST /chat/batch { prompts[], priority, max_cost }
    ▼
┌────────────────────────────────────────────────┐
│                  LLM Gateway                    │
│                                                 │
│  FastAPI  →  Cache  →  Router  →  Provider      │
│                           │                     │
│                    ┌──────┴──────┐              │
│                    ▼             ▼              │
│               Rule-based    ML classifier       │
│               (priority,    (TF-IDF+LR or       │
│                cost)         zero-shot)         │
│                                                 │
│                        Logger (SQLite)          │
└────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
    OpenAI API            HuggingFace API
    (GPT-4)               (Mistral 7B)
```

**Request path:**
1. Cache check — if `(prompt, model)` seen before, return immediately
2. Router — picks provider and model, records the reason
3. Provider — calls the model
4. Logger — writes prompt, model, reason, latency, response length to SQLite

---

## Routing Decision Flow

```
  incoming request
        │
        ▼
  priority == "high"?      ──yes──► GPT-4        [rule-based]
        │ no
        ▼
  max_cost < 0.01?         ──yes──► Mistral       [rule-based]
        │ no
        ▼
  trained model available?
        │ yes                          │ no
        ▼                              ▼
  predict_model(prompt)        classify_prompt(prompt)   [zero-shot]
        │                              │
        └──────────────┬───────────────┘
                       ▼
              route to model
```

---

## Phases

| Phase | What it adds | Type |
|---|---|---|
| 1 | FastAPI skeleton, if/else router, SQLite logger | Engineering |
| 2 | Real provider API calls (OpenAI SDK, HF InferenceClient) | Engineering + NLP |
| 3 | Zero-shot classifier replaces length heuristic | NLP |
| 4 | TF-IDF + LR trained on synthetic data, zero-shot fallback | NLP |
| 5 | Exact-match cache, batch endpoint, routing metrics | Engineering |

<details>
<summary>Phase 1 — Core Gateway</summary>

**Decision:** SQLite over a flat file. Queryable from day one, zero extra infra.

**Decision:** If/else routing before any ML. You need the plumbing correct before you add intelligence. A classifier adds 100-200ms. `priority == "high"` adds zero.

</details>

<details>
<summary>Phase 2 — Real Providers</summary>

Both providers share one interface: `generate_response(prompt) → str`. The router never calls a provider directly — it calls the interface. Adding a new provider (Anthropic, Cohere) means adding one file, not touching the router.

Mocked in this repo — no API keys needed. To wire real calls: add `OPENROUTER_API_KEY` to `.env`, swap providers in `router.py`.

</details>

<details>
<summary>Phase 3 — Zero-Shot Classification</summary>

**What:** Model `typeform/distilbert-base-uncased-mnli` classifies prompt intent — code generation, summarization, Q&A, general chat — without being trained on any examples.

**Why not rule-based:** Length is a bad proxy. A 50-character prompt can be complex. A 200-character prompt can be trivial. Zero-shot reads meaning, not character count.

**Why this model:** ~260MB, no GPU needed, runs in ~100-200ms on any machine. `facebook/bart-large-mnli` is more accurate but 1.6GB — too heavy for local dev.

**Known limit:** Small model misclassifies ambiguous prompts. "What is the capital of France?" sometimes comes back as summarization. Phase 4 fixes this with training.

</details>

<details>
<summary>Phase 4 — Learned Router</summary>

**What:** TF-IDF vectorizer + Logistic Regression classifier trained on 103 synthetic `(prompt, model)` pairs. Predicts which model to use without any rules.

**Why not DistilBERT fine-tuning:** Three real constraints:
- Intel Mac (2015) maxes at torch 2.2.2. Fine-tuning needs 2.4+.
- CPU fine-tuning takes hours. Not practical.
- 103 examples is too few. Fine-tuning needs 1000+ minimum.

TF-IDF + LR trains in under a second, requires no GPU, gets 81% accuracy on held-out synthetic data. Many production routing systems use exactly this approach.

**On the 81% number:** The first version got 100% — trained on 60 similar examples and memorized them. After adding diverse, ambiguous, and short-form examples, accuracy dropped to 81% but out-of-sample predictions improved. 100% on a small clean dataset is a warning sign.

**Upgrade path:** Accumulate 1000+ real requests from SQLite, fine-tune DistilBERT on Colab, swap `predict_model()` in `learned_router.py`. Nothing else changes.

</details>

<details>
<summary>Phase 5 — Efficiency + Observability</summary>

**Cache:** Exact-match dict keyed on `(prompt, model)`. O(1) lookup, zero false positives. Semantic caching (embed + nearest neighbor) catches paraphrases but adds ~50ms overhead per miss and can return wrong answers for near-similar prompts. Exact-match is the right default for a gateway that doesn't own the semantic meaning of prompts.

**Batch endpoint:** `POST /chat/batch` accepts a list of prompts with shared priority/cost settings. Each prompt routes independently and benefits from the cache.

**Metrics:** `python metrics.py` shows usage per model, avg latency, and routing distribution. Cache hits show up as `cache_hit:learned_router` etc. — separate from cold requests.

</details>

---

## NLP Concepts

<details>
<summary>Zero-Shot Classification</summary>

Normal classification needs labeled training data. Zero-shot skips it — the model figures out which category fits from the label names alone.

```python
classifier = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

result = classifier(
    "write a function to sort a list",
    candidate_labels=["code generation", "summarization", "Q&A", "general chat"]
)
# → code generation: 84%

result = classifier(
    "make this shorter",   # ambiguous
    candidate_labels=["code generation", "summarization", "Q&A", "general chat"]
)
# → summarization: 41%, general chat: 35%  ← model is unsure
```

**Relevant in 2026?** Yes for cheap local classification before you have training data. Once you have logs, replace it with a fine-tuned model.

</details>

<details>
<summary>TF-IDF + Logistic Regression</summary>

TF-IDF turns text into numbers: words that appear often in one class but rarely overall get high weight. Logistic Regression draws a boundary between classes in that number space.

Not a transformer. Trains in milliseconds, runs in microseconds, uses no GPU. Interpretable — you can inspect which words push toward which class.

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(prompts, labels)
model.predict(["write a sorting algorithm in Python"])
# → "mistral"
```

**Relevant in 2026?** Yes. Fast, cheap, explainable. Used in production routing at scale — not everything needs a transformer.

</details>

<details>
<summary>Fine-Tuning</summary>

Taking a pretrained model (DistilBERT, Llama) and continuing training on your specific task. The model already understands language — fine-tuning adds the last mile.

Not implemented here due to hardware constraints. But the architecture is ready: once `logs/requests.db` has enough real `(prompt, model)` pairs, fine-tune DistilBERT on Colab and swap `predict_model()`. One function, nothing else changes.

**Relevant in 2026?** Yes for narrow classification tasks. A fine-tuned 66M-param DistilBERT costs a fraction of a GPT-4 call per request at scale.

</details>

<details>
<summary>Quantization</summary>

Running a model at lower numerical precision — 8-bit or 4-bit integers instead of 32-bit floats. Cuts memory 4-8x with minimal quality loss.

```
fp32:   28GB  — needs a high-end GPU
fp16:   14GB  — needs a GPU
8-bit:   7GB  — most GPUs work
4-bit:   4GB  — runs on a MacBook
```

Not implemented here (no local GPU). Relevant if you run self-hosted models.

**Relevant in 2026?** Standard. `bitsandbytes` and GGUF are the common tools. Anyone running open-source models locally uses this.

</details>

---

## What's Relevant in 2026, What's Not

**Still relevant:**
- Provider abstraction pattern — every inference platform uses this
- Layered routing (rules → learned → fallback) — production pattern
- TF-IDF classifiers for cheap high-volume routing decisions
- Zero-shot classification for bootstrapping before you have labels
- Exact-match caching — cuts cost 30-60% for workloads with repeated prompts
- SQLite for local observability — simple, works, queryable

**Less relevant / simplified for this project:**
- Fine-tuning on CPU — not practical anymore. Use cloud GPUs (Colab, Modal, RunPod)
- Synthetic training data — real logs are always better. Synthetic is a bootstrap only
- Exact-match cache at scale — semantic caching (embeddings + vector DB) handles paraphrases. Relevant for production, overkill here
- Quantization — matters when you self-host. Less relevant if you're calling APIs

---

## Takeaways

The routing logic is the core engineering problem. It's not "call GPT-4 for everything" — it's deciding which model is good enough for each request, at what cost, with what latency budget. That decision happens on every request at production scale.

The 5-phase structure mirrors how real systems evolve. Phase 1 rules still run in production — they handle hard constraints faster than any model can. The learned router runs for everything else.

Logging every routing decision isn't optional. It's how you know if the router is working, and it's your training data for the next version.

81% accuracy on the learned router sounds unimpressive. On 200 requests a day where the alternative is sending everything to GPT-4, it's the difference between a $10 bill and a $2 bill. Routing matters.

---

## Project Structure

```
app/
  main.py               # FastAPI entry, cache check, /chat and /chat/batch
  router.py             # Routing logic — rules, learned, zero-shot fallback
  schemas.py            # Request/response Pydantic models
  logger.py             # SQLite logger with DB migration
  cache.py              # Exact-match in-memory cache
  classifier.py         # Zero-shot classification pipeline
  learned_router.py     # TF-IDF + LR model loader and predictor
  providers/
    base.py
    openai_provider.py
    huggingface_provider.py
    openrouter_provider.py  # built, not wired — ready for real API calls
scripts/
  generate_training_data.py
  train_router.py
  smoke_test.py         # End-to-end test against running server
tests/
  test_router.py
  test_api.py
  test_classifier.py
  test_cache_and_batch.py
metrics.py              # CLI: usage per model, latency, routing distribution
```

---

## Setup

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive API.

**Train the learned router:**
```bash
python scripts/generate_training_data.py
python scripts/train_router.py
```

**Run tests:**
```bash
python -m pytest tests/ -v
```

**Smoke test against running server:**
```bash
python scripts/smoke_test.py
```

**Check metrics:**
```bash
python metrics.py
```
