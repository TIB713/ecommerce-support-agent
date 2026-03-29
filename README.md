# E-commerce Support Resolution Agent

**Multi-agent Retrieval-Augmented Generation (RAG)** system that turns customer support messages into **policy-grounded**, **structured** resolutions. A **FastAPI** backend and optional **Streamlit** UI send tickets through **triage → retrieval → drafting → compliance**, using **LangChain**, **Groq**, **HuggingFace embeddings**, and a **FAISS** vector index over a synthetic policy corpus.

Public reference repo: [TIB713/ecommerce-support-agent](https://github.com/TIB713/ecommerce-support-agent).

---

## Problem Statement

Customer support automation must follow **real business rules**. A plain LLM can sound confident while **inventing** refund windows, shipping guarantees, or legal obligations. That creates **compliance risk**, inconsistent buyer experience, and **hallucinated** policy.

This project reduces that risk by:

- **Grounding** answers in retrieved policy text, not the model’s priors.
- Emitting **structured decisions** (`approve`, `deny`, `partial`, `escalate`, `needs_info`) with **citations** tied to chunk IDs.
- Adding a **compliance** pass that checks citations and flags unsupported claims before the response is final.

Structured output also helps logging, QA, and integration with ticketing or CRM tools—not only natural language replies.

---

## Architecture

The pipeline is **sequential**: each stage consumes the previous output. The retriever reads from an offline **FAISS** index built from `data/policies/`.

```text
User query + order context
        │
        ▼
┌───────────────┐
│ Triage Agent  │  classify issue, confidence, clarifying questions
└───────┬───────┘
        ▼
┌───────────────┐     ┌─────────────┐
│ Retriever     │────▶│ FAISS index │  top-k policy chunks + metadata
└───────┬───────┘     └─────────────┘
        ▼
┌───────────────┐
│ Writer Agent  │  draft decision, rationale, citations, customer reply
└───────┬───────┘
        ▼
┌───────────────┐
│ Compliance    │  validate citations ⊆ retrieval; rewrite or escalate
└───────┬───────┘
        ▼
   JSON output
```

| Stage | Role | Key file(s) |
|--------|------|--------------|
| Triage | `refund` / `shipping` / `payment` / `promo` / `fraud` / `other` + optional questions | `agents/triage_agent.py` |
| Retriever | Embed query, fetch top-k chunks with `chunk_id`, `document_name` | `agents/policy_retriever_agent.py`, `rag/retriever.py` |
| Writer | Produce draft **only** from retrieved chunks; escalate if evidence too thin | `agents/resolution_writer_agent.py` |
| Compliance | Second pass + deterministic citation checks | `agents/compliance_agent.py` |
| Orchestration | Wires stages | `agents/workflow.py` |

---

## Tech Stack

| Layer | Technology | Role |
|--------|------------|------|
| Language | **Python** | Agents, RAG, API |
| API | **FastAPI** | `POST /ingest`, `POST /query`, `GET /health` |
| UI | **Streamlit** | Demo UI calling the API (`frontend/app.py`) |
| Orchestration | **LangChain** | Prompts and LLM calls |
| LLM | **Groq API** | Fast inference for triage, writer, and compliance |
| Embeddings | **HuggingFace** `sentence-transformers/all-MiniLM-L6-v2` | Dense vectors for policies and queries |
| Vector store | **FAISS** | Persisted under `data/faiss_index/` |

---

## System Workflow

1. **Input**: Customer **ticket** text and optional **order context** JSON (dates, status, category, region, payment method, etc.).
2. **Triage**: Classifies the issue so retrieval can condition the query (ticket + label).
3. **Retriever**: Embeds the query, searches FAISS, returns the most relevant **policy chunks** (with IDs for citation).
4. **Writer**: Generates a draft **decision**, **rationale**, **customer-facing message**, and **citations** referencing only those chunk IDs. If retrieval is empty or below minimum evidence, the writer **escalates** instead of guessing.
5. **Compliance**: Confirms citations are valid, reduces unsupported wording, and may **rewrite** or **escalate** the draft.
6. **Output**: Single JSON object suitable for APIs, logs, or the Streamlit UI.

---

## RAG Pipeline

Implemented in `rag/ingest.py`, `rag/pipeline.py`, `rag/embeddings.py`, `rag/retriever.py`.

- **Ingestion**: Loads `.txt` / `.md` from `data/policies/`.
- **Cleaning**: Normalizes whitespace so chunks are consistent.
- **Chunking**: `RecursiveCharacterTextSplitter` with **500** characters and **50** overlap (configurable via `.env`)—large enough for a rule + exception, small enough to limit cross-topic noise.
- **Embedding**: Each chunk is embedded with the MiniLM model; queries use the same space.
- **FAISS**: Vectors are indexed and saved; retrieval returns top-**k** neighbors.
- **Grounding**: The writer prompt is restricted to retrieved text, so answers are **anchored** to corpus content; compliance enforces citation integrity.

---

## Output Format

Typical API response fields include classification from triage, clarifying questions, compliance-checked decision, rationale, citations, and customer-safe text. Shape is aligned with `agents/workflow.py` and compliance schema.

**Illustrative** response (values are examples only):

```json
{
  "classification": "refund",
  "confidence": 0.88,
  "clarifying_questions": [],
  "decision": "approve",
  "rationale": "Policy allows refund for salable items returned within 30 days of delivery; ticket states unopened and within window.",
  "citations": ["refund_policy_chunk_3", "shipping_policy_chunk_1"],
  "customer_response": "You're within our standard return window for an unopened item. I'll outline how to start your return and when to expect your refund once we receive it.",
  "internal_notes": ""
}
```

Escalation or `needs_info` decisions are returned when evidence is missing or ambiguous—still as structured JSON.

---

## Evaluation Strategy

- **Dataset**: **20** cases in `evaluation/test_cases.json`, each with `id`, `category`, `input`, and `expected` (`classification`, `decision`).
- **Categories**:
  - **normal** — straightforward policy fits (8)
  - **exception** — edge timelines, bundles, mis-shipment, hygiene, etc. (6)
  - **conflict** — overlapping or contradictory customer expectations (3)
  - **not_policy** — questions outside the policy corpus (3)
- **Runner**: `python evaluation/run_evaluation.py` loads `cases`, uses `input` (or legacy `ticket`), runs the full pipeline, and writes `evaluation/evaluation_report.json`.

**Metrics** (illustrative—depend on model and prompts):

| Metric | Meaning |
|--------|---------|
| **Citation coverage** | Share of cited chunk IDs that were actually retrieved |
| **Unsupported claim proxy** | Runs where `compliance.passed` is false |
| **Escalation correctness** | Match of final `decision == "escalate"` to expected escalation when derivable from `expected.decision` |

Triage labels today are **six-way** (`refund`, `shipping`, etc.); dataset `expected.classification` may use finer labels (e.g. `replacement`, `compensation`) for **offline** grading or future alignment.

---

## How to Run the Project

### 1. Clone and environment

```bash
git clone https://github.com/TIB713/ecommerce-support-agent.git
cd ecommerce-support-agent
python -m venv .venv
```

**Windows**

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment variables

Copy `.env.example` to `.env` and set **`GROQ_API_KEY`**.

### 3. Build the FAISS index

Start the API, then ingest:

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/ingest
```

Or call `ingest_policies()` from Python (see `rag/ingest.py`).

### 4. Query the API

```bash
curl -X POST http://127.0.0.1:8000/query ^
  -H "Content-Type: application/json" ^
  -d "{\"ticket\": \"I want to return an unopened blender delivered last week.\", \"order_context\": {\"order_status\": \"delivered\", \"delivery_date\": \"2026-03-20\", \"item_category\": \"home\"}}"
```

(Use `\` instead of `^` for line continuation on bash.)

### 5. Streamlit UI

```bash
streamlit run frontend/app.py
```

Point the UI at the same API host/port if not using defaults.

### 6. Evaluation

```bash
python evaluation/run_evaluation.py
```

Review `evaluation/evaluation_report.json`.

---

## Example Use Case

**Input (ticket)**

```text
My laptop arrived with a cracked screen. I have photos of the box and the damage. I need a replacement.
```

**Input (order context)** — optional but helps triage:

```json
{
  "order_date": "2026-03-01",
  "delivery_date": "2026-03-05",
  "item_category": "electronics",
  "order_status": "delivered",
  "shipping_region": "domestic",
  "fulfillment_type": "expedited",
  "payment_method": "credit_card"
}
```

**Output (sketch)**

- **classification**: e.g. `refund` or routing label triage maps damaged goods to.
- **decision**: often `approve` for replacement path when policy chunks support damage-in-transit / DOA flows.
- **citations**: chunk IDs from policies such as damaged-items or replacement rules.
- **customer_response**: concise next steps (RMA, photos, timeline)—wording varies per model run.

---

## Limitations

- **Residual hallucination risk**: Prompts and compliance reduce but do not **mathematically eliminate** unsupported text; edge cases can still slip through.
- **Corpus-bound**: Answers are only as good as **coverage and clarity** of `data/policies/`; real production needs vetted legal policy.
- **Retrieval errors**: Wrong or partial chunks can mislead the writer; no reranker is required in the baseline.
- **Evaluation**: Current metrics are **proxy** measures (e.g. compliance pass rate, citation overlap); they do not replace human QA or legal review.
- **External deps**: Requires **Groq** availability and accepts latency/cost tradeoffs of cloud LLM calls.

---

## Future Improvements

- **Metrics**: Per-label accuracy vs `expected.classification` / `expected.decision`, human rubrics, and regression suites on policy updates.
- **Retrieval**: Cross-encoder reranking, query expansion, or hybrid sparse+dense search.
- **UI**: Live citation preview, diff vs policy text, agent handoff button, role-based views.
- **Policies**: Connect to a **CMS or PDF pipeline**, versioned documents, and locale-specific corpora.
- **Safety & observability**: Structured tracing (OpenTelemetry), PII redaction, and A/B prompts.

---

## Project Structure

```text
backend/           # FastAPI app (/ingest, /query, /health)
agents/            # Triage, retriever, writer, compliance, workflow
rag/               # Ingest, chunk, embed, FAISS, retrieve
data/policies/     # Policy corpus (.txt / .md)
data/faiss_index/  # Generated index (after ingest)
frontend/          # Streamlit client
evaluation/        # test_cases.json, run_evaluation.py, reports
utils/             # Settings, schemas, helpers
.env               # Secrets (create from .env.example)
```

---

## License / disclaimer

Example policies are **synthetic** and for **research / portfolio** use—not legal advice. Replace with counsel-approved text before any production customer-facing deployment.
