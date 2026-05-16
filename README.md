# 🤖 LumoraLab Brand Voice

> RAG-powered AI content generator that writes in your brand voice — backed by a custom evaluation harness with measurable quality scores.

**Built by:** [Natthaporn Gulgalkhai](https://linkedin.com/in/natthapon-gulgalkhai) · AI Automation Specialist
**Stack:** Python · FastAPI · Claude Sonnet 4.5 · Voyage AI · ChromaDB · DeepEval

---

## ✨ What it does

Most businesses can't scale content creation because their team can't capture their unique voice. This system fixes that:

1. Upload past brand content (blogs, emails, social posts)
2. System chunks, embeds, and stores it in a vector database
3. User requests new content ("Write a LinkedIn post about X")
4. Retrieval finds the most relevant brand examples
5. Claude generates new content in the brand's voice — measurably

End-to-end RAG pipeline in under 5 seconds.

---

## 📊 Evaluation results

This system includes a custom evaluation harness that measures real quality across 3 dimensions using a Claude-as-judge methodology:

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Voice consistency | **9.0/10** | 5.0/10 | ✓ Well above |
| Retrieval precision | **58%** | 40% | ✓ Above |
| Faithfulness | **5.0/10** | 5.0/10 | ✓ At threshold |
| Pass rate | **67%** (2/3) | 50% | ✓ Above |
| Pipeline errors | **0** | — | ✓ Clean |

Tests run in **61 seconds**. Full eval report generated as HTML.

---

## 🏗️ Architecture

```
Past brand content → chunk → embed (Voyage AI) → ChromaDB
↓
User query → embed → semantic search → top-k chunks
↓
Claude + retrieved context → generated content
↓
Evaluation harness → voice/retrieval/faithfulness scores
```

---

## 🎯 Key engineering decisions

**Claude-as-judge for voice scoring.** Instead of fuzzy keyword matching, the eval system asks Claude to score how well generated content matches defined brand voice traits. Reproducible, explainable, and surfaces *why* a score is what it is.

**Sentence-aware chunking with overlap.** Chunks at 800 chars with 100 char overlap, splitting at paragraph then sentence boundaries — context isn't lost at chunk edges.

**Per-content-type system prompts.** LinkedIn posts, blog posts, emails, and social media each get different generation instructions while sharing the same retrieval logic.

**Fail-fast config validation.** Server validates all required env vars on startup. Clear error before first request beats confusing 500s later.

**Non-fatal evaluation.** Eval failures don't break the API — they get logged and scored. The pipeline degrades gracefully.

---

## 🛠️ Tech stack

| Layer | Tool | Why |
|-------|------|-----|
| Web framework | FastAPI | Async-first, automatic validation |
| AI | Claude Sonnet 4.5 | Best-in-class for nuanced voice matching |
| Embeddings | Voyage AI (voyage-3) | Anthropic's recommended partner |
| Vector DB | ChromaDB | Persistent local storage, no infra needed |
| Evaluation | DeepEval + pytest | Standard Python eval framework |
| Frontend | Vanilla HTML/JS + Tailwind | Single-file, no build step |

---

## 🚀 Run locally

```bash
git clone https://github.com/Natt1111/lumoralab-brand-voice.git
cd lumoralab-brand-voice
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and VOYAGE_API_KEY
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## 🧪 Run evaluation

```bash
source venv/bin/activate
pytest evaluation/test_rag_eval.py -v -s
```

Generates HTML report at `evaluation/reports/`.

---

## 📁 Project structure

```
lumoralab-brand-voice/
├── main.py                        # FastAPI app, CORS, lifespan startup validation
├── requirements.txt
├── .env.example                   # API key template (never commit .env)
├── test_ingestion.py              # Standalone Phase 1 smoke test: chunk → embed → store
├── test_generation.py             # Standalone Phase 2 smoke test: retrieve → generate
│
├── app/
│   ├── config.py                  # Loads .env, validates required keys on startup
│   ├── models.py                  # Pydantic models: ingest, retrieval, generation
│   ├── ingestion.py               # chunk_text → embed_chunks → store in ChromaDB
│   ├── retrieval.py               # embed_query → semantic search → SearchResult list
│   ├── generator.py               # retrieve → build prompt → call Claude → response
│   └── routes/
│       ├── ingest.py              # POST /api/ingest
│       └── generate.py            # POST /api/generate
│
├── evaluation/
│   ├── test_dataset.json          # 10 test cases with expected themes + voice traits
│   ├── metrics.py                 # VoiceConsistency, RetrievalPrecision, Faithfulness
│   ├── runner.py                  # Runs dataset through RAG, returns EvalResult list
│   ├── report.py                  # Generates Markdown + HTML evaluation reports
│   └── test_rag_eval.py           # pytest suite (11 tests, session-scoped fixtures)
│
├── static/
│   └── index.html                 # Single-file dark-themed UI (Tailwind + vanilla JS)
│
└── data/
    └── chroma/                    # ChromaDB persistent store (git-ignored)
```

---

## 🎨 Frontend

Dark-themed interface with:
- Tabbed UI (Ingest + Generate)
- Live status indicator (pulsing green dot)
- Streaming reasoning trace
- Source attribution cards with similarity scores
- Self-typing generated content with copy-to-clipboard
- Mobile responsive

---

## 🧠 What I learned

- **Eval matters more than building.** Without measurable quality, you don't know if your RAG works — you just hope it does.
- **Claude-as-judge is reproducible.** Custom metrics that use an LLM to score outputs feel "soft" but produce consistent, explainable results.
- **Chunking strategy dominates retrieval quality.** Even good embeddings can't fix bad chunks.
- **Fail-fast saves debugging hours.** Validating env vars at startup turns a 30-minute mystery into a 2-second error.

---

## 👋 About me

AI Automation Specialist transitioning into AI Engineering. Background in healthcare (dental) and hospitality operations gives me real-world domain expertise that pure tech engineers often lack.

- 🌐 Portfolio: [lumoralab.lovable.app](https://lumoralab.lovable.app)
- 💼 LinkedIn: [linkedin.com/in/natthapon-gulgalkhai](https://linkedin.com/in/natthapon-gulgalkhai)
- 📧 Email: ngulgalkhai@gmail.com
- 📍 Florida, USA · Open to relocation
