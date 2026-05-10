# LumoraLab Brand Voice Generator

> A RAG-powered content generator that learns from real brand content and produces new content in your voice.

<!-- TODO: Fill in full project description, architecture overview, and demo screenshots -->

## Tech Stack

- **FastAPI** — web framework and REST API
- **Anthropic Claude Sonnet 4.5** — content generation
- **Voyage AI (voyage-3)** — semantic embeddings
- **ChromaDB** — local vector database for retrieval
- **Python 3.10+**

## Getting Started

> **Note:** Use Python 3.11 (not 3.12+). ChromaDB's `onnxruntime` dependency has no wheel for Python 3.12/3.13 on macOS x86_64.

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd lumoralab-brand-voice

# 2. Create a Python 3.11 virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env and add your API keys

# 5. Run the server
uvicorn main:app --reload
```

<!-- TODO: Add API usage examples, ingestion walkthrough, and deployment notes -->
