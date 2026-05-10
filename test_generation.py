"""
End-to-end test for Phase 2: Retrieval + Generation.
Run with: venv/bin/python test_generation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion import ingest_document
from app.retrieval import _get_collection
from app.generator import generate_content
from app.models import GenerateRequest

SAMPLE_TEXT = """
LumoraLab helps small businesses automate their workflows with AI-powered tools that are intuitive and affordable. We believe that cutting-edge technology should not be reserved for enterprise companies alone.

Our platform connects directly with the tools you already use — from scheduling and invoicing to customer communication and reporting. With LumoraLab, you spend less time on repetitive tasks and more time doing work that matters.

We built LumoraLab because we saw brilliant small business owners drowning in administrative work. A bakery owner spending Sunday nights reconciling receipts. A freelance designer chasing five different clients for overdue invoices. A yoga studio manager manually sending appointment reminders one by one.

These are not problems born of incompetence. They are problems born of having too much to do and too little time. LumoraLab exists to fix that.

Our AI doesn't just execute tasks — it learns your preferences, adapts to your workflows, and gets smarter the more you use it. Think of it as hiring a very capable operations assistant who never sleeps, never forgets, and never charges overtime.

Small businesses are the backbone of every community. The dentist who remembers your kids' names. The bookshop that stocks exactly what your neighbourhood reads. The independent gym where the trainer knows your injury history. These businesses deserve tools that work as hard as they do.

LumoraLab integrates with your existing calendar, email, and invoicing software. Setup takes minutes, not months. There's no IT department required, no six-week onboarding, and no contract that locks you in for three years. Just automation that works from day one.

We measure our success by one thing: how much time we give back to you each week. Our customers typically save 8–12 hours of administrative work per week. That's a full workday returned to doing what you actually love.
"""


def ensure_ingested() -> None:
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        print("Vector store is empty — ingesting sample content first...")
        result = ingest_document(
            text=SAMPLE_TEXT,
            source="lumoralab_about.txt",
            metadata={"content_type": "about"},
        )
        print(f"  Ingested {result.chunks_created} chunks from '{result.source}'\n")
    else:
        print(f"Vector store already has {count} chunks — skipping ingestion.\n")


def test_generation() -> None:
    print("=== Generation Test ===")
    request = GenerateRequest(
        prompt="Write a LinkedIn post about how AI can help small dental practices",
        content_type="linkedin_post",
        top_k=5,
    )

    result = generate_content(request)

    print(f"Sources used   : {result.retrieved_sources}")
    print(f"Chunks used    : {result.num_chunks_used}")
    print(f"\n{'─' * 60}")
    print(result.generated_content)
    print(f"{'─' * 60}\n")

    assert result.generated_content, "FAILED — generated_content is empty"
    assert result.num_chunks_used > 0, "FAILED — no chunks were used"
    assert result.retrieved_sources, "FAILED — retrieved_sources is empty"
    print("All assertions passed.")


if __name__ == "__main__":
    ensure_ingested()
    test_generation()
