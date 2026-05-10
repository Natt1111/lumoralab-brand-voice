"""
Quick end-to-end test for the ingestion pipeline.
Run with: python test_ingestion.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.ingestion import chunk_text, ingest_document

SAMPLE_TEXT = """
LumoraLab helps small businesses automate their workflows with AI-powered tools that are intuitive and affordable. We believe that cutting-edge technology should not be reserved for enterprise companies alone.

Our platform connects directly with the tools you already use — from scheduling and invoicing to customer communication and reporting. With LumoraLab, you spend less time on repetitive tasks and more time doing work that matters.

We built LumoraLab because we saw brilliant small business owners drowning in administrative work. A bakery owner spending Sunday nights reconciling receipts. A freelance designer chasing five different clients for overdue invoices. A yoga studio manager manually sending appointment reminders one by one.

These are not problems born of incompetence. They are problems born of having too much to do and too little time. LumoraLab exists to fix that.

Our AI doesn't just execute tasks — it learns your preferences, adapts to your workflows, and gets smarter the more you use it. Think of it as hiring a very capable operations assistant who never sleeps, never forgets, and never charges overtime.
"""


def test_chunking():
    print("=== 1. Chunking (no API calls) ===")
    chunks = chunk_text(SAMPLE_TEXT, chunk_size=300, overlap=60)
    print(f"  Chunks created : {len(chunks)}")
    print(f"  First chunk    : {chunks[0][:120]}...")
    print(f"  Avg length     : {sum(len(c) for c in chunks) // len(chunks)} chars")
    print()
    return chunks


def test_full_pipeline():
    print("=== 2. Full pipeline (embed + store) ===")
    result = ingest_document(
        text=SAMPLE_TEXT,
        source="lumoralab_about.txt",
        metadata={"content_type": "about", "version": "1"},
    )
    print(f"  Success        : {result.success}")
    print(f"  Source         : {result.source}")
    print(f"  Chunks stored  : {result.chunks_created}")
    print()
    return result


if __name__ == "__main__":
    test_chunking()
    result = test_full_pipeline()

    if result.success:
        print("All tests passed.")
    else:
        print("Ingestion returned success=False — check your input text.")
        sys.exit(1)
