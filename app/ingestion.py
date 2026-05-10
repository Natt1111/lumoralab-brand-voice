import re
import uuid
from pathlib import Path
from typing import Optional

import voyageai
import chromadb

from app.config import settings
from app.models import IngestResponse

voyage_client = voyageai.Client(api_key=settings.voyage_api_key)

_collection = None


def _get_collection():
    """Lazily initialize the ChromaDB collection so the directory is created on first use."""
    global _collection
    if _collection is None:
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = client.get_or_create_collection(name=settings.collection_name)
    return _collection


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries.

    Strategy:
      1. Normalize excess blank lines, then split into paragraphs.
      2. Split each paragraph into sentences on punctuation boundaries.
      3. Accumulate sentences into chunks up to `chunk_size` characters.
      4. When flushing a chunk, roll back to keep the last ~`overlap` characters
         as the start of the next chunk so context is preserved across boundaries.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    sentence_re = re.compile(r"(?<=[.!?])\s+")

    sentences: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            sentences.extend(s.strip() for s in sentence_re.split(para) if s.strip())

    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s_len = len(sentence) + 1  # +1 for the joining space

        if current_len + s_len > chunk_size and current:
            chunks.append(" ".join(current))

            # Build overlap tail: walk backwards keeping sentences that fit
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                needed = len(s) + 1
                if overlap_len + needed <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += needed
                else:
                    break
            current = overlap_sents
            current_len = overlap_len

        current.append(sentence)
        current_len += s_len

    if current:
        chunks.append(" ".join(current))

    return chunks


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed text chunks with Voyage AI, batching up to 128 at a time."""
    embeddings: list[list[float]] = []
    for i in range(0, len(chunks), 128):
        batch = chunks[i : i + 128]
        result = voyage_client.embed(batch, model="voyage-3", input_type="document")
        embeddings.extend(result.embeddings)
    return embeddings


def store_chunks(
    chunks: list[str],
    embeddings: list[list[float]],
    source: str,
    metadata: Optional[dict],
) -> int:
    """Upsert chunks and embeddings into ChromaDB. Returns count stored."""
    collection = _get_collection()

    base_meta: dict = {"source": source}
    if metadata:
        # ChromaDB only accepts str | int | float | bool as metadata values
        for k, v in metadata.items():
            base_meta[k] = v if isinstance(v, (str, int, float, bool)) else str(v)

    collection.upsert(
        ids=[str(uuid.uuid4()) for _ in chunks],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[base_meta] * len(chunks),
    )
    return len(chunks)


def ingest_document(
    text: str, source: str, metadata: Optional[dict] = None
) -> IngestResponse:
    """Full ingestion pipeline: chunk → embed → store."""
    chunks = chunk_text(text)
    if not chunks:
        return IngestResponse(chunks_created=0, source=source, success=False)

    embeddings = embed_chunks(chunks)
    count = store_chunks(chunks, embeddings, source, metadata)
    return IngestResponse(chunks_created=count, source=source, success=True)
