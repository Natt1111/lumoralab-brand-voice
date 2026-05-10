from pathlib import Path

import voyageai
import chromadb

from app.config import settings
from app.models import DocumentChunk, SearchResult

voyage_client = voyageai.Client(api_key=settings.voyage_api_key)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = client.get_or_create_collection(name=settings.collection_name)
    return _collection


def embed_query(query: str) -> list[float]:
    """Embed a query string with Voyage AI using the 'query' input type."""
    result = voyage_client.embed([query], model="voyage-3", input_type="query")
    return result.embeddings[0]


def search_similar(query: str, top_k: int = 5) -> list[SearchResult]:
    """Embed query → find top_k nearest chunks in ChromaDB → return as SearchResults.

    Similarity score is derived from the L2 distance ChromaDB returns.
    Voyage AI embeddings are unit-normalized, so for two normalized vectors:
      cosine_similarity = 1 - L2² / 2   →   score = max(0, 1 - distance / 2)
    Score of 1.0 = identical, 0.0 = completely dissimilar.
    """
    collection = _get_collection()
    stored = collection.count()
    if stored == 0:
        return []

    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, stored),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    search_results: list[SearchResult] = []
    for doc, meta, dist in zip(docs, metas, dists):
        chunk = DocumentChunk(
            id=meta.get("id", ""),
            text=doc,
            source=meta.get("source", "unknown"),
            metadata={k: v for k, v in meta.items() if k not in ("id", "source")},
        )
        similarity_score = round(max(0.0, 1.0 - dist / 2), 4)
        search_results.append(SearchResult(chunk=chunk, similarity_score=similarity_score))

    return search_results
