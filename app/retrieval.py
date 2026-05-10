# TODO: Import voyageai and chromadb

from app.config import settings
from app.models import RetrievedChunk


# TODO: Initialize Voyage AI client (can share singleton with ingestion.py later)
# voyage_client = voyageai.Client(api_key=settings.voyage_api_key)

# TODO: Initialize ChromaDB client and reference the same collection as ingestion
# chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
# collection = chroma_client.get_or_create_collection(name=settings.collection_name)


def embed_query(query: str) -> list[float]:
    """Embed a user query with Voyage AI using the query input type."""
    # TODO: Implement query embedding:
    #   result = voyage_client.embed([query], model="voyage-3", input_type="query")
    #   return result.embeddings[0]
    raise NotImplementedError


def search(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Semantic search: embed query → query ChromaDB → return ranked chunks."""
    # TODO: Implement retrieval:
    #   1. query_embedding = embed_query(query)
    #   2. results = collection.query(
    #          query_embeddings=[query_embedding],
    #          n_results=top_k,
    #          include=["documents", "metadatas", "distances"],
    #      )
    #   3. Zip results together into list[RetrievedChunk]
    #   4. Return the list (ChromaDB returns lowest distance = most similar)
    raise NotImplementedError
