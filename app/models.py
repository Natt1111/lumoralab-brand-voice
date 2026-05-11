from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    id: str
    text: str
    source: str
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    text: str = Field(..., description="Raw text content to ingest")
    source: str = Field(..., description="Identifier for this document (e.g. filename or URL)")
    metadata: Optional[dict] = Field(default=None, description="Optional extra metadata to store")


class IngestResponse(BaseModel):
    chunks_created: int
    source: str
    success: bool


# ---------------------------------------------------------------------------
# Retrieval models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    chunk: DocumentChunk
    similarity_score: float


# ---------------------------------------------------------------------------
# Generation models
# ---------------------------------------------------------------------------

ContentType = Literal["linkedin_post", "blog_post", "email", "social_media", "general"]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="What you want the content to be about")
    content_type: ContentType = Field(default="linkedin_post")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve")


class ChunkPreview(BaseModel):
    source: str
    text_preview: str
    similarity_score: float


class GenerateResponse(BaseModel):
    generated_content: str
    retrieved_sources: list[str]
    num_chunks_used: int
    chunks: list[ChunkPreview] = Field(default_factory=list)
