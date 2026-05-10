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
# Generation models (Phase 2)
# ---------------------------------------------------------------------------

ContentType = Literal["blog_post", "social_post", "email", "case_study", "ad_copy"]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="What you want the content to be about")
    content_type: ContentType = Field(default="blog_post")
    tone: str = Field(default="professional yet approachable")


class RetrievedChunk(BaseModel):
    chunk_id: str
    source_file: str
    content: str
    distance: float


class GenerateResponse(BaseModel):
    content_type: ContentType
    generated_content: str
    retrieved_chunks: list[RetrievedChunk]
