from fastapi import APIRouter, HTTPException

from app.ingestion import ingest_document
from app.models import IngestRequest, IngestResponse

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest a piece of text into the vector store."""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="'text' field cannot be empty.")
    try:
        return ingest_document(request.text, request.source, request.metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")
