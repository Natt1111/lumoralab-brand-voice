from fastapi import APIRouter, HTTPException

from app.generator import generate_content
from app.models import GenerateRequest, GenerateResponse

router = APIRouter(tags=["generation"])


@router.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    """Generate brand-voice content using retrieved context from the vector store."""
    try:
        return generate_content(request)
    except ValueError as exc:
        # Raised when the vector store is empty
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")
