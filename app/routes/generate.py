from fastapi import APIRouter, HTTPException

from app.generator import generate_content
from app.models import GenerateRequest, GenerateResponse

router = APIRouter(tags=["generation"])


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate brand-voice content using retrieved context from the vector store."""
    # TODO: Add a guard that checks the ChromaDB collection is non-empty
    #       and returns a helpful 400 error if no documents have been ingested yet

    try:
        result = await generate_content(request)
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="Generation pipeline not yet implemented. See app/generator.py TODOs.",
        )
    except Exception as exc:
        # TODO: Replace broad exception with specific error types as pipeline is built
        raise HTTPException(status_code=500, detail=str(exc))

    return result
