# TODO: Import anthropic

from app.config import settings
from app.models import GenerateRequest, GenerateResponse, RetrievedChunk
from app.retrieval import search


# TODO: Initialize Anthropic client
# anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

MODEL = "claude-sonnet-4-5"


def build_system_prompt(content_type: str, tone: str) -> str:
    """Build the system prompt that establishes LumoraLab brand voice."""
    # TODO: Craft a detailed system prompt that:
    #   - Identifies Claude as a LumoraLab brand content writer
    #   - Describes the brand's voice, values, and audience
    #   - Sets expectations for tone and content_type formatting
    #   - Instructs Claude to use the retrieved context faithfully
    return f"You are a content writer for LumoraLab. Write in a {tone} tone. Produce a {content_type}."


def build_user_message(request: GenerateRequest, context_chunks: list[RetrievedChunk]) -> str:
    """Assemble the user message with retrieved context + the generation prompt."""
    # TODO: Format the message so Claude clearly sees:
    #   1. A labeled "BRAND CONTEXT" block with the retrieved chunks
    #   2. The user's generation request / prompt
    #   Example structure:
    #
    #   ## Brand Context (retrieved from LumoraLab content)
    #   [Chunk 1 — source: ...]
    #   ...
    #
    #   ## Your Task
    #   {request.prompt}
    context = "\n\n".join(
        f"[Source: {c.source_file}]\n{c.content}" for c in context_chunks
    )
    return f"## Brand Context\n\n{context}\n\n## Your Task\n\n{request.prompt}"


async def generate_content(request: GenerateRequest) -> GenerateResponse:
    """Full generation pipeline: retrieve context → build prompt → call Claude."""
    # TODO: Implement the full pipeline:
    #   1. context_chunks = search(request.prompt, top_k=5)
    #   2. system = build_system_prompt(request.content_type, request.tone)
    #   3. user_message = build_user_message(request, context_chunks)
    #   4. response = anthropic_client.messages.create(
    #          model=MODEL,
    #          max_tokens=2048,
    #          system=system,
    #          messages=[{"role": "user", "content": user_message}],
    #      )
    #   5. Return GenerateResponse with generated text + retrieved chunks
    raise NotImplementedError
