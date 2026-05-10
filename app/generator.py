import anthropic

from app.config import settings
from app.models import GenerateRequest, GenerateResponse, SearchResult
from app.retrieval import search_similar

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

MODEL = "claude-sonnet-4-5"

# Format instructions keyed by content_type — these guide both length and structure.
_FORMAT_GUIDANCE: dict[str, str] = {
    "linkedin_post": (
        "a LinkedIn post (150–300 words). Open with a hook — a concrete scenario or "
        "surprising insight, not a question. Use short paragraphs for mobile readability. "
        "Include 2–3 key insights from the topic. Close with a genuine call to action or "
        "an open question that invites discussion. No hashtag blocks — at most 2–3 inline."
    ),
    "blog_post": (
        "a blog post (450–750 words). Use a compelling headline, a short relatable intro, "
        "2–4 body sections with optional subheadings, and a concise conclusion with a "
        "takeaway. Conversational but authoritative tone throughout."
    ),
    "email": (
        "a professional email. First line: 'Subject: <subject here>'. Then a brief, "
        "warm intro (1–2 sentences). Core message in 2–3 short paragraphs. End with a "
        "single, clear call to action. Keep total length under 200 words."
    ),
    "social_media": (
        "a social media post. For Twitter/X: under 280 characters, punchy and direct. "
        "For other platforms: 80–150 words, casual and conversational. Emojis are fine "
        "if they add meaning, not decoration. One clear point — don't try to say everything."
    ),
    "general": (
        "a piece of content with a balanced, professional yet approachable tone. "
        "Adapt length and structure to suit the topic naturally — don't over-format."
    ),
}


def build_system_prompt(content_type: str) -> str:
    format_note = _FORMAT_GUIDANCE.get(content_type, _FORMAT_GUIDANCE["general"])
    return f"""You are the content writer for LumoraLab — an AI automation platform built specifically for small businesses.

LumoraLab's brand voice:
- Human-first and empathetic: we start from the real frustrations of running a small business, not from technology
- Clear and direct: plain language only — no jargon, no corporate buzzwords, no throat-clearing
- Quietly confident: authoritative without being condescending; we show, we don't shout
- Optimistic but grounded: we believe AI genuinely helps, and we back every claim with something concrete
- Story-driven: we lead with real scenarios (a bakery owner, a freelance designer, a yoga studio manager) before we talk about solutions

Your task: write {format_note}

Rules you must follow:
1. Study the STYLE EXAMPLES below carefully. Absorb the sentence rhythm, word choices, level of specificity, and paragraph length. Then set them aside.
2. Do NOT quote the examples directly or reference them in any way — use them only to calibrate your voice.
3. Write as LumoraLab. Use "we" naturally when referring to the company.
4. Banned phrases: "In today's fast-paced world", "game-changer", "leverage", "cutting-edge", "unlock your potential", "seamlessly", "robust solution", "at the end of the day".
5. Be specific. Replace "many small businesses" with "the bakery owner, the freelance designer, the yoga studio" — concrete always beats vague.
6. Output only the final content. No preamble, no "Here is your post:", no meta-commentary."""


def build_user_message(request: GenerateRequest, results: list[SearchResult]) -> str:
    examples = "\n\n---\n\n".join(
        f"[Source: {r.chunk.source} | Relevance: {r.similarity_score:.2f}]\n{r.chunk.text}"
        for r in results
    )
    return f"""## STYLE EXAMPLES
(Retrieved from real LumoraLab brand content — use to calibrate voice and tone only)

{examples}

---

## YOUR TASK

{request.prompt}"""


def generate_content(request: GenerateRequest) -> GenerateResponse:
    """Full generation pipeline: retrieve context → build prompt → call Claude."""
    results = search_similar(request.prompt, top_k=request.top_k)

    if not results:
        raise ValueError(
            "The vector store is empty. Ingest some LumoraLab content first via POST /api/ingest."
        )

    system = build_system_prompt(request.content_type)
    user_message = build_user_message(request, results)

    response = _client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0.7,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )

    generated_text = response.content[0].text

    # Deduplicate sources while preserving retrieval order
    seen: set[str] = set()
    sources: list[str] = []
    for r in results:
        if r.chunk.source not in seen:
            seen.add(r.chunk.source)
            sources.append(r.chunk.source)

    return GenerateResponse(
        generated_content=generated_text,
        retrieved_sources=sources,
        num_chunks_used=len(results),
    )
