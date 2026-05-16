"""
Custom evaluation metrics for the LumoraLab RAG system.

Three metrics:
  VoiceConsistencyMetric  — Claude-as-judge: does the output match LumoraLab's brand voice?
  RetrievalPrecisionMetric — keyword match: did retrieved chunks contain the expected themes?
  FaithfulnessMetric      — Claude-as-judge: is the output grounded in the retrieved context?
"""
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from app.config import settings

# Haiku is used for the judge calls — fast and cheap, still capable of scoring tasks.
JUDGE_MODEL = "claude-haiku-4-5-20251001"
_judge = anthropic.Anthropic(api_key=settings.anthropic_api_key)

BRAND_VOICE_SUMMARY = """\
LumoraLab brand voice characteristics:
- Human-first: starts from real small-business struggles, not from technology features
- Clear and direct: plain language, no corporate buzzwords or jargon
- Quietly confident: shows rather than shouts; authoritative without being arrogant
- Story-driven: leads with concrete scenarios (bakery owner, dental practice, yoga studio)
- Optimistic but grounded: backs every claim with something specific and real
- Banned phrases: "game-changer", "leverage", "cutting-edge", "seamlessly", "robust solution"\
"""


@dataclass
class MetricResult:
    name: str
    score: float        # 0.0 – 1.0 (normalised)
    raw_score: float    # original value before normalisation (e.g. 1–10 for judge metrics)
    reason: str
    passed: bool
    threshold: float


def _ask_judge(prompt: str) -> float:
    """Send a scoring prompt to the judge model. Expects a single integer 1–10 in the reply."""
    resp = _judge.messages.create(
        model=JUDGE_MODEL,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    m = re.search(r"\d+", raw)
    return float(m.group()) if m else 5.0


class VoiceConsistencyMetric:
    """
    Uses Claude Haiku as a judge to score how well generated content matches
    LumoraLab's brand voice. Score 1–10, normalised to 0–1.

    Threshold: 0.6 (i.e. ≥ 6/10).
    """

    name = "voice_consistency"
    threshold = 0.6

    def measure(self, generated_content: str, **_) -> MetricResult:
        prompt = f"""{BRAND_VOICE_SUMMARY}

--- GENERATED CONTENT ---
{generated_content[:2500]}
--- END ---

Score this content from 1 to 10 based on how well it matches LumoraLab's brand voice.

Scoring guide:
9-10 Excellent — empathetic, story-driven, concrete, zero buzzwords
7-8  Good       — mostly aligned, minor generic moments
5-6  Acceptable — some alignment but feels like stock AI copy in places
3-4  Poor       — generic, over-formal, or uses banned phrases
1-2  Failing    — contradicts brand voice entirely

Reply with ONLY a single integer (1–10). Nothing else."""

        raw = _ask_judge(prompt)
        score = min(raw, 10.0) / 10.0

        return MetricResult(
            name=self.name,
            score=score,
            raw_score=raw,
            reason=f"Claude judge rated {raw:.0f}/10 for brand voice alignment",
            passed=score >= self.threshold,
            threshold=self.threshold,
        )


class RetrievalPrecisionMetric:
    """
    Keyword-based precision check: what fraction of the expected themes appear
    in the combined text of retrieved chunks?

    No API call — pure string matching.
    Threshold: 0.4 (at least 40 % of expected themes present).
    """

    name = "retrieval_precision"
    threshold = 0.4

    def measure(
        self,
        retrieved_chunks: list[str],
        expected_themes: list[str],
        **_,
    ) -> MetricResult:
        if not expected_themes or not retrieved_chunks:
            return MetricResult(
                name=self.name,
                score=0.0,
                raw_score=0.0,
                reason="No themes or chunks provided",
                passed=False,
                threshold=self.threshold,
            )

        haystack = " ".join(retrieved_chunks).lower()
        hits = [t for t in expected_themes if t.lower() in haystack]
        score = len(hits) / len(expected_themes)

        return MetricResult(
            name=self.name,
            score=score,
            raw_score=score,
            reason=f"{len(hits)}/{len(expected_themes)} themes found: {hits}",
            passed=score >= self.threshold,
            threshold=self.threshold,
        )


class FaithfulnessMetric:
    """
    Uses Claude Haiku as a judge to score whether generated content stays
    grounded in the retrieved context (not hallucinating claims).

    Score 1–10, normalised to 0–1. Threshold: 0.5 (≥ 5/10).
    """

    name = "faithfulness"
    threshold = 0.5

    def measure(
        self,
        generated_content: str,
        retrieved_chunks: list[str],
        **_,
    ) -> MetricResult:
        if not retrieved_chunks:
            return MetricResult(
                name=self.name,
                score=0.0,
                raw_score=0.0,
                reason="No retrieved context to evaluate against",
                passed=False,
                threshold=self.threshold,
            )

        context = "\n\n".join(
            f"[Chunk {i+1}]: {c[:500]}" for i, c in enumerate(retrieved_chunks[:5])
        )

        prompt = f"""--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---

--- GENERATED CONTENT ---
{generated_content[:2500]}
--- END ---

Score from 1 to 10 how faithful the generated content is to the retrieved context.

Scoring guide:
9-10 Fully grounded  — every claim traces back to the context
7-8  Mostly faithful — minor elaborations that don't contradict context
5-6  Partial         — some claims go beyond what the context supports
3-4  Loose           — many claims unsupported by context
1-2  Unfaithful      — content contradicts or ignores the context

Reply with ONLY a single integer (1–10). Nothing else."""

        raw = _ask_judge(prompt)
        score = min(raw, 10.0) / 10.0

        return MetricResult(
            name=self.name,
            score=score,
            raw_score=raw,
            reason=f"Claude judge rated {raw:.0f}/10 for faithfulness to retrieved context",
            passed=score >= self.threshold,
            threshold=self.threshold,
        )
