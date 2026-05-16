"""
Evaluation runner: loads the test dataset, runs each case through the RAG pipeline,
scores with all three metrics, and returns a list of EvalResult objects.
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.generator import generate_content
from app.models import GenerateRequest
from app.retrieval import search_similar
from evaluation.metrics import (
    FaithfulnessMetric,
    MetricResult,
    RetrievalPrecisionMetric,
    VoiceConsistencyMetric,
)

DATASET_PATH = Path(__file__).parent / "test_dataset.json"


@dataclass
class EvalResult:
    test_id: str
    query: str
    content_type: str
    generated_content: str
    retrieved_sources: list[str]
    num_chunks: int
    # Per-metric results
    voice: MetricResult
    retrieval: MetricResult
    faithfulness: MetricResult
    # Rolled-up
    passed: bool
    duration_seconds: float
    error: Optional[str] = None

    @property
    def voice_score(self) -> float:
        return self.voice.score

    @property
    def retrieval_precision(self) -> float:
        return self.retrieval.score

    @property
    def faithfulness_score(self) -> float:
        return self.faithfulness.score


_DUMMY_METRIC = MetricResult(
    name="error", score=0.0, raw_score=0.0,
    reason="skipped due to error", passed=False, threshold=0.0,
)


def run_evaluation(
    dataset_path: Path = DATASET_PATH,
    sample_size: Optional[int] = None,
    verbose: bool = True,
) -> list[EvalResult]:
    """
    Run the full evaluation suite.

    Args:
        dataset_path:  Path to the JSON test dataset.
        sample_size:   If set, only run the first N cases (handy for quick checks).
        verbose:       Print per-case results to stdout.

    Returns:
        List of EvalResult, one per test case.
    """
    with open(dataset_path) as fh:
        dataset: list[dict] = json.load(fh)

    if sample_size:
        dataset = dataset[:sample_size]

    voice_metric      = VoiceConsistencyMetric()
    retrieval_metric  = RetrievalPrecisionMetric()
    faithful_metric   = FaithfulnessMetric()

    results: list[EvalResult] = []

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  LumoraLab RAG Evaluation — {len(dataset)} test cases")
        print(f"{'─'*60}")

    for case in dataset:
        t0 = time.time()
        tid = case["id"]

        try:
            # ── 1. Generate content ─────────────────────────────────────────
            request = GenerateRequest(
                prompt=case["query"],
                content_type=case["content_type"],
                top_k=5,
            )
            response = generate_content(request)

            # ── 2. Fetch full chunk texts for faithfulness (separate call) ──
            # generate_content() only exposes 80-char previews; search_similar()
            # gives us the full text needed for a meaningful faithfulness score.
            search_results = search_similar(case["query"], top_k=5)
            full_chunks = [r.chunk.text for r in search_results]

            # ── 3. Score ────────────────────────────────────────────────────
            voice_r      = voice_metric.measure(generated_content=response.generated_content)
            retrieval_r  = retrieval_metric.measure(
                retrieved_chunks=full_chunks,
                expected_themes=case.get("expected_themes", []),
            )
            faithful_r   = faithful_metric.measure(
                generated_content=response.generated_content,
                retrieved_chunks=full_chunks,
            )

            passed = voice_r.passed and retrieval_r.passed and faithful_r.passed
            duration = round(time.time() - t0, 1)

            result = EvalResult(
                test_id=tid,
                query=case["query"],
                content_type=case["content_type"],
                generated_content=response.generated_content,
                retrieved_sources=response.retrieved_sources,
                num_chunks=response.num_chunks_used,
                voice=voice_r,
                retrieval=retrieval_r,
                faithfulness=faithful_r,
                passed=passed,
                duration_seconds=duration,
            )

            if verbose:
                status = "✓ PASS" if passed else "✗ FAIL"
                print(
                    f"  [{tid}] {status}  "
                    f"voice={voice_r.raw_score:.0f}/10  "
                    f"retrieval={retrieval_r.score:.0%}  "
                    f"faithful={faithful_r.raw_score:.0f}/10  "
                    f"({duration}s)"
                )

        except Exception as exc:
            duration = round(time.time() - t0, 1)
            result = EvalResult(
                test_id=tid,
                query=case["query"],
                content_type=case["content_type"],
                generated_content="",
                retrieved_sources=[],
                num_chunks=0,
                voice=_DUMMY_METRIC,
                retrieval=_DUMMY_METRIC,
                faithfulness=_DUMMY_METRIC,
                passed=False,
                duration_seconds=duration,
                error=str(exc),
            )
            if verbose:
                print(f"  [{tid}] ERROR — {exc}")

        results.append(result)

    if verbose:
        _print_summary(results)

    return results


def _print_summary(results: list[EvalResult]) -> None:
    valid = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    print(f"\n{'═'*60}")
    print("  EVALUATION SUMMARY")
    print(f"{'═'*60}")

    if valid:
        avg_voice  = sum(r.voice_score for r in valid) / len(valid)
        avg_ret    = sum(r.retrieval_precision for r in valid) / len(valid)
        avg_faith  = sum(r.faithfulness_score for r in valid) / len(valid)
        n_passed   = sum(1 for r in valid if r.passed)
        total_time = sum(r.duration_seconds for r in results)

        print(f"  Cases run        : {len(results)} ({len(errors)} errors)")
        print(f"  Passed           : {n_passed}/{len(valid)}")
        print(f"  Avg voice        : {avg_voice*10:.1f}/10")
        print(f"  Avg retrieval    : {avg_ret:.0%}")
        print(f"  Avg faithfulness : {avg_faith*10:.1f}/10")
        print(f"  Total wall time  : {total_time:.0f}s")

    print(f"{'═'*60}\n")
