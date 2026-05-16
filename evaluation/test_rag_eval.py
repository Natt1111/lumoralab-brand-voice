"""
Pytest evaluation suite for the LumoraLab RAG system.

Run:  pytest evaluation/test_rag_eval.py -v -s

Session flow
  1. ensure_ingested  — seeds the vector store if empty
  2. eval_results     — runs the full evaluation (3 cases in CI mode, all 10 for full run)
  3. Test classes     — assert on voice, retrieval, faithfulness, and overall pass rate
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from evaluation.report import save_markdown_report
from evaluation.runner import EvalResult, run_evaluation

# ── Optional deepeval integration ────────────────────────────────────────────
try:
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

# ── Sample content used to seed the vector store for evaluation ───────────────
_EVAL_SEED_TEXT = """
LumoraLab helps small businesses automate their workflows with AI tools that are intuitive and affordable. We believe that cutting-edge technology should not be reserved for enterprise companies alone.

A dental practice in Portland was losing $3,000 a month to no-shows. Not because patients didn't care — they forgot. The front desk was already swamped with phones, insurance checks, and sterilisation logs. Adding "call every patient 48 hours before their appointment" wasn't realistic. An AI assistant that responds to "Can I reschedule?" and "Do you take my insurance?" any time of day changed everything. Their no-show rate dropped 60% in three months.

Maria runs a small restaurant. Every Monday she spent three hours manually updating the week's reservations, calling no-shows, and texting regulars about specials. With LumoraLab, those three hours are gone. She uses them to create new dishes instead.

The yoga studio on Main Street had their manager spend every Sunday evening confirming Tuesday classes by phone. One by one. Thirty calls. With automation, members get a text, can confirm or cancel with one tap, and the class list updates itself. The manager now has her Sundays back.

David is a freelance designer. He was spending eight hours a week on invoicing, chasing late payments, and answering "where are we on project X?" emails. LumoraLab automates his follow-up sequences and client status updates. He calls it the best hire he never made.

A local bakery was drowning in Instagram DMs asking "are you open on Sundays?" and "do you do custom cakes?" — the same ten questions, all day, every day. Their AI assistant handles all of them instantly. The baker bakes. The AI handles the inbox.

We measure our success by one metric: how many hours per week we give back to our customers. The average is 8-12 hours. That is a full working day returned to the work you actually love doing.

Small businesses are the backbone of every community. The dentist who remembers your kids' names. The bookshop that stocks what your neighbourhood reads. The independent gym where the trainer knows your injury history. These businesses deserve tools that work as hard as they do. LumoraLab exists for them.

Our AI doesn't just execute tasks. It learns your preferences, adapts to your workflow, and gets smarter the more you use it. Think of it as hiring a very capable operations assistant who never sleeps, never forgets, and never charges overtime. Setup takes minutes. No IT department required. No six-week onboarding. No contract that locks you in for three years. Just automation that works from day one.
"""

# ── Session fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def ensure_ingested():
    """Seed the vector store with LumoraLab content before evaluation begins."""
    from app.ingestion import ingest_document
    from app.retrieval import _get_collection

    collection = _get_collection()
    count = collection.count()
    if count == 0:
        print("\n[eval] Vector store is empty — ingesting seed content...")
        result = ingest_document(
            text=_EVAL_SEED_TEXT,
            source="eval_seed.txt",
            metadata={"type": "eval_fixture"},
        )
        print(f"[eval] Ingested {result.chunks_created} chunks from '{result.source}'")
    else:
        print(f"\n[eval] Vector store has {count} chunks — skipping seed ingest")


@pytest.fixture(scope="session")
def eval_results(ensure_ingested) -> list[EvalResult]:
    """
    Run the evaluation suite once per pytest session.
    Use sample_size=3 for a quick smoke-test; remove it for the full 10-case run.
    """
    print("\n[eval] Starting evaluation run (3 of 10 cases)...")
    results = run_evaluation(sample_size=3)

    # Write reports
    md_path = save_markdown_report(results)
    print(f"[eval] Markdown report saved → {md_path}")

    return results


# ── Voice consistency tests ───────────────────────────────────────────────────

class TestVoiceConsistency:
    def test_average_voice_score_above_threshold(self, eval_results):
        """Average voice score must be ≥ 5/10 (Claude judge)."""
        valid = [r for r in eval_results if not r.error]
        assert valid, "No valid results to evaluate"
        avg = sum(r.voice_score for r in valid) / len(valid)
        print(f"\n  Avg voice score : {avg*10:.1f}/10")
        assert avg >= 0.5, f"Avg voice score {avg*10:.1f}/10 is below threshold 5.0/10"

    def test_no_critically_low_voice_scores(self, eval_results):
        """No single result should score below 3/10 on voice consistency."""
        failures = [r for r in eval_results if not r.error and r.voice_score < 0.3]
        for f in failures:
            print(f"\n  CRITICAL: {f.test_id} voice={f.voice.raw_score}/10 — {f.voice.reason}")
        assert not failures, f"{len(failures)} result(s) with critically low voice scores (<3/10)"


# ── Retrieval quality tests ───────────────────────────────────────────────────

class TestRetrievalQuality:
    def test_chunks_retrieved_for_all_cases(self, eval_results):
        """Every successful test case must retrieve at least one chunk."""
        missing = [r for r in eval_results if not r.error and r.num_chunks == 0]
        assert not missing, f"{len(missing)} case(s) retrieved zero chunks"

    def test_average_retrieval_precision(self, eval_results):
        """Average retrieval precision must be ≥ 30 % of expected themes."""
        valid = [r for r in eval_results if not r.error]
        avg = sum(r.retrieval_precision for r in valid) / len(valid)
        print(f"\n  Avg retrieval precision : {avg:.0%}")
        assert avg >= 0.3, f"Avg retrieval precision {avg:.0%} is below 30 %"

    def test_sources_attributed(self, eval_results):
        """Every successful result must list at least one source."""
        missing = [r for r in eval_results if not r.error and not r.retrieved_sources]
        assert not missing, f"{len(missing)} case(s) have no attributed sources"


# ── Faithfulness tests ────────────────────────────────────────────────────────

class TestFaithfulness:
    def test_average_faithfulness_above_threshold(self, eval_results):
        """Average faithfulness must be ≥ 5/10 (Claude judge)."""
        valid = [r for r in eval_results if not r.error]
        avg = sum(r.faithfulness_score for r in valid) / len(valid)
        print(f"\n  Avg faithfulness : {avg*10:.1f}/10")
        assert avg >= 0.5, f"Avg faithfulness {avg*10:.1f}/10 is below threshold 5.0/10"


# ── Overall / integration tests ───────────────────────────────────────────────

class TestOverall:
    def test_no_pipeline_errors(self, eval_results):
        """The generation pipeline must not throw for any test case."""
        errors = [r for r in eval_results if r.error]
        for e in errors:
            print(f"\n  PIPELINE ERROR [{e.test_id}]: {e.error}")
        assert not errors, f"{len(errors)} case(s) raised pipeline errors"

    def test_pass_rate_above_50_pct(self, eval_results):
        """At least 50 % of test cases must pass all three metric thresholds."""
        valid = [r for r in eval_results if not r.error]
        passed = [r for r in valid if r.passed]
        rate = len(passed) / len(valid) if valid else 0
        print(f"\n  Pass rate : {len(passed)}/{len(valid)} ({rate:.0%})")
        assert rate >= 0.5, f"Pass rate {rate:.0%} is below 50 %"

    def test_generated_content_is_non_empty(self, eval_results):
        """Every generated response must contain actual text."""
        empty = [r for r in eval_results if not r.error and not r.generated_content.strip()]
        assert not empty, f"{len(empty)} case(s) produced empty generated content"

    def test_print_final_summary(self, eval_results):
        """Print a human-readable summary table (always passes)."""
        valid = [r for r in eval_results if not r.error]
        print(f"\n{'═'*62}")
        print(f"  {'ID':<10} {'Content type':<16} {'Voice':>6} {'Ret':>6} {'Faith':>6} {'Pass':>5}")
        print(f"  {'─'*58}")
        for r in eval_results:
            if r.error:
                print(f"  {r.test_id:<10} {'ERROR':<16}")
            else:
                ok = "✓" if r.passed else "✗"
                print(
                    f"  {r.test_id:<10} {r.content_type:<16} "
                    f"{r.voice.raw_score:>5.0f}/10 "
                    f"{r.retrieval_precision:>5.0%} "
                    f"{r.faithfulness.raw_score:>5.0f}/10 "
                    f"  {ok}"
                )
        print(f"{'═'*62}")
        assert True


# ── Optional deepeval smoke-test ──────────────────────────────────────────────

@pytest.mark.skipif(not DEEPEVAL_AVAILABLE, reason="deepeval not installed")
def test_deepeval_test_case_wrapping(eval_results):
    """Verify that results can be wrapped as deepeval LLMTestCase objects."""
    for r in eval_results:
        if r.error:
            continue
        tc = LLMTestCase(
            input=r.query,
            actual_output=r.generated_content,
            retrieval_context=r.retrieved_sources,
        )
        assert tc.input == r.query
        assert tc.actual_output == r.generated_content
    print(f"\n  deepeval LLMTestCase wrapping: OK ({len(eval_results)} cases)")
