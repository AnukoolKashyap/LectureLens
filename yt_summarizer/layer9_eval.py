"""Layer 9 — evaluation and human feedback.

Automated metrics:
  * ROUGE-1 / ROUGE-2 / ROUGE-L — overlap between the generated summary and a
    reference. Uses the `rouge-score` package when installed; otherwise a
    self-contained implementation of the same measures runs instead.
  * faithfulness — share of generated content words supported by the retrieved
    context. A lexical proxy for entailment, labelled as such everywhere.
  * relevance — cosine similarity between the summary embedding and the mean
    chunk embedding, when a real embedder is in use.
  * compression — transcript words divided by summary words.

Human feedback is recorded to a JSONL file. Nothing in this module writes back
into the pipeline at runtime: reviewer ratings are read by a person deciding
whether to change a setting, which is why the Review 2 diagram labels this
"offline and reviewer-approved, no self-updating loop".
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import DATA_DIR, settings
from .layer7_errors import get_logger
from .stage2_preprocess import content_terms

log = get_logger(__name__)

FEEDBACK_PATH = DATA_DIR / "human_feedback.jsonl"


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _content_tokens(text: str) -> List[str]:
    return content_terms(text)


# --------------------------------------------------------------------------
# ROUGE
# --------------------------------------------------------------------------

def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def _f1(overlap: int, candidate_total: int, reference_total: int) -> float:
    if overlap == 0 or candidate_total == 0 or reference_total == 0:
        return 0.0
    precision = overlap / candidate_total
    recall = overlap / reference_total
    return 2 * precision * recall / (precision + recall)


def _rouge_n(candidate: List[str], reference: List[str], n: int) -> float:
    def ngrams(seq: List[str]) -> Counter:
        return Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))

    cand_grams, ref_grams = ngrams(candidate), ngrams(reference)
    overlap = sum((cand_grams & ref_grams).values())
    return _f1(overlap, sum(cand_grams.values()), sum(ref_grams.values()))


def rouge_scores(candidate: str, reference: str) -> Dict[str, float]:
    """ROUGE-1/2/L F-measure. Prefers the `rouge-score` package when present."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        return {
            "rouge_1": float(scores["rouge1"].fmeasure),
            "rouge_2": float(scores["rouge2"].fmeasure),
            "rouge_l": float(scores["rougeL"].fmeasure),
            "rouge_backend": "rouge-score",
        }
    except ImportError:
        pass

    cand, ref = _tokens(candidate), _tokens(reference)
    return {
        "rouge_1": _rouge_n(cand, ref, 1),
        "rouge_2": _rouge_n(cand, ref, 2),
        "rouge_l": _f1(_lcs_length(cand, ref), len(cand), len(ref)),
        "rouge_backend": "built-in",
    }


# --------------------------------------------------------------------------
# Faithfulness, relevance, compression
# --------------------------------------------------------------------------

def faithfulness(generated: str, context: str) -> float:
    """Share of generated content words that appear in the retrieved context.

    Lexical proxy for entailment. High scores mean the wording is drawn from the
    source; they do not prove the claims are logically entailed.
    """
    gen_terms = _content_tokens(generated)
    if not gen_terms:
        return 0.0
    context_terms = set(_content_tokens(context))
    return sum(1 for t in gen_terms if t in context_terms) / len(gen_terms)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def relevance(generated: str, chunk_texts: Sequence[str], embedder) -> Optional[float]:
    if embedder is None or not chunk_texts:
        return None
    try:
        vectors = embedder.encode([generated] + list(chunk_texts))
    except Exception as exc:
        log.warning("Relevance scoring skipped: %s", exc)
        return None
    summary_vec, chunk_vecs = vectors[0], vectors[1:]
    dim = len(summary_vec)
    centroid = [sum(v[i] for v in chunk_vecs) / len(chunk_vecs) for i in range(dim)]
    return _cosine(summary_vec, centroid)


def compression_ratio(transcript: str, summary: str) -> float:
    summary_words = len(summary.split())
    if summary_words == 0:
        return 0.0
    return len(transcript.split()) / summary_words


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    rouge_1: Optional[float] = None
    rouge_2: Optional[float] = None
    rouge_l: Optional[float] = None
    rouge_backend: Optional[str] = None
    rouge_reference: Optional[str] = None
    faithfulness: float = 0.0
    relevance: Optional[float] = None
    compression_ratio: float = 0.0
    summary_words: int = 0
    transcript_words: int = 0
    trustworthy: bool = True
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        data = {k: v for k, v in asdict(self).items() if v is not None}
        return data

    def display_rows(self) -> List[tuple]:
        rows = []
        if self.rouge_1 is not None:
            rows.append(("ROUGE-1", "%.3f" % self.rouge_1))
            rows.append(("ROUGE-2", "%.3f" % self.rouge_2))
            rows.append(("ROUGE-L", "%.3f" % self.rouge_l))
        rows.append(("Faithfulness (lexical proxy)", "%.3f" % self.faithfulness))
        if self.relevance is not None:
            rows.append(("Relevance", "%.3f" % self.relevance))
        rows.append(("Compression", "%.1fx" % self.compression_ratio))
        return rows


def evaluate(
    summary_text: str,
    key_points: Sequence[str],
    context: str,
    transcript: str,
    chunk_texts: Sequence[str],
    embedder=None,
    reference: Optional[str] = None,
    generated_by_llm: bool = True,
    degraded_embedder: bool = False,
) -> EvaluationReport:
    combined = " ".join([summary_text] + list(key_points)).strip()

    report = EvaluationReport(
        faithfulness=faithfulness(combined, context),
        compression_ratio=compression_ratio(transcript, combined),
        summary_words=len(combined.split()),
        transcript_words=len(transcript.split()),
    )

    if reference:
        scores = rouge_scores(combined, reference)
        report.rouge_1 = scores["rouge_1"]
        report.rouge_2 = scores["rouge_2"]
        report.rouge_l = scores["rouge_l"]
        report.rouge_backend = scores["rouge_backend"]
        report.rouge_reference = "supplied"
    else:
        report.notes.append(
            "ROUGE not computed: it needs a human-written reference summary. "
            "Pass one with --reference to score it."
        )

    if embedder is not None and not degraded_embedder:
        report.relevance = relevance(combined, chunk_texts, embedder)

    if not generated_by_llm:
        report.trustworthy = False
        report.notes.append(
            "Measured on offline extractive fallback output. These figures describe "
            "the fallback, not the generative pipeline, and must not be reported as "
            "project results."
        )
    if degraded_embedder:
        report.trustworthy = False
        report.notes.append(
            "The hashing fallback embedder was in use, so retrieval was lexical rather "
            "than semantic. Figures are not representative of the real system."
        )

    log.info(
        "Evaluation: faithfulness=%.3f compression=%.1fx trustworthy=%s",
        report.faithfulness, report.compression_ratio, report.trustworthy,
    )
    return report


# --------------------------------------------------------------------------
# Human feedback (offline, reviewer-approved)
# --------------------------------------------------------------------------

def record_feedback(
    video_id: str,
    rating: int,
    comment: str = "",
    reviewer: str = "anonymous",
    path: Optional[Path] = None,
) -> Path:
    """Append one reviewer rating to the feedback log.

    The log is read by a person when deciding whether to change a setting. The
    pipeline never reads it at runtime.
    """
    path = Path(path or FEEDBACK_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_id": video_id,
        "reviewer": reviewer,
        "rating": int(rating),
        "comment": comment.strip(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Recorded reviewer feedback for %s (rating %d)", video_id, rating)
    return path


def feedback_summary(path: Optional[Path] = None) -> Dict:
    path = Path(path or FEEDBACK_PATH)
    if not path.exists():
        return {"count": 0, "mean_rating": None}
    ratings = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ratings.append(int(json.loads(line)["rating"]))
        except Exception:
            continue
    return {
        "count": len(ratings),
        "mean_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
    }
