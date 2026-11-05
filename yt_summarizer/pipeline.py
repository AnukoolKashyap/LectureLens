"""Pipeline runner — executes stages 1-6 with layers 7-9 applied across them.

This is a fixed, linear sequence of function calls. Each stage's output is the
next stage's input; nothing here chooses its own steps, calls tools on its own
initiative, or loops back on itself. Stage 9 writes reviewer feedback to a file
that a person reads offline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .config import settings
from .layer7_errors import PipelineError, UnsupportedContentError, get_logger
from .layer8_redteam import SafetyReport, merge, neutralise_injection, screen_input, screen_output
from .layer9_eval import EvaluationReport, evaluate
from .stage1_ingest import VideoDocument, course_title, ingest, ingest_many
from .stage2_preprocess import Chunk, chunk_document, chunk_documents
from .stage3_embed import IndexResult, build_index
from .stage4_retrieve import (RetrievedChunk, build_context, retrieve,
                              retrieve_across_course, retrieve_for_summary)
from .stage5_summarize import (SummaryResult, answer_question, summarise,
                               summarise_course)
from .stage6_output import (CourseReport, VideoSummary, assemble, export,
                            export_course)

log = get_logger(__name__)

ProgressFn = Callable[[int, str], None]


@dataclass
class PipelineRun:
    document: VideoDocument
    chunks: List[Chunk]
    index: IndexResult
    retrieved: List[RetrievedChunk]
    summary: VideoSummary
    evaluation: EvaluationReport
    safety: SafetyReport
    stage_times: Dict[str, float] = field(default_factory=dict)
    exports: Dict[str, Path] = field(default_factory=dict)

    @property
    def context(self) -> str:
        return build_context(self.retrieved)

    def ask(self, question: str, profile_key: str = "lecture") -> str:
        hits = retrieve(question, self.index)
        return answer_question(question, hits, profile_key)


def _noop(step: int, label: str) -> None:
    log.info("Stage %d — %s", step, label)


def run_pipeline(
    url: str = "",
    local_path: Optional[Path] = None,
    profile_key: str = "lecture",
    reference_summary: Optional[str] = None,
    do_export: bool = True,
    progress: Optional[ProgressFn] = None,
    block_on_unsafe: bool = True,
) -> PipelineRun:
    """Run stages 1-6 end to end and return everything the UI needs."""
    report = progress or _noop
    times: Dict[str, float] = {}

    def timed(step: int, label: str, fn):
        report(step, label)
        started = time.time()
        value = fn()
        times[label] = round(time.time() - started, 2)
        return value

    # --- Stage 1 ----------------------------------------------------------
    document = timed(1, "ingestion", lambda: ingest(url, local_path))

    # --- Layer 8 on the input --------------------------------------------
    input_safety = screen_input(document.full_text)
    if input_safety.content and block_on_unsafe:
        raise UnsupportedContentError(
            "The content safety screen flagged this transcript (%s). It was not summarised."
            % ", ".join(f.label for f in input_safety.content)
        )

    # --- Stage 2 ----------------------------------------------------------
    chunks = timed(2, "preprocessing", lambda: chunk_document(document))
    if not chunks:
        raise PipelineError("The transcript produced no usable chunks.")

    if input_safety.injection:
        log.warning("Neutralising %d injection attempt(s) before prompting",
                    len(input_safety.injection))
        for chunk in chunks:
            chunk.text = neutralise_injection(chunk.text)

    # --- Stage 3 ----------------------------------------------------------
    index = timed(3, "indexing", lambda: build_index(chunks))

    # --- Stage 4 ----------------------------------------------------------
    retrieved = timed(4, "retrieval", lambda: retrieve_for_summary(index, chunks))

    # --- Stage 5 ----------------------------------------------------------
    result: SummaryResult = timed(5, "summarisation", lambda: summarise(retrieved, profile_key))

    # --- Layer 8 on the output -------------------------------------------
    context = build_context(retrieved)
    output_safety = screen_output(result.summary, result.key_points, context)
    safety = merge(input_safety, output_safety)

    # --- Stage 6 ----------------------------------------------------------
    summary = timed(6, "output", lambda: assemble(document, result))

    # --- Layer 9 ----------------------------------------------------------
    evaluation = evaluate(
        summary_text=result.summary,
        key_points=result.key_points,
        context=context,
        transcript=document.full_text,
        chunk_texts=[c.text for c in retrieved],
        embedder=index.embedder,
        reference=reference_summary,
        generated_by_llm=result.generated,
        degraded_embedder=index.degraded,
    )

    summary.metrics = evaluation.to_dict()
    summary.safety = safety.to_dict()
    summary.warnings.extend(evaluation.notes)
    summary.diagnostics = {
        "chunk_count": len(chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "retrieved_count": len(retrieved),
        "top_k": settings.top_k,
        "rerank": settings.rerank,
        "embedding_model": index.embedder.name,
        "embedder_degraded": index.degraded,
        "vector_store": index.store.backend_name,
        "llm_latency_seconds": round(result.latency_seconds, 2),
        "stage_times": times,
        "use_case_profile": profile_key,
    }

    exports: Dict[str, Path] = {}
    if do_export:
        exports = export(summary)

    return PipelineRun(
        document=document,
        chunks=chunks,
        index=index,
        retrieved=retrieved,
        summary=summary,
        evaluation=evaluation,
        safety=safety,
        stage_times=times,
        exports=exports,
    )


# ---------------------------------------------------------------------------
# Courses: several videos, one index
# ---------------------------------------------------------------------------

@dataclass
class CourseRun:
    documents: List[VideoDocument]
    chunks: List[Chunk]
    index: IndexResult
    report: CourseReport
    exports: Dict[str, Path] = field(default_factory=dict)
    # The passages each video's summary was written from, in course order.
    retrieved: List[RetrievedChunk] = field(default_factory=list)

    def ask(self, question: str, profile_key: str = "lecture") -> str:
        """Answer using the whole course, capped so one video cannot dominate."""
        hits = retrieve_across_course(question, self.index)
        return answer_question(question, hits, profile_key)

    def where_is(self, topic: str, limit: int = 5) -> List[Dict]:
        """Locate a topic across the course: which video, which moment.

        This is the thing a course gives you that a single video cannot — the
        same idea often appears in more than one lecture.
        """
        hits = retrieve_across_course(topic, self.index, top_k=limit, per_video_cap=2)
        return [
            {
                "video_index": h.video_index,
                "video_title": h.video_title,
                "timestamp": h.timestamp,
                "url": h.watch_url(),
                "score": round(h.final_score, 3),
                "preview": h.text[:200],
            }
            for h in hits
        ]


def run_course(
    sources,
    profile_key: str = "lecture",
    do_export: bool = True,
    progress: Optional[ProgressFn] = None,
    local_paths=None,
) -> CourseRun:
    """Run the pipeline over a playlist or a set of videos.

    Every video is chunked into one shared index, so retrieval and the chat can
    cross video boundaries. Each video is then summarised on its own, and those
    summaries are synthesised into one course guide — a map-reduce, because a
    whole course never fits in a single prompt.
    """
    report_fn = progress or _noop
    started = time.time()
    times: Dict[str, float] = {}

    # --- Stage 1 ----------------------------------------------------------
    report_fn(1, "ingestion")
    t0 = time.time()
    if local_paths:
        from .stage1_ingest import load_local_course
        documents, skipped = load_local_course(local_paths), []
    else:
        documents, skipped = ingest_many(sources)
    times["ingestion"] = round(time.time() - t0, 2)

    # --- Layer 8 on every transcript -------------------------------------
    input_safety = SafetyReport()
    for doc in documents:
        input_safety = merge(input_safety, screen_input(doc.full_text))

    # --- Stage 2 ----------------------------------------------------------
    report_fn(2, "preprocessing")
    t0 = time.time()
    chunks = chunk_documents(documents)
    if not chunks:
        raise PipelineError("The course produced no usable chunks.")
    if input_safety.injection:
        log.warning("Neutralising %d injection attempt(s) across the course",
                    len(input_safety.injection))
        for chunk in chunks:
            chunk.text = neutralise_injection(chunk.text)
    times["preprocessing"] = round(time.time() - t0, 2)

    # --- Stage 3: one index for the whole course --------------------------
    report_fn(3, "indexing")
    t0 = time.time()
    index = build_index(chunks)
    times["indexing"] = round(time.time() - t0, 2)

    # --- Stages 4-6, per video -------------------------------------------
    video_summaries: List[VideoSummary] = []
    compact: List[Dict] = []
    safety = input_safety
    all_notes: List[str] = []
    all_retrieved: List[RetrievedChunk] = []
    t_retrieve = t_sum = 0.0

    for doc in documents:
        report_fn(4, "video %d of %d" % (doc.course_index, len(documents)))
        t0 = time.time()
        retrieved = retrieve_for_summary(index, chunks, video_id=doc.video_id)
        all_retrieved.extend(retrieved)
        t_retrieve += time.time() - t0

        t0 = time.time()
        result: SummaryResult = summarise(retrieved, profile_key)
        t_sum += time.time() - t0

        context = build_context(retrieved)
        safety = merge(safety, screen_output(result.summary, result.key_points, context))

        summary = assemble(doc, result)
        evaluation = evaluate(
            summary_text=result.summary,
            key_points=result.key_points,
            context=context,
            transcript=doc.full_text,
            chunk_texts=[c.text for c in retrieved],
            embedder=index.embedder,
            reference=None,
            generated_by_llm=result.generated,
            degraded_embedder=index.degraded,
        )
        summary.metrics = evaluation.to_dict()
        summary.diagnostics = {
            "course_index": doc.course_index,
            "chunk_count": sum(1 for c in chunks if c.video_id == doc.video_id),
            "retrieved_count": len(retrieved),
            "llm_latency_seconds": round(result.latency_seconds, 2),
        }
        all_notes.extend(evaluation.notes)
        video_summaries.append(summary)
        compact.append({
            "index": doc.course_index,
            "title": doc.title,
            "summary": result.summary,
            "key_points": result.key_points,
        })

    times["retrieval"] = round(t_retrieve, 2)
    times["summarisation"] = round(t_sum, 2)

    # --- Reduce: one guide for the course ---------------------------------
    report_fn(5, "course synthesis")
    t0 = time.time()
    course = summarise_course(compact)
    times["course_synthesis"] = round(time.time() - t0, 2)

    # --- Stage 6 ----------------------------------------------------------
    report_fn(6, "output")
    trustworthy = [v for v in video_summaries if v.metrics.get("trustworthy")]
    def _mean(key):
        vals = [v.metrics[key] for v in trustworthy if v.metrics.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    metrics = {
        "videos_scored": len(trustworthy),
        "faithfulness": _mean("faithfulness"),
        "relevance": _mean("relevance"),
        "compression_ratio": _mean("compression_ratio"),
        "transcript_words": sum(len(d.full_text.split()) for d in documents),
    }
    metrics = {k: v for k, v in metrics.items() if v is not None}

    report = CourseReport(
        title=course_title(documents, sources),
        source=", ".join(d.url for d in documents[:3]) + (" ..." if len(documents) > 3 else ""),
        videos=video_summaries,
        overview=course.overview,
        themes=course.themes,
        progression=course.progression,
        study_order=course.study_order,
        key_takeaways=course.key_takeaways,
        model=course.model,
        generated=course.generated,
        skipped=[s.to_dict() for s in skipped],
        metrics=metrics,
        warnings=sorted(set(course.warnings + all_notes)),
        diagnostics={
            "video_count": len(documents),
            "skipped_count": len(skipped),
            "chunk_count": len(chunks),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "top_k": settings.top_k,
            "embedding_model": index.embedder.name,
            "embedder_degraded": index.degraded,
            "vector_store": index.store.backend_name,
            "stage_times": times,
            "total_seconds": round(time.time() - started, 2),
            "safety": safety.to_dict()["counts"],
        },
    )

    exports: Dict[str, Path] = export_course(report) if do_export else {}
    return CourseRun(documents=documents, chunks=chunks, index=index,
                     report=report, exports=exports, retrieved=all_retrieved)
