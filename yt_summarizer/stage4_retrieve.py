"""Stage 4 — retrieval engine.

Encodes the query, pulls `fetch_k` candidates from the vector store, re-ranks
them, and returns the `top_k` chunks that become the grounding context.

Re-ranking blends three signals:
  * the vector similarity score from stage 3
  * lexical overlap between the query terms and the chunk's stop-word-filtered
    search key, which rescues chunks where an exact term matters
  * a small positional prior keeping the timeline readable when scores tie

A course puts many videos in one index. Retrieval can therefore be scoped to a
single video (`video_id=`) or run across the whole course, and every retrieved
chunk carries the video it came from so the summariser can cite it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import settings
from .layer7_errors import get_logger
from .stage1_ingest import format_timestamp
from .stage2_preprocess import Chunk, build_search_key
from .stage3_embed import EmbeddingBackend, Hit, IndexResult

log = get_logger(__name__)

W_VECTOR = 0.75
W_LEXICAL = 0.20
W_POSITION = 0.05


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    timestamp: str
    start: float
    end: float
    vector_score: float
    lexical_score: float
    final_score: float
    video_id: str = ""
    video_title: str = ""
    video_index: int = 0

    @property
    def label(self) -> str:
        """How this chunk is cited. Includes the video only in a course."""
        if self.video_index:
            return "V%d %s" % (self.video_index, self.timestamp)
        return self.timestamp

    def as_context_block(self, n: int) -> str:
        return "[%d] (%s) %s" % (n, self.label, self.text)

    def watch_url(self) -> str:
        if not self.video_id:
            return ""
        return "https://www.youtube.com/watch?v=%s&t=%ds" % (self.video_id, int(self.start))

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "timestamp": self.timestamp,
            "start": self.start,
            "end": self.end,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "video_index": self.video_index,
            "vector_score": round(self.vector_score, 4),
            "lexical_score": round(self.lexical_score, 4),
            "final_score": round(self.final_score, 4),
            "preview": self.text[:160] + ("..." if len(self.text) > 160 else ""),
        }


def _lexical_overlap(query_key: str, chunk_text: str) -> float:
    query_terms = set(query_key.split())
    if not query_terms:
        return 0.0
    chunk_terms = set(build_search_key(chunk_text).split())
    if not chunk_terms:
        return 0.0
    return len(query_terms & chunk_terms) / len(query_terms)


def retrieve(
    query: str,
    index: IndexResult,
    top_k: Optional[int] = None,
    fetch_k: Optional[int] = None,
    video_id: Optional[str] = None,
) -> List[RetrievedChunk]:
    """Retrieve and re-rank. `video_id` limits the search to one video."""
    top_k = top_k or settings.top_k
    fetch_k = max(fetch_k or settings.fetch_k, top_k)

    query_vector = index.embedder.encode([query])[0]
    where = {"video_id": video_id} if video_id else None
    candidates: List[Hit] = index.store.query(query_vector, fetch_k, where=where)
    if not candidates:
        return []

    query_key = build_search_key(query)
    total = max(len(candidates), 1)

    scored: List[RetrievedChunk] = []
    for rank, hit in enumerate(candidates):
        lexical = _lexical_overlap(query_key, hit.text)
        position_prior = 1.0 - (rank / total)
        final = (
            W_VECTOR * hit.score
            + W_LEXICAL * lexical
            + W_POSITION * position_prior
        ) if settings.rerank else hit.score

        meta = hit.metadata or {}
        scored.append(
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                text=hit.text,
                timestamp=meta.get("timestamp") or format_timestamp(meta.get("start", 0.0)),
                start=float(meta.get("start", 0.0)),
                end=float(meta.get("end", 0.0)),
                vector_score=float(hit.score),
                lexical_score=float(lexical),
                final_score=float(final),
                video_id=str(meta.get("video_id", "")),
                video_title=str(meta.get("video_title", "")),
                video_index=int(meta.get("video_index", 0) or 0),
            )
        )

    scored.sort(key=lambda c: c.final_score, reverse=True)
    selected = scored[:top_k]
    log.info(
        "Retrieved %d of %d candidates (top_k=%d, rerank=%s%s)",
        len(selected), len(candidates), top_k, settings.rerank,
        ", video=%s" % video_id if video_id else "",
    )
    return selected


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as the numbered context block the prompts expect."""
    return "\n\n".join(c.as_context_block(i + 1) for i, c in enumerate(chunks))


COVERAGE_PROBES = [
    "main topic and overall purpose of this video",
    "key concepts definitions and terminology explained",
    "worked examples demonstrations and steps shown",
    "conclusions results takeaways and summary",
]


def retrieve_for_summary(
    index: IndexResult,
    all_chunks: List[Chunk],
    video_id: Optional[str] = None,
) -> List[RetrievedChunk]:
    """Coverage-oriented retrieval for summarising one video.

    A single query would bias the summary toward one part of the video, so the
    whole timeline is swept with a set of broad queries and the union of hits is
    returned in chronological order.

    In a course, `video_id` keeps the sweep inside one video — otherwise a long
    video would crowd out the short ones.
    """
    scope = [c for c in all_chunks if not video_id or c.video_id == video_id]
    seen: Dict[str, RetrievedChunk] = {}
    per_probe = max(2, settings.top_k // 2)

    for probe in COVERAGE_PROBES:
        for chunk in retrieve(probe, index, top_k=per_probe, video_id=video_id):
            existing = seen.get(chunk.chunk_id)
            if existing is None or chunk.final_score > existing.final_score:
                seen[chunk.chunk_id] = chunk

    # Guarantee the opening and closing of the video are represented.
    ordered = sorted(scope, key=lambda c: c.start)
    for edge in (ordered[:1] + ordered[-1:]):
        if edge.chunk_id not in seen:
            seen[edge.chunk_id] = RetrievedChunk(
                chunk_id=edge.chunk_id,
                text=edge.text,
                timestamp=edge.timestamp,
                start=edge.start,
                end=edge.end,
                vector_score=0.0,
                lexical_score=0.0,
                final_score=0.0,
                video_id=edge.video_id,
                video_title=edge.video_title,
                video_index=edge.video_index,
            )

    result = sorted(seen.values(), key=lambda c: c.start)
    log.info("Coverage retrieval selected %d chunk(s)%s",
             len(result), " for video %s" % video_id if video_id else "")
    return result


def retrieve_across_course(
    query: str,
    index: IndexResult,
    top_k: Optional[int] = None,
    per_video_cap: int = 2,
) -> List[RetrievedChunk]:
    """Answer a question using the whole course, without one video dominating.

    Plain top-k across a course tends to return five chunks from whichever video
    happens to match best, which hides the fact that the same idea is covered
    again elsewhere. This pulls a wider candidate set, then caps how many chunks
    any single video may contribute before taking the best overall.
    """
    top_k = top_k or settings.top_k
    wide = retrieve(query, index, top_k=top_k * 4, fetch_k=max(settings.fetch_k, top_k * 6))

    per_video: Dict[str, int] = {}
    kept: List[RetrievedChunk] = []
    for chunk in wide:
        used = per_video.get(chunk.video_id, 0)
        if used >= per_video_cap:
            continue
        per_video[chunk.video_id] = used + 1
        kept.append(chunk)
        if len(kept) >= top_k:
            break

    log.info("Course retrieval: %d chunk(s) across %d video(s)", len(kept), len(per_video))
    return kept
