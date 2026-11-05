"""Stage 2 — text preprocessing and chunking.

Cleans the raw transcript, then splits it into overlapping character chunks of
`chunk_size` with `chunk_overlap` shared characters. Chunk boundaries are kept
aligned to transcript segments so every chunk carries a true start and end time.

Stop-word removal is applied only to the *retrieval key* of a chunk, never to
the text handed to the summariser. Removing stop words from the generated
context would damage fluency; removing them from the search key sharpens
lexical matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, List

from .config import settings
from .layer7_errors import get_logger
from .stage1_ingest import Segment, VideoDocument, format_timestamp

log = get_logger(__name__)

STOP_WORDS = {
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between",
    "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it",
    "its", "just", "me", "more", "most", "my", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you",
    "your", "yours",
}

# Filler tokens typical of auto-generated captions.
_FILLERS = re.compile(
    r"\b(uh|um+|erm|hmm+|you know|i mean|sort of|kind of|like,)\b", re.IGNORECASE
)
_BRACKETED = re.compile(r"\[(?:music|applause|laughter|inaudible|silence)[^\]]*\]", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    search_key: str
    start: float
    end: float
    index: int
    # Which video this chunk came from. A course puts many videos in one index,
    # so without these a retrieved chunk cannot be cited or filtered.
    video_id: str = ""
    video_title: str = ""
    video_index: int = 0

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    def metadata(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "start": self.start,
            "end": self.end,
            "index": self.index,
            "timestamp": self.timestamp,
            "video_id": self.video_id,
            "video_title": self.video_title,
            "video_index": self.video_index,
        }

    def to_dict(self) -> Dict:
        return asdict(self)


def clean_text(text: str) -> str:
    text = _BRACKETED.sub(" ", text)
    text = _FILLERS.sub(" ", text)
    text = text.replace(" ", " ")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def stem(token: str) -> str:
    """Conservative suffix stripping.

    Not linguistically principled, but applied symmetrically to both sides of
    every comparison, which is what matters: without it "rate" fails to match
    "rates" and a correctly grounded sentence gets flagged as invented.
    """
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    # "-es" is only a plural marker after a sibilant; elsewhere the plural is a
    # bare "-s", and stripping two characters would turn "rates" into "rat".
    if len(token) > 4 and token.endswith(("ses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def content_terms(text: str) -> List[str]:
    """Stemmed, stop-word-filtered terms. The shared basis for every text comparison."""
    return [
        stem(t)
        for t in re.findall(r"[a-z0-9']+", text.lower())
        if t not in STOP_WORDS and len(t) > 2
    ]


def build_search_key(text: str) -> str:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return " ".join(stem(t) for t in tokens if t not in STOP_WORDS and len(t) > 1)


def _normalise_segments(segments: List[Segment]) -> List[Segment]:
    cleaned = []
    for seg in segments:
        text = clean_text(seg.text)
        if text:
            cleaned.append(Segment(text=text, start=seg.start, duration=seg.duration))
    return cleaned


def chunk_documents(docs: List[VideoDocument]) -> List[Chunk]:
    """Chunk every video of a course into one flat list.

    Each chunk keeps its video's id, title and position, so a chunk retrieved
    later can say which video and which moment it came from.
    """
    out: List[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc))
    log.info("Chunked %d video(s) into %d chunks in total", len(docs), len(out))
    return out


def chunk_document(doc: VideoDocument) -> List[Chunk]:
    """Split a document into overlapping chunks aligned to segment boundaries."""
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    segments = _normalise_segments(doc.segments)
    if not segments:
        return []

    chunks: List[Chunk] = []
    buffer: List[Segment] = []
    buffer_len = 0
    index = 0

    def flush(buf: List[Segment], idx: int) -> Chunk:
        text = " ".join(s.text for s in buf)
        return Chunk(
            chunk_id="%s-%04d" % (doc.video_id, idx),
            text=text,
            search_key=build_search_key(text),
            start=buf[0].start,
            end=buf[-1].end,
            index=idx,
            video_id=doc.video_id,
            video_title=doc.title,
            video_index=getattr(doc, "course_index", 0),
        )

    for seg in segments:
        buffer.append(seg)
        buffer_len += len(seg.text) + 1

        if buffer_len >= size:
            chunks.append(flush(buffer, index))
            index += 1

            # Carry back whole segments until the overlap budget is filled.
            carried: List[Segment] = []
            carried_len = 0
            for prev in reversed(buffer):
                if carried_len >= overlap:
                    break
                carried.insert(0, prev)
                carried_len += len(prev.text) + 1
            buffer = carried
            buffer_len = carried_len

    tail_text = " ".join(s.text for s in buffer)
    if buffer and len(tail_text) >= settings.min_chunk_chars:
        chunks.append(flush(buffer, index))

    log.info(
        "Chunked into %d chunks (size=%d, overlap=%d), mean %d chars",
        len(chunks), size, overlap,
        int(sum(len(c.text) for c in chunks) / max(len(chunks), 1)),
    )
    return chunks
