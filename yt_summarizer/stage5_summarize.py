"""Stage 5 — summarisation engine (Mistral).

Sends the grounded prompt to Mistral and parses the JSON response. The call is
made with plain `urllib` so the only hard dependency is the standard library.

When no API key is configured an extractive fallback runs instead. Its output is
always flagged `generated=False` and labelled in the UI, because an extractive
summary is explicitly NOT what the project claims to deliver — it exists so the
rest of the pipeline can be demonstrated offline, never to stand in for the
generative result.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .config import settings
from .layer7_errors import LLMError, get_logger
from .prompts import build_qa_prompt, build_summary_prompt
from .stage2_preprocess import STOP_WORDS
from .stage4_retrieve import RetrievedChunk, build_context

log = get_logger(__name__)

NOT_COVERED = "not covered in this video"


@dataclass
class SummaryResult:
    summary: str
    key_points: List[str]
    timestamps: List[Dict[str, str]]
    mind_map: Dict
    model: str
    generated: bool                     # True only when an LLM produced this
    raw_response: str = ""
    latency_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "summary": self.summary,
            "key_points": self.key_points,
            "timestamps": self.timestamps,
            "mind_map": self.mind_map,
            "model": self.model,
            "generated": self.generated,
            "latency_seconds": round(self.latency_seconds, 2),
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Mistral transport
# --------------------------------------------------------------------------

def call_mistral(prompt: str, temperature: Optional[float] = None, json_mode: bool = False) -> str:
    if not settings.mistral_api_key:
        raise LLMError("MISTRAL_API_KEY is not set. Add it to your .env file.")

    body = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": settings.temperature if temperature is None else temperature,
        "max_tokens": settings.max_tokens,
    }
    if json_mode:
        # Smaller models drift from the schema without this.
        body["response_format"] = {"type": "json_object"}
    payload = json.dumps(body).encode("utf-8")

    request = urllib.request.Request(
        settings.llm_endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer %s" % settings.mistral_api_key,
        },
        method="POST",
    )

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=settings.request_timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                wait = 2 ** attempt
                log.warning("Mistral returned %s; retrying in %ss", exc.code, wait)
                time.sleep(wait)
                last_error = exc
                continue
            raise LLMError("Mistral returned HTTP %s: %s" % (exc.code, detail)) from exc
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise LLMError("Could not reach Mistral: %s" % exc) from exc

    raise LLMError("Could not reach Mistral: %s" % last_error)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(raw: str) -> Dict:
    """Extract the JSON object from the model response.

    The schema instruction forbids code fences, but models occasionally add
    them anyway, so they are stripped before parsing rather than treated as a
    failure.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"```\s*$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise LLMError("The model did not return parseable JSON.")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError("The model returned malformed JSON: %s" % exc) from exc


def _coerce(payload: Dict) -> Dict:
    """Normalise a parsed payload into the shape the rest of the app expects."""
    key_points = payload.get("key_points") or []
    if isinstance(key_points, str):
        key_points = [key_points]

    timestamps = []
    for item in payload.get("timestamps") or []:
        if isinstance(item, dict) and item.get("time"):
            timestamps.append({"time": str(item["time"]), "topic": str(item.get("topic", "")).strip()})

    mind_map = payload.get("mind_map") or {}
    if not isinstance(mind_map, dict):
        mind_map = {}

    return {
        "summary": str(payload.get("summary", "")).strip(),
        "key_points": [str(p).strip() for p in key_points if str(p).strip()],
        "timestamps": timestamps,
        "mind_map": mind_map,
    }


# --------------------------------------------------------------------------
# Timestamp validation
# --------------------------------------------------------------------------

def parse_clock(label: str) -> Optional[float]:
    """Parse 'mm:ss' or 'h:mm:ss' into seconds. Returns None if unparseable."""
    text = str(label).strip()
    # A course answer may arrive as "V2 12:08"; keep only the clock part.
    if " " in text:
        text = text.split()[-1]
    parts = text.split(":")
    if not 2 <= len(parts) <= 3:
        return None
    try:
        values = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0.0
    for value in values:
        seconds = seconds * 60 + value
    return seconds


def validate_timestamps(timestamps, chunks, tolerance: float = 15.0) -> tuple:
    """Keep only timestamps that land inside a retrieved chunk; snap near-misses.

    An exact string match against the chunk labels is too brittle. The model is
    told to copy timestamps from the context, but it rounds, paraphrases, and
    sometimes reports the moment a point is *made* rather than the label at the
    head of the block it came from. Exact matching discards all of those and the
    demo then shows an empty timestamp table beside a perfectly good summary.

    A timestamp is accepted when it falls inside a retrieved chunk's span (plus
    a tolerance for rounding) and is relabelled to that chunk's own start time,
    which is a real, clickable position. One that matches no chunk is discarded:
    that is the hallucination this check exists to catch.

    In a course each chunk also carries its video, so the kept entry records
    which video the moment belongs to.

    Returns (kept, dropped_count, snapped_count).
    """
    if not chunks:
        return [], len(timestamps), 0

    kept, seen = [], set()
    dropped = snapped = 0

    for item in timestamps:
        seconds = parse_clock(item.get("time", ""))
        if seconds is None:
            dropped += 1
            continue

        match = None
        for chunk in chunks:
            if chunk.start - tolerance <= seconds <= chunk.end + tolerance:
                match = chunk
                break
        if match is None:
            dropped += 1
            continue

        if match.timestamp != str(item.get("time", "")).strip():
            snapped += 1
        key = (match.video_id, match.timestamp)
        if key in seen:
            continue
        seen.add(key)

        entry = {"time": match.timestamp, "topic": item.get("topic", "")}
        if match.video_index:
            entry["video_index"] = match.video_index
            entry["video_id"] = match.video_id
            entry["video_title"] = match.video_title
        kept.append(entry)

    kept.sort(key=lambda t: (t.get("video_index", 0), parse_clock(t["time"]) or 0.0))
    return kept, dropped, snapped


# --------------------------------------------------------------------------
# Offline extractive fallback
# --------------------------------------------------------------------------

def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 40]


def extractive_fallback(chunks: Sequence[RetrievedChunk]) -> SummaryResult:
    """Rank sentences by stop-word-filtered term frequency and select the top ones."""
    corpus = " ".join(c.text for c in chunks)
    sentences = _sentences(corpus)
    if not sentences:
        sentences = [corpus[:400]]

    freq: Dict[str, int] = {}
    for word in re.findall(r"[a-z0-9']+", corpus.lower()):
        if word not in STOP_WORDS and len(word) > 2:
            freq[word] = freq.get(word, 0) + 1

    def score(sentence: str) -> float:
        words = [w for w in re.findall(r"[a-z0-9']+", sentence.lower()) if w in freq]
        return sum(freq[w] for w in words) / (len(words) + 1e-6) if words else 0.0

    ranked = sorted(sentences, key=score, reverse=True)
    selected = ranked[: min(6, len(ranked))]
    ordered = [s for s in sentences if s in selected]

    timestamps = [
        {"time": c.timestamp, "topic": " ".join(c.text.split()[:7])}
        for c in list(chunks)[:6]
    ]
    top_terms = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:4]

    return SummaryResult(
        summary=" ".join(ordered[:4]),
        key_points=ordered,
        timestamps=timestamps,
        mind_map={
            "root": "Video topic (offline fallback)",
            "branches": [{"name": term, "children": []} for term, _ in top_terms],
        },
        model="offline-extractive-fallback",
        generated=False,
        warnings=[
            "Produced by the offline extractive fallback, not by a generative model. "
            "These sentences are copied from the transcript. Do not report this output "
            "or any metric computed from it as a result of the generative pipeline."
        ],
    )


# --------------------------------------------------------------------------
# Stage entry point
# --------------------------------------------------------------------------

def summarise(
    chunks: Sequence[RetrievedChunk],
    profile_key: str = "lecture",
    allow_fallback: bool = True,
) -> SummaryResult:
    context = build_context(chunks)

    if not settings.has_llm:
        if not allow_fallback:
            raise LLMError("No API key configured and the fallback is disabled.")
        log.warning("No Mistral API key configured — running the extractive fallback.")
        return extractive_fallback(chunks)

    prompt = build_summary_prompt(context, profile_key)
    started = time.time()
    raw = call_mistral(prompt, json_mode=True)
    elapsed = time.time() - started

    payload = _coerce(parse_response(raw))

    warnings: List[str] = []
    payload["timestamps"], dropped, snapped = validate_timestamps(
        payload["timestamps"], chunks)
    if dropped:
        warnings.append(
            "%d timestamp(s) pointed outside every retrieved chunk and were "
            "discarded." % dropped)
    if snapped:
        warnings.append(
            "%d timestamp(s) fell inside a retrieved chunk but did not match its "
            "label exactly, and were snapped to the chunk's own start time." % snapped)

    if not payload["summary"]:
        warnings.append("The model returned an empty summary field.")

    log.info("Summary generated by %s in %.1fs", settings.llm_model, elapsed)
    return SummaryResult(
        summary=payload["summary"],
        key_points=payload["key_points"],
        timestamps=payload["timestamps"],
        mind_map=payload["mind_map"],
        model=settings.llm_model,
        generated=True,
        raw_response=raw,
        latency_seconds=elapsed,
        warnings=warnings,
    )


def answer_question(
    question: str,
    chunks: Sequence[RetrievedChunk],
    profile_key: str = "lecture",
) -> str:
    if not settings.has_llm:
        return (
            "Question answering needs the generative model. Set MISTRAL_API_KEY to "
            "enable it."
        )
    return call_mistral(build_qa_prompt(build_context(chunks), question, profile_key)).strip()


# --------------------------------------------------------------------------
# Course-level synthesis
# --------------------------------------------------------------------------

@dataclass
class CourseResult:
    overview: str
    themes: List[Dict]
    progression: str
    study_order: List[Dict]
    key_takeaways: List[str]
    model: str
    generated: bool
    latency_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "overview": self.overview,
            "themes": self.themes,
            "progression": self.progression,
            "study_order": self.study_order,
            "key_takeaways": self.key_takeaways,
            "model": self.model,
            "generated_by_llm": self.generated,
            "latency_seconds": round(self.latency_seconds, 2),
            "warnings": self.warnings,
        }


def _summaries_block(video_summaries: Sequence[Dict]) -> str:
    """Render per-video summaries as the input to the course prompt."""
    parts = []
    for item in video_summaries:
        lines = ["V%d — %s" % (item["index"], item["title"])]
        if item.get("summary"):
            lines.append(item["summary"])
        for point in (item.get("key_points") or [])[:6]:
            lines.append("  - %s" % point)
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _course_fallback(video_summaries: Sequence[Dict]) -> CourseResult:
    """Offline stand-in: stitches the per-video summaries without a model."""
    overview = " ".join(
        (item.get("summary") or "").strip() for item in video_summaries
    )[:900]
    return CourseResult(
        overview=overview or "No per-video summaries were available.",
        themes=[],
        progression="",
        study_order=[{"video": item["index"], "why": "Original playlist order."}
                     for item in video_summaries],
        key_takeaways=[p for item in video_summaries
                       for p in (item.get("key_points") or [])[:2]],
        model="offline-course-fallback",
        generated=False,
        warnings=[
            "Produced without a generative model: the per-video summaries were "
            "concatenated, not synthesised. Do not report this as a course summary."
        ],
    )


def summarise_course(video_summaries: Sequence[Dict],
                     allow_fallback: bool = True) -> CourseResult:
    """Build one study guide from the per-video summaries.

    This is the reduce half of a map-reduce. A full course transcript is far too
    long for one prompt, so each video is summarised on its own first (the map),
    and only those summaries are synthesised here. That keeps the cost flat as
    the course grows: one extra call per video, plus this one.
    """
    if not video_summaries:
        raise LLMError("There are no video summaries to combine.")

    if not settings.has_llm:
        if not allow_fallback:
            raise LLMError("No API key configured and the fallback is disabled.")
        log.warning("No API key — stitching the course summary offline.")
        return _course_fallback(video_summaries)

    from .prompts import build_course_prompt

    prompt = build_course_prompt(_summaries_block(video_summaries))
    started = time.time()
    raw = call_mistral(prompt, json_mode=True)
    elapsed = time.time() - started
    payload = parse_response(raw)

    valid = {item["index"] for item in video_summaries}
    warnings: List[str] = []

    themes = []
    for theme in payload.get("themes") or []:
        if not isinstance(theme, dict) or not theme.get("name"):
            continue
        vids = [v for v in (theme.get("videos") or []) if v in valid]
        themes.append({
            "name": str(theme["name"]).strip(),
            "videos": vids,
            "detail": str(theme.get("detail", "")).strip(),
        })

    order, placed = [], set()
    for step in payload.get("study_order") or []:
        if not isinstance(step, dict):
            continue
        vid = step.get("video")
        if vid in valid and vid not in placed:
            placed.add(vid)
            order.append({"video": vid, "why": str(step.get("why", "")).strip()})
    missing = [v for v in sorted(valid) if v not in placed]
    if missing:
        warnings.append(
            "The study order omitted video(s) %s; they were appended in playlist order."
            % ", ".join("V%d" % v for v in missing))
        order.extend({"video": v, "why": ""} for v in missing)

    log.info("Course summary generated by %s in %.1fs", settings.llm_model, elapsed)
    return CourseResult(
        overview=str(payload.get("overview", "")).strip(),
        themes=themes,
        progression=str(payload.get("progression", "")).strip(),
        study_order=order,
        key_takeaways=[str(t).strip() for t in (payload.get("key_takeaways") or [])
                       if str(t).strip()],
        model=settings.llm_model,
        generated=True,
        latency_seconds=elapsed,
        warnings=warnings,
    )


def answer_across_course(question: str, chunks: Sequence[RetrievedChunk],
                         profile_key: str = "lecture") -> str:
    """Answer a question using chunks drawn from anywhere in the course.

    Identical to answer_question; it exists as its own name because the context
    blocks carry video labels and the answer is expected to cite them.
    """
    return answer_question(question, chunks, profile_key)


def naive_summary(chunks: Sequence[RetrievedChunk]) -> str:
    """The weak-prompt baseline used for the slide 4 before/after comparison."""
    from .prompts import NAIVE_PROMPT

    if not settings.has_llm:
        return "Needs MISTRAL_API_KEY — this comparison calls the model twice."
    return call_mistral(NAIVE_PROMPT.format(context=build_context(chunks)), temperature=0.7).strip()
