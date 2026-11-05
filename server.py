#!/usr/bin/env python3
"""Web app: a FastAPI backend over the pipeline, serving the frontend in web/.

    python server.py              then open http://localhost:8000
    python server.py --port 9000  pick another port

Endpoints
    GET  /api/status      model, profiles, defaults
    POST /api/summarize   runs stages 1-6, streams progress as NDJSON lines
    POST /api/ask         answers a question from the current video's chunks
    POST /api/feedback    appends a rating to the feedback log
    GET  /api/export/md   current summary as Markdown
    GET  /api/export/json current summary as JSON

The server holds one video at a time: the vector store keeps a single
collection, so summarising a new video replaces the previous one.
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from yt_summarizer.config import settings
from yt_summarizer.layer7_errors import PipelineError
from yt_summarizer.layer9_eval import record_feedback
from yt_summarizer.pipeline import CourseRun, PipelineRun, run_course, run_pipeline
from yt_summarizer.prompts import USE_CASE_PROFILES
from yt_summarizer.stage1_ingest import default_sample
from yt_summarizer.stage3_embed import build_embedder
from yt_summarizer.stage4_retrieve import retrieve
from yt_summarizer.stage5_summarize import answer_question
from yt_summarizer.stage6_output import to_markdown

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="AI-Powered Video Summarization")

_current: Dict = {"run": None, "course": None, "profile": "lecture"}
_pipeline_lock = threading.Lock()

STAGES = {
    1: "Fetching transcript",
    2: "Cleaning and chunking",
    3: "Embedding into the vector store",
    4: "Retrieving key passages",
    5: "Writing the summary",
    6: "Assembling output",
}


# --------------------------------------------------------------------------
# Request bodies
# --------------------------------------------------------------------------

class SummarizeRequest(BaseModel):
    url: str = ""
    # Course mode: a playlist link, or several video links. Either one switches
    # the run from a single video to a whole course.
    playlist: str = ""
    urls: List[str] = []
    use_sample: bool = False
    profile: str = "lecture"
    top_k: Optional[int] = None
    offline: bool = False
    reference: str = ""


class Turn(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    history: List[Turn] = []


class FeedbackRequest(BaseModel):
    rating: int
    comment: str = ""


class WhereIsRequest(BaseModel):
    topic: str
    limit: int = 5


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _serialise(run: PipelineRun) -> Dict:
    data = run.summary.to_dict()
    data["evaluation"] = [
        {"label": label, "value": value} for label, value in run.evaluation.display_rows()
    ]
    data["rouge_computed"] = run.evaluation.rouge_1 is not None
    data["sources"] = [
        {"time": c.timestamp, "start": c.start, "text": c.text, "score": round(c.final_score, 3)}
        for c in run.retrieved
    ]
    segments = run.document.segments
    data["stats"] = {
        "words": len(run.document.full_text.split()),
        "chunks": len(run.chunks),
        "retrieved": len(run.retrieved),
        "duration": round(segments[-1].start + segments[-1].duration) if segments else 0,
        "source": run.document.source,
    }
    data["stage_times"] = run.stage_times
    return data


def _serialise_course(run: CourseRun) -> Dict:
    """Serialise a course.

    The payload deliberately also fills the single-video fields the existing UI
    already renders (summary, key_points, mind_map, evaluation, safety), so a
    course displays without the front end having to change. The course-only
    fields — videos, themes, study_order — ride alongside for a richer view.

    `timestamps` is left empty on purpose: the player holds one video, so a
    merged cross-video timeline would seek to the wrong place. Per-video
    moments stay inside each entry of `videos`.
    """
    report = run.report
    data = report.to_dict()
    data["mode"] = "course"

    first_url = report.videos[0].url if report.videos else ""
    data["video"] = {
        "video_id": "course",
        "title": report.title or "Course summary",
        "author": "%d video%s" % (len(report.videos), "" if len(report.videos) == 1 else "s"),
        "url": first_url,
    }

    # Takeaways first, then a one-line index of the course.
    points = list(report.key_takeaways)
    for video in report.videos:
        number = video.diagnostics.get("course_index", 0)
        head = (video.summary or "").split(". ")[0].strip()
        points.append("V%d — %s%s" % (number, video.title, (" — " + head) if head else ""))
    data["key_points"] = points
    data["summary"] = report.overview
    data["timestamps"] = []

    # A mind map of themes reads far better than one stitched from six videos.
    data["mind_map"] = {
        "root": report.title or "Course",
        "branches": [
            {"name": t["name"],
             "children": [_video_title(report, v) for v in t.get("videos", [])]}
            for t in report.themes
        ] or [{"name": v.title, "children": []} for v in report.videos],
    }

    rows = []
    if report.metrics.get("faithfulness") is not None:
        rows.append({"label": "Faithfulness (course mean)",
                     "value": "%.3f" % report.metrics["faithfulness"]})
    if report.metrics.get("relevance") is not None:
        rows.append({"label": "Relevance (course mean)",
                     "value": "%.3f" % report.metrics["relevance"]})
    if report.metrics.get("compression_ratio") is not None:
        rows.append({"label": "Compression (course mean)",
                     "value": "%.1fx" % report.metrics["compression_ratio"]})
    rows.append({"label": "Videos summarised", "value": str(len(report.videos))})
    data["evaluation"] = rows
    data["rouge_computed"] = False
    data["safety"] = {"counts": report.diagnostics.get("safety", {})}
    # The player holds one video, so course passages link out to their own
    # video at the right moment instead of seeking the embedded one.
    data["sources"] = [
        {"time": c.timestamp, "start": c.start, "text": c.text,
         "score": round(c.final_score, 3), "video": c.video_title,
         "video_index": c.video_index, "url": c.watch_url()}
        for c in run.retrieved
    ]
    data["stage_times"] = report.diagnostics.get("stage_times", {})
    data["stats"] = {
        "videos": len(report.videos),
        "skipped": len(report.skipped),
        "chunks": report.diagnostics.get("chunk_count", 0),
        "retrieved": sum(v.diagnostics.get("retrieved_count", 0) for v in report.videos),
        "words": report.metrics.get("transcript_words", 0),
        "duration": 0,
        "seconds": report.diagnostics.get("total_seconds", 0),
        "source": "course",
    }
    return data


def _video_title(report, number: int) -> str:
    for v in report.videos:
        if v.diagnostics.get("course_index") == number:
            return v.title
    return "Video %d" % number


def _current_course() -> CourseRun:
    run = _current.get("course")
    if run is None:
        raise HTTPException(status_code=409, detail="Summarise a course first.")
    return run


def _current_run() -> PipelineRun:
    run = _current["run"]
    if run is None:
        raise HTTPException(status_code=409, detail="Summarise a video first.")
    return run


# "[2]", optionally followed by the block's own "(8:55)" the model tends to echo.
_CITATION = re.compile(r"\[(\d+)\](?:\s*\(\d{1,2}:\d{2}(?::\d{2})?\))?")


def _cite_timestamps(answer: str, hits) -> tuple:
    """The model cites context blocks as [n]; swap in the block's timestamp."""
    cited: List[int] = []

    def swap(match):
        n = int(match.group(1))
        if 0 < n <= len(hits):
            if n not in cited:
                cited.append(n)
            return "[%s]" % hits[n - 1].timestamp
        return match.group(0)

    answer = _CITATION.sub(swap, answer)
    picks = [hits[n - 1] for n in cited] or list(hits[:2])
    return answer, [{"time": h.timestamp, "start": h.start} for h in picks]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@app.get("/api/status")
def status() -> Dict:
    return {
        "model": settings.llm_model,
        "llm": settings.has_llm,
        "offline": settings.offline_mode,
        "top_k": settings.top_k,
        "profiles": [{"key": k, "label": p.label} for k, p in USE_CASE_PROFILES.items()],
        "has_run": _current["run"] is not None,
        "has_course": _current.get("course") is not None,
        "course_supported": True,
    }


COURSE_STAGES = {
    1: "Fetching transcripts",
    2: "Cleaning and chunking every video",
    3: "Embedding the whole course",
    4: "Summarising each video",
    5: "Building the course guide",
    6: "Assembling output",
}


@app.post("/api/summarize")
def summarize(req: SummarizeRequest) -> StreamingResponse:
    course_sources = req.playlist.strip() or [u for u in req.urls if u.strip()]
    if not course_sources and not req.url.strip() and not req.use_sample:
        raise HTTPException(
            status_code=400,
            detail="Enter a YouTube URL, a playlist, or use the sample.")
    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Another video is being summarised.")

    events: "queue.Queue[Optional[Dict]]" = queue.Queue()

    def progress(step: int, label: str) -> None:
        table = COURSE_STAGES if course_sources else STAGES
        text = table.get(step, "")
        # run_course passes "video 2 of 7" through on step 4; keep that detail.
        if course_sources and label and label.startswith("video "):
            text = "%s — %s" % (text, label)
        events.put({"type": "progress", "step": step, "label": text})

    def work() -> None:
        try:
            settings.offline_mode = req.offline
            if req.top_k:
                settings.top_k = max(1, min(15, req.top_k))

            if course_sources:
                run = run_course(
                    sources=course_sources,
                    profile_key=req.profile,
                    progress=progress,
                )
                _current["course"] = run
                _current["run"] = None
                _current["profile"] = req.profile
                events.put({"type": "result", "data": _serialise_course(run)})
                return

            run = run_pipeline(
                url=req.url.strip(),
                local_path=default_sample() if req.use_sample else None,
                profile_key=req.profile,
                reference_summary=req.reference.strip() or None,
                progress=progress,
            )
            _current["run"] = run
            _current["course"] = None
            _current["profile"] = req.profile
            events.put({"type": "result", "data": _serialise(run)})
        except PipelineError as exc:
            events.put({"type": "error", "message": exc.user_message, "detail": str(exc)})
        except Exception as exc:  # surfaced to the UI rather than a bare 500
            events.put({"type": "error", "message": "Unexpected error.", "detail": repr(exc)})
        finally:
            _pipeline_lock.release()
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                return
            yield json.dumps(event, ensure_ascii=False, default=str) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/ask")
def ask(req: AskRequest) -> Dict:
    course = _current.get("course")
    run = course if course is not None else _current_run()
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    # Follow-ups like "explain that more" need the previous question to retrieve
    # the right chunks and to make sense to the model.
    previous = [t for t in req.history[-4:] if t.role in ("user", "assistant")]
    last_user = next((t.content for t in reversed(previous) if t.role == "user"), "")
    query = ("%s %s" % (last_user, question)).strip()
    prompt_question = question
    if previous:
        transcript = "\n".join(
            "%s: %s" % ("Student" if t.role == "user" else "Tutor", t.content[:600])
            for t in previous
        )
        prompt_question = "(Earlier in this conversation:\n%s)\n\n%s" % (transcript, question)

    try:
        if course is not None:
            from yt_summarizer.stage4_retrieve import retrieve_across_course
            hits = retrieve_across_course(query, run.index)
        else:
            hits = retrieve(query, run.index)
        answer = answer_question(prompt_question, hits, _current["profile"])
    except PipelineError as exc:
        raise HTTPException(status_code=502, detail="%s %s" % (exc.user_message, exc))

    answer, sources = _cite_timestamps(answer, hits)
    return {"answer": answer, "sources": sources}


@app.post("/api/where-is")
def where_is(req: WhereIsRequest) -> Dict:
    """Every place a topic is covered across the course."""
    run = _current_course()
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Empty topic.")
    return {"topic": topic, "hits": run.where_is(topic, max(1, min(20, req.limit)))}


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> Dict:
    run = _current_run()
    path = record_feedback(run.summary.video_id, max(1, min(5, req.rating)), req.comment)
    return {"saved_to": str(path)}


@app.get("/api/export/{fmt}")
def export(fmt: str) -> Response:
    course = _current.get("course")
    if course is not None:
        from yt_summarizer.stage6_output import course_to_markdown
        report = course.report
        if fmt == "md":
            return Response(
                course_to_markdown(report),
                media_type="text/markdown",
                headers={"Content-Disposition": 'attachment; filename="course_summary.md"'},
            )
        if fmt == "json":
            return Response(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False, default=str),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="course_summary.json"'},
            )
        raise HTTPException(status_code=404, detail="Use md or json.")

    summary = _current_run().summary
    if fmt == "md":
        return Response(
            to_markdown(summary),
            media_type="text/markdown",
            headers={"Content-Disposition": 'attachment; filename="%s_summary.md"' % summary.video_id},
        )
    if fmt == "json":
        return Response(
            json.dumps(summary.to_dict(), indent=2, ensure_ascii=False, default=str),
            media_type="application/json",
            headers={"Content-Disposition": 'attachment; filename="%s_summary.json"' % summary.video_id},
        )
    raise HTTPException(status_code=404, detail="Use md or json.")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


# --------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    # Load the embedding weights now so the first summary does not wait for them.
    threading.Thread(target=build_embedder, daemon=True).start()

    url = "http://localhost:%d" % args.port
    print("\n  Video Summarizer running at %s\n" % url)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
