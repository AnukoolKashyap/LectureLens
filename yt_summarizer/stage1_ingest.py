"""Stage 1 — ingestion and transcript extraction.

Validates the URL, pulls video metadata, and fetches the timed transcript via
the YouTube Transcript API. Each transcript segment keeps its start time so the
timestamps that reach the final summary are real, not inferred.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings, SAMPLE_DIR
from .layer7_errors import (
    InvalidURLError,
    PipelineError,
    TranscriptUnavailableError,
    extract_video_id,
    get_logger,
    require_min_words,
)

log = get_logger(__name__)


@dataclass
class Segment:
    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class VideoDocument:
    video_id: str
    url: str
    title: str
    author: str
    segments: List[Segment]
    source: str = "youtube"  # or "local" when loaded from sample_data
    course_index: int = 0    # 1-based position in a course; 0 when standalone

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def duration_seconds(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["duration_seconds"] = self.duration_seconds
        d["word_count"] = len(self.full_text.split())
        return d


def format_timestamp(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%d:%02d:%02d" % (h, m, s)
    return "%d:%02d" % (m, s)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------

def fetch_metadata(video_id: str) -> Dict[str, str]:
    """Fetch title and author via YouTube's public oEmbed endpoint.

    Metadata is a nicety, not a requirement: if the call fails the pipeline
    continues with placeholders rather than aborting the run.
    """
    url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode(
        {"url": "https://www.youtube.com/watch?v=%s" % video_id, "format": "json"}
    )
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return {
            "title": payload.get("title", "Untitled video"),
            "author": payload.get("author_name", "Unknown channel"),
        }
    except Exception as exc:  # network blocked, video private, rate limited
        log.warning("Metadata lookup failed (%s); continuing without it.", exc)
        return {"title": "Video %s" % video_id, "author": "Unknown channel"}


# --------------------------------------------------------------------------
# Transcript
# --------------------------------------------------------------------------

def _fetch_transcript_segments(video_id: str) -> List[Segment]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as exc:  # pragma: no cover
        raise TranscriptUnavailableError(
            "youtube-transcript-api is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        # The library's API changed across major versions; support both shapes.
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            raw = YouTubeTranscriptApi.get_transcript(
                video_id, languages=list(settings.languages)
            )
        else:  # >= 1.0 instance API
            raw = YouTubeTranscriptApi().fetch(
                video_id, languages=list(settings.languages)
            ).to_raw_data()
    except Exception as exc:
        raise TranscriptUnavailableError(
            "No transcript could be retrieved for this video (%s)." % type(exc).__name__
        ) from exc

    segments = [
        Segment(
            text=str(item["text"]).replace("\n", " ").strip(),
            start=float(item.get("start", 0.0)),
            duration=float(item.get("duration", 0.0)),
        )
        for item in raw
        if str(item.get("text", "")).strip()
    ]
    if not segments:
        raise TranscriptUnavailableError()
    return segments


def load_local_transcript(path: Path) -> VideoDocument:
    """Load a transcript from disk.

    Supports the JSON shape produced by `export_transcript` and a plain .txt
    file (which is split into 30-second pseudo-segments so downstream stages
    still receive timings).
    """
    path = Path(path)
    if not path.exists():
        raise TranscriptUnavailableError("No transcript file at %s" % path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        segments = [
            Segment(text=s["text"], start=float(s["start"]), duration=float(s.get("duration", 0)))
            for s in payload["segments"]
        ]
        return VideoDocument(
            video_id=payload.get("video_id", path.stem),
            url=payload.get("url", "local://%s" % path.name),
            title=payload.get("title", path.stem),
            author=payload.get("author", "Local file"),
            segments=segments,
            source="local",
        )

    text = path.read_text(encoding="utf-8")
    words = text.split()
    segments, bucket, start = [], [], 0.0
    for word in words:
        bucket.append(word)
        if len(bucket) >= 75:  # ~30 s of speech at 150 wpm
            segments.append(Segment(" ".join(bucket), start, 30.0))
            start += 30.0
            bucket = []
    if bucket:
        segments.append(Segment(" ".join(bucket), start, 30.0))

    return VideoDocument(
        video_id=path.stem,
        url="local://%s" % path.name,
        title=path.stem.replace("_", " ").title(),
        author="Local file",
        segments=segments,
        source="local",
    )


def export_transcript(doc: VideoDocument, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video_id": doc.video_id,
        "url": doc.url,
        "title": doc.title,
        "author": doc.author,
        "segments": [asdict(s) for s in doc.segments],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def ingest(url: str, local_path: Optional[Path] = None) -> VideoDocument:
    """Stage 1 entry point.

    `local_path` lets the demo run without network access — the pipeline is
    identical from stage 2 onward, only the source of the transcript changes.
    """
    if local_path is not None:
        doc = load_local_transcript(Path(local_path))
        log.info("Loaded local transcript '%s' (%d segments)", doc.title, len(doc.segments))
        require_min_words(doc.full_text)
        return doc

    video_id = extract_video_id(url)
    log.info("Validated URL, video id = %s", video_id)

    meta = fetch_metadata(video_id)
    segments = _fetch_transcript_segments(video_id)

    doc = VideoDocument(
        video_id=video_id,
        url="https://www.youtube.com/watch?v=%s" % video_id,
        title=meta["title"],
        author=meta["author"],
        segments=segments,
    )
    log.info(
        "Transcript fetched: %d segments, %d words, %s long",
        len(segments), len(doc.full_text.split()), format_timestamp(doc.duration_seconds),
    )
    require_min_words(doc.full_text)
    return doc


# --------------------------------------------------------------------------
# Courses: several videos at once
# --------------------------------------------------------------------------

_PLAYLIST_ID = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")

# Playlist id -> its title, filled in as playlists are read.
PLAYLIST_TITLES: Dict[str, str] = {}


def course_title(documents, sources=None) -> str:
    """Name a course.

    In order of preference: the playlist's own title; a shared prefix of the
    video titles long enough to be meaningful ("Linear Algebra Lecture 1",
    "... Lecture 2" -> "Linear Algebra Lecture"); otherwise the first video's
    title with a count. Never a sentence pulled out of a transcript.
    """
    for key, title in PLAYLIST_TITLES.items():
        if title and (not sources or key in str(sources)):
            return title

    titles = [d.title for d in documents if d.title]
    if not titles:
        return "Course"
    if len(titles) == 1:
        return titles[0]

    prefix = titles[0]
    for other in titles[1:]:
        while prefix and not other.startswith(prefix):
            prefix = prefix[:-1]
    prefix = prefix.rstrip(" -\u2013\u2014:|.#0123456789")
    if len(prefix) >= 12:
        return prefix

    return "%s (+%d more video%s)" % (
        titles[0], len(titles) - 1, "" if len(titles) == 2 else "s")


def extract_playlist_id(url: str) -> Optional[str]:
    """Return the playlist id in a YouTube URL, or None if there is not one."""
    if not url:
        return None
    match = _PLAYLIST_ID.search(url.strip())
    return match.group(1) if match else None


def list_playlist_videos(url_or_id: str) -> List[str]:
    """Return the video URLs of a playlist, in order.

    The transcript API works one video at a time and cannot enumerate a
    playlist, so this uses yt-dlp in flat mode — it reads the playlist index
    without downloading any media. If yt-dlp is not installed the caller is
    told to paste the video links instead, which needs no extra package.
    """
    playlist_id = extract_playlist_id(url_or_id) or url_or_id.strip()
    try:
        import yt_dlp
    except ImportError as exc:
        raise PipelineError(
            "Reading a playlist needs yt-dlp, which is not installed. Either run "
            "`pip install yt-dlp`, or paste the individual video links instead."
        ) from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": "in_playlist",  # index only, no media
        "skip_download": True,
    }
    target = "https://www.youtube.com/playlist?list=%s" % playlist_id
    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(target, download=False)
    except Exception as exc:
        raise PipelineError(
            "Could not read that playlist (%s). Check that it is public or "
            "unlisted rather than private." % type(exc).__name__
        ) from exc

    entries = [e for e in (info.get("entries") or []) if e]
    urls = []
    for entry in entries:
        vid = entry.get("id")
        if vid:
            urls.append("https://www.youtube.com/watch?v=%s" % vid)
    if not urls:
        raise PipelineError("That playlist appears to be empty.")

    # Remembered so the course can be named after the playlist rather than
    # after a sentence lifted out of the first video.
    PLAYLIST_TITLES[playlist_id] = str(info.get("title") or "").strip()

    log.info("Playlist %s (%r) holds %d video(s)",
             playlist_id, PLAYLIST_TITLES[playlist_id], len(urls))
    return urls


def expand_sources(sources) -> List[str]:
    """Turn whatever the user supplied into a flat, ordered list of video URLs.

    Accepts a playlist link, a single video link, or several links separated by
    newlines, commas or spaces. Duplicates are dropped, order is kept.
    """
    if isinstance(sources, str):
        raw = [sources]
    else:
        raw = list(sources)

    pieces: List[str] = []
    for item in raw:
        pieces.extend(p for p in re.split(r"[\s,]+", str(item).strip()) if p)

    urls: List[str] = []
    for piece in pieces:
        # A watch link that also carries ?list= is one video, not the playlist.
        if extract_playlist_id(piece) and "/watch?" not in piece and "youtu.be/" not in piece:
            urls.extend(list_playlist_videos(piece))
        else:
            urls.append(piece)

    seen, ordered = set(), []
    for url in urls:
        try:
            key = extract_video_id(url)
        except InvalidURLError:
            key = url
        if key not in seen:
            seen.add(key)
            ordered.append(url)
    return ordered


@dataclass
class SkippedVideo:
    url: str
    reason: str

    def to_dict(self) -> Dict:
        return {"url": self.url, "reason": self.reason}


def ingest_many(sources, progress=None) -> tuple:
    """Fetch every video of a course.

    Returns (documents, skipped). One video without captions must not sink a
    30-video course, so a failure is recorded and the rest continue. The caller
    decides what to tell the user about the ones that were skipped.
    """
    urls = expand_sources(sources)
    if not urls:
        raise InvalidURLError("No video links were found in that input.")

    docs: List[VideoDocument] = []
    skipped: List[SkippedVideo] = []

    for i, url in enumerate(urls, 1):
        if progress:
            progress(i, len(urls), url)
        try:
            doc = ingest(url)
            doc.course_index = len(docs) + 1
            docs.append(doc)
        except PipelineError as exc:
            log.warning("Skipping %s — %s", url, exc.user_message)
            skipped.append(SkippedVideo(url=url, reason=exc.user_message))

    if not docs:
        raise TranscriptUnavailableError(
            "None of the %d video(s) had a usable transcript." % len(urls)
        )

    log.info("Course ingested: %d video(s) usable, %d skipped", len(docs), len(skipped))
    return docs, skipped


def load_local_course(paths) -> List[VideoDocument]:
    """Load several local transcripts as a course. Used by the offline tests."""
    docs = []
    for i, path in enumerate(paths, 1):
        doc = load_local_transcript(Path(path))
        doc.course_index = i
        docs.append(doc)
    return docs


def default_sample() -> Path:
    return SAMPLE_DIR / "sample_lecture.json"
