"""Layer 7 — error handling and validation.

Cross-cutting layer applied across stages 1-6. Every failure the pipeline can
reasonably hit is converted into a typed exception with a message that is safe
to show a user, so no stage ever surfaces a raw stack trace in the UI.
"""
from __future__ import annotations

import logging
import re
import sys
from typing import Optional

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


log = get_logger(__name__)


class PipelineError(Exception):
    """Base class. Carries a message intended for the end user."""

    user_message = "Something went wrong while processing this video."

    def __init__(self, message: Optional[str] = None):
        self.user_message = message or self.user_message
        super().__init__(self.user_message)


class InvalidURLError(PipelineError):
    user_message = "That does not look like a YouTube video link. Check the URL and try again."


class TranscriptUnavailableError(PipelineError):
    user_message = (
        "This video has no transcript available. Captions may be disabled, or the video "
        "may be in a language the transcript service does not cover."
    )


class TranscriptTooShortError(PipelineError):
    user_message = "The transcript is too short to summarise meaningfully."


class EmbeddingError(PipelineError):
    user_message = "The embedding model could not be loaded. See the README for setup steps."


class VectorStoreError(PipelineError):
    user_message = "The vector store could not be opened or written to."


class LLMError(PipelineError):
    user_message = "The summarisation model could not be reached."


class UnsupportedContentError(PipelineError):
    user_message = "This content was blocked by the safety screen and was not summarised."


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

_YOUTUBE_PATTERNS = (
    re.compile(r"(?:youtube\.com/watch\?(?:.*&)?v=)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtu\.be/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})"),
    re.compile(r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})"),
)

_BARE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url: str) -> str:
    """Return the 11-character YouTube video id, or raise InvalidURLError."""
    if not url or not isinstance(url, str):
        raise InvalidURLError()
    candidate = url.strip()

    if _BARE_ID.match(candidate):
        return candidate

    for pattern in _YOUTUBE_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)

    raise InvalidURLError()


def require_min_words(text: str, minimum: int = 150) -> None:
    words = len(text.split())
    if words < minimum:
        raise TranscriptTooShortError(
            "The transcript contains only %d words; at least %d are needed to build a "
            "useful summary." % (words, minimum)
        )
