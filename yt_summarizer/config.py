"""Central configuration for the AI-Powered Video Summarization pipeline.

Every number quoted on the Review 2 pipeline diagram lives here, so the slide
and the code cannot drift apart. Change a value here and the diagram caption is
the only other place that needs updating.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: loads a .env file when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not a requirement
    pass

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
EXPORT_DIR = ROOT / "exports"
SAMPLE_DIR = ROOT / "sample_data"

for _d in (DATA_DIR, CHROMA_DIR, EXPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    # --- Stage 2: preprocessing and chunking -------------------------------
    chunk_size: int = 1000           # characters per chunk
    chunk_overlap: int = 200         # characters shared with the previous chunk
    min_chunk_chars: int = 120       # discard fragments shorter than this

    # --- Stage 3: embedding and vector store -------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "video_chunks"
    chroma_path: str = str(CHROMA_DIR)

    # --- Stage 4: retrieval ------------------------------------------------
    top_k: int = 5                   # chunks returned to the summariser
    fetch_k: int = 12                # candidates pulled before re-ranking
    rerank: bool = True

    # --- Stage 5: summarisation -------------------------------------------
    llm_model: str = "open-mistral-nemo"
    llm_endpoint: str = "https://api.mistral.ai/v1/chat/completions"
    temperature: float = 0.2
    max_tokens: int = 1600
    request_timeout: int = 90

    # --- Stage 6: output ---------------------------------------------------
    export_dir: str = str(EXPORT_DIR)

    # --- Layer 9: evaluation ----------------------------------------------
    target_compression: float = 10.0  # transcript words / summary words

    # --- Runtime -----------------------------------------------------------
    offline_mode: bool = field(default_factory=lambda: os.getenv("OFFLINE_MODE", "0") == "1")
    languages: tuple = ("en", "en-US", "en-GB", "hi")

    @property
    def mistral_api_key(self) -> str:
        return os.getenv("MISTRAL_API_KEY", "").strip()

    @property
    def has_llm(self) -> bool:
        return bool(self.mistral_api_key) and not self.offline_mode


settings = Settings()
