"""Stage 3 — embedding and vector store.

Chunks are encoded with the sentence-transformers model named on the pipeline
diagram (all-MiniLM-L6-v2, an encoder-only Transformer) and indexed in ChromaDB.

Two fallbacks exist so a demo never dies on a missing download:

  * `HashingBackend` replaces the Transformer encoder when the model weights
    cannot be downloaded. It is a lexical hashing vectoriser, NOT a semantic
    model, and every result carries `degraded=True` so no measurement taken in
    this mode is ever reported as a result of the real system.
  * `MemoryStore` replaces ChromaDB when the package is unavailable. Same
    interface, brute-force cosine search.

Neither fallback is used when the real dependencies are installed.
"""
from __future__ import annotations

import threading

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import settings
from .layer7_errors import EmbeddingError, VectorStoreError, get_logger
from .stage2_preprocess import Chunk

log = get_logger(__name__)

EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------

class EmbeddingBackend:
    name = "base"
    degraded = False

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


# Loaded weights, keyed by model name. Loading takes seconds, so a long-running
# server should pay that once rather than on every video.
_LOADED_MODELS: dict = {}
_LOAD_LOCK = threading.Lock()


class SentenceTransformerBackend(EmbeddingBackend):
    """The real encoder: all-MiniLM-L6-v2."""

    degraded = False

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError("sentence-transformers is not installed.") from exc
        try:
            with _LOAD_LOCK:
                if model_name not in _LOADED_MODELS:
                    _LOADED_MODELS[model_name] = SentenceTransformer(model_name)
            self._model = _LOADED_MODELS[model_name]
        except Exception as exc:
            raise EmbeddingError(
                "Could not load '%s'. The weights download from Hugging Face may have "
                "failed. %s" % (model_name, exc)
            ) from exc
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vectors]


class HashingBackend(EmbeddingBackend):
    """Offline fallback. Lexical, not semantic — results are marked degraded."""

    name = "hashing-fallback"
    degraded = True

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> List[str]:
        words = re.findall(r"[a-z0-9']+", text.lower())
        bigrams = ["%s_%s" % (a, b) for a, b in zip(words, words[1:])]
        return words + bigrams

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in self._tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                slot = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[slot] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def build_embedder(force_fallback: bool = False) -> EmbeddingBackend:
    if force_fallback or settings.offline_mode:
        log.warning("Using the hashing fallback embedder — results are degraded.")
        return HashingBackend()
    try:
        backend = SentenceTransformerBackend(settings.embedding_model)
        log.info("Embedding model loaded: %s", backend.name)
        return backend
    except EmbeddingError as exc:
        log.warning("%s Falling back to the hashing embedder.", exc.user_message)
        return HashingBackend()


# --------------------------------------------------------------------------
# Vector stores
# --------------------------------------------------------------------------

@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    metadata: Dict


class BaseStore:
    backend_name = "base"

    def reset(self, name: str) -> None:
        raise NotImplementedError

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        raise NotImplementedError

    def query(self, vector: Sequence[float], k: int,
              where: Optional[Dict] = None) -> List[Hit]:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class ChromaStore(BaseStore):
    backend_name = "chromadb"

    def __init__(self, path: str):
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreError("chromadb is not installed.") from exc
        try:
            self._client = chromadb.PersistentClient(path=path)
        except Exception as exc:
            raise VectorStoreError("Could not open the ChromaDB store at %s" % path) from exc
        self._collection = None

    def reset(self, name: str) -> None:
        try:
            self._client.delete_collection(name)
        except Exception:
            pass  # collection did not exist yet
        self._collection = self._client.create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if self._collection is None:
            raise VectorStoreError("Collection not initialised; call reset() first.")
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[list(v) for v in vectors],
            metadatas=[c.metadata() for c in chunks],
        )

    def query(self, vector: Sequence[float], k: int,
              where: Optional[Dict] = None) -> List[Hit]:
        kwargs = {
            "query_embeddings": [list(vector)],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        result = self._collection.query(**kwargs)
        hits: List[Hit] = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append(
                Hit(chunk_id=meta["chunk_id"], text=doc, score=1.0 - float(dist), metadata=dict(meta))
            )
        return hits

    def count(self) -> int:
        return self._collection.count() if self._collection else 0


class MemoryStore(BaseStore):
    """Brute-force cosine store used only when ChromaDB is unavailable."""

    backend_name = "in-memory-fallback"

    def __init__(self):
        self._ids: List[str] = []
        self._texts: List[str] = []
        self._vectors: List[List[float]] = []
        self._metas: List[Dict] = []

    def reset(self, name: str) -> None:
        self._ids, self._texts, self._vectors, self._metas = [], [], [], []

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        for chunk, vector in zip(chunks, vectors):
            self._ids.append(chunk.chunk_id)
            self._texts.append(chunk.text)
            self._vectors.append(list(vector))
            self._metas.append(chunk.metadata())

    def query(self, vector: Sequence[float], k: int,
              where: Optional[Dict] = None) -> List[Hit]:
        scored: List[Tuple[float, int]] = []
        for i, stored in enumerate(self._vectors):
            if where and any(self._metas[i].get(key) != val for key, val in where.items()):
                continue
            scored.append((sum(a * b for a, b in zip(vector, stored)), i))
        scored.sort(reverse=True)
        return [
            Hit(
                chunk_id=self._ids[i],
                text=self._texts[i],
                score=float(score),
                metadata=dict(self._metas[i]),
            )
            for score, i in scored[:k]
        ]

    def count(self) -> int:
        return len(self._ids)


def build_store() -> BaseStore:
    try:
        store = ChromaStore(settings.chroma_path)
        log.info("Vector store: ChromaDB at %s", settings.chroma_path)
        return store
    except VectorStoreError as exc:
        log.warning("%s Falling back to the in-memory store.", exc.user_message)
        return MemoryStore()


# --------------------------------------------------------------------------
# Stage entry point
# --------------------------------------------------------------------------

@dataclass
class IndexResult:
    store: BaseStore
    embedder: EmbeddingBackend
    chunk_count: int
    degraded: bool


def build_index(
    chunks: List[Chunk],
    embedder: Optional[EmbeddingBackend] = None,
    store: Optional[BaseStore] = None,
) -> IndexResult:
    if not chunks:
        raise VectorStoreError("There are no chunks to index.")

    embedder = embedder or build_embedder()
    store = store or build_store()

    store.reset(settings.collection_name)
    vectors = embedder.encode([c.text for c in chunks])
    store.add(chunks, vectors)

    log.info(
        "Indexed %d chunks into %s using %s",
        store.count(), store.backend_name, embedder.name,
    )
    return IndexResult(
        store=store,
        embedder=embedder,
        chunk_count=store.count(),
        degraded=embedder.degraded,
    )
