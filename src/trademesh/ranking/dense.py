"""Dense retrieval (proposal 9.4, System 3): bi-encoder embeddings, cosine similarity, exact search.

The encoder is pluggable. Production uses sentence-transformers `all-MiniLM-L6-v2` (384-d, loaded
lazily, so importing this module never needs torch). Tests use `HashingEncoder`, a dev-only
feature-hashing encoder that needs no model download; it is NOT a result-bearing system.
Embeddings are cached on disk keyed by a hash of (model, texts), so a re-run costs nothing.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from ..types import Hit, Passage


class Encoder(Protocol):
    name: str
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...     # (n, d), L2-normalised


class SentenceTransformerEncoder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.name, self.batch_size, self._model = model, batch_size, None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer     # optional dependency
            self._model = SentenceTransformer(self.name)
        return np.asarray(self._model.encode(list(texts), batch_size=self.batch_size,
                                             normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


class HashingEncoder:
    """Dev/test only: signed feature hashing of word unigrams and bigrams. Deterministic, offline."""
    name = "hashing-dev"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", t.lower())
            for feat in words + [" ".join(p) for p in zip(words, words[1:])]:
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                out[r, h % self.dim] += 1.0 if (h >> 64) & 1 else -1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norm == 0, 1, norm)


def _cache_path(cache_dir: Path, encoder_name: str, texts: Sequence[str]) -> Path:
    h = hashlib.sha1(encoder_name.encode())
    for t in texts:
        h.update(t.encode()); h.update(b"\0")
    return cache_dir / f"emb_{h.hexdigest()[:20]}.npy"


class DenseRetriever:
    def __init__(self, passages: list[Passage], encoder: Encoder, name: str = "dense",
                 cache_dir: str | Path | None = "cache", add_metadata: bool = False):
        self.name, self.encoder = name, encoder
        self.ids = [p.id for p in passages]
        texts = [f"{p.ticker} {p.company}. {p.text}" if add_metadata else p.text for p in passages]
        path = _cache_path(Path(cache_dir), encoder.name, texts) if cache_dir else None
        if path is not None and path.exists():
            self.matrix = np.load(path)
        else:
            self.matrix = np.asarray(encoder.encode(texts), dtype=np.float32) if texts else np.zeros((0, 1), np.float32)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, self.matrix)

    def search(self, query: str, k: int = 30) -> list[Hit]:
        if not len(self.ids):
            return []
        q = np.asarray(self.encoder.encode([query]), dtype=np.float32)[0]
        sims = self.matrix @ q                    # cosine, as both sides are unit length
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.lexsort((top, -sims[top]))]
        return [Hit(self.ids[i], float(sims[i])) for i in top]
