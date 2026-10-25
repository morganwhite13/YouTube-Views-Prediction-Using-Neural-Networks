"""On-disk cache for pretrained-model embeddings.

Encoding thousands of titles with a sentence-transformer (or thumbnails with CLIP) on a CPU
takes minutes, and a training run encodes the same texts several times (train split, refit,
every tuning trial). Embeddings are cached by a hash of the *content* (the text, or the image
file's bytes), per model, in a small SQLite file.

Location: ``$YTVIEWS_CACHE_DIR`` (default ``~/.cache/ytviews``). Set ``YTVIEWS_CACHE=0`` to
disable. Thumbnail face detection is cached the same way; the offline TF-IDF encoder and plain
image statistics are fast and are not cached.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def cache_dir() -> Path:
    return Path(os.environ.get("YTVIEWS_CACHE_DIR", Path.home() / ".cache" / "ytviews"))


def enabled() -> bool:
    return os.environ.get("YTVIEWS_CACHE", "1").lower() not in ("0", "false", "no", "off")


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_key(path) -> str | None:
    """Hash of a file's bytes, or None when there is no readable file."""
    if path is None or (isinstance(path, float) and path != path) or not str(path):
        return None
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except (OSError, ValueError):
        return None


class EmbeddingCache:
    """One SQLite connection to the cache file; use as a context manager so it is closed."""

    def __init__(self, namespace: str, directory: Path | None = None) -> None:
        directory = directory or cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        # A generous timeout lets concurrent writers (API threads, parallel runs) take turns.
        self.conn = sqlite3.connect(directory / "embeddings.sqlite", timeout=30)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (ns TEXT, key TEXT, vec BLOB, PRIMARY KEY (ns, key))"
        )

    def __enter__(self) -> EmbeddingCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(keys))
        for i in range(0, len(unique), 500):
            chunk = unique[i : i + 500]
            rows = self.conn.execute(
                f"SELECT key, vec FROM emb WHERE ns = ? AND key IN ({','.join('?' * len(chunk))})",
                [self.namespace, *chunk],
            )
            out.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
                [
                    (self.namespace, k, np.asarray(v, np.float32).tobytes())
                    for k, v in items.items()
                ],
            )

    def size(self) -> int:
        q = "SELECT COUNT(*) FROM emb WHERE ns = ?"
        return int(self.conn.execute(q, [self.namespace]).fetchone()[0])


def cached_encode(
    namespace: str,
    keys: Sequence[str],
    items: Sequence,
    encode: Callable[[list], np.ndarray],
    dim: int,
) -> np.ndarray:
    """Encode ``items`` (identified by ``keys``), reusing cached vectors and de-duplicating.

    ``encode`` receives only the unique, uncached items; it may return a NaN row for an item it
    could not encode, which is passed through and never cached. The cache is an optimisation:
    if it can't be read or written (locked, read-only disk), encoding still succeeds.
    """
    out = np.zeros((len(items), dim), dtype=np.float32)
    if not items:
        return out
    cache = None
    known: dict[str, np.ndarray] = {}
    if enabled():
        try:
            cache = EmbeddingCache(namespace)
            known = cache.get_many(keys)
        except (sqlite3.Error, OSError) as exc:
            log.warning("embedding cache unavailable (%s); encoding without it", exc)
            cache = None
    try:
        todo = {}
        for key, item in zip(keys, items, strict=True):
            if key not in known and key not in todo:
                todo[key] = item
        if todo:
            fresh = np.asarray(encode(list(todo.values())), dtype=np.float32)
            new = dict(zip(todo, fresh, strict=True))
            known.update(new)
            if cache:
                try:
                    cache.put_many({k: v for k, v in new.items() if np.isfinite(v).all()})
                except sqlite3.Error as exc:
                    log.warning("could not write the embedding cache (%s)", exc)
    finally:
        if cache:
            cache.close()
    for i, key in enumerate(keys):
        out[i] = known[key]
    return out
