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
import os
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np


def cache_dir() -> Path:
    return Path(os.environ.get("YTVIEWS_CACHE_DIR", Path.home() / ".cache" / "ytviews"))


def enabled() -> bool:
    return os.environ.get("YTVIEWS_CACHE", "1").lower() not in ("0", "false", "no", "off")


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def file_key(path: str | Path) -> str | None:
    try:
        return hashlib.sha1(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


class EmbeddingCache:
    def __init__(self, namespace: str, directory: Path | None = None) -> None:
        directory = directory or cache_dir()
        directory.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.conn = sqlite3.connect(directory / "embeddings.sqlite")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS emb (ns TEXT, key TEXT, vec BLOB, PRIMARY KEY (ns, key))"
        )

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

    ``encode`` receives only the unique, uncached items.
    """
    out = np.zeros((len(items), dim), dtype=np.float32)
    if not items:
        return out
    cache = EmbeddingCache(namespace) if enabled() else None
    known = cache.get_many(keys) if cache else {}
    todo = {}
    for key, item in zip(keys, items, strict=True):
        if key not in known and key not in todo:
            todo[key] = item
    if todo:
        fresh = np.asarray(encode(list(todo.values())), dtype=np.float32)
        new = dict(zip(todo, fresh, strict=True))
        known.update(new)
        if cache:
            cache.put_many(new)
    for i, key in enumerate(keys):
        out[i] = known[key]
    return out
