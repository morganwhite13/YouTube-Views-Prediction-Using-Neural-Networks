"""Saved drafts for the web app: a video's inputs under a name, in one small JSON file.

The file is ``$YTVIEWS_DRAFTS`` or ``~/.ytviews/drafts.json``. Each draft is the form's
fields as plain values (title, description, tags, category, length, channel numbers...), so the
file stays readable and survives app and model changes.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MAX_DRAFTS = 200
MAX_NAME = 80
# The web app serves every browser session from threads of one process: saves and deletes
# (read, change, write the whole file) must not interleave.
_LOCK = threading.Lock()


def drafts_path() -> Path:
    return Path(os.environ.get("YTVIEWS_DRAFTS") or Path.home() / ".ytviews" / "drafts.json")


def load_drafts(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Name -> fields, newest first. A missing or unreadable file is no drafts."""
    path = path or drafts_path()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return _newest_first({str(k): v for k, v in data.items() if isinstance(v, dict)})


def save_draft(name: str, fields: dict[str, Any], path: Path | None = None) -> str:
    """Save (or overwrite) a draft; returns the name it was saved under."""
    path = path or drafts_path()
    name = " ".join(str(name).split()) or " ".join(str(fields.get("title") or "Untitled").split())
    name = name[:MAX_NAME]
    with _LOCK:
        drafts = load_drafts(path)
        drafts[name] = {**fields, "saved_at": datetime.now(timezone.utc).isoformat()}
        _write(path, dict(list(_newest_first(drafts).items())[:MAX_DRAFTS]))
    return name


def delete_draft(name: str, path: Path | None = None) -> bool:
    path = path or drafts_path()
    with _LOCK:
        drafts = load_drafts(path)
        if name not in drafts:
            return False
        del drafts[name]
        _write(path, drafts)
    return True


def _newest_first(drafts: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return dict(sorted(drafts.items(), key=lambda kv: str(kv[1].get("saved_at", "")), reverse=True))


def _write(path: Path, drafts: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".drafts-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(drafts, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)  # never leave a half-written file
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def range_sentence(
    low: str,
    high: str,
    days: float,
    unit: str,
    coverage: float | None = None,
    nominal: float = 0.8,
) -> str:
    """What a predicted range means, in plain words (``low``/``high`` already formatted).

    ``nominal`` is the share of outcomes the range is built to contain (0.8 for p10-p90);
    ``coverage`` is how often it actually did on held-out videos, when known.
    """
    text = (
        f"Most videos like this ({round(nominal * 10):g} in 10) end up between **{low}** and "
        f"**{high}** {unit} after {days:g} days; the middle estimate is the single most likely "
        "value."
    )
    if coverage is not None:
        text += (
            " On videos held out from training, the true count landed inside the range "
            f"{coverage:.0%} of the time."
        )
    return text
