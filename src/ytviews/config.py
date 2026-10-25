"""Runtime settings, read from the environment (and a local .env file if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    data_dir: Path
    model_dir: Path

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ytviews.db"

    @property
    def default_model_path(self) -> Path:
        return self.model_dir / "latest"


def load_settings() -> Settings:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - python-dotenv is a core dependency
        pass
    return Settings(
        api_key=os.environ.get("YOUTUBE_API_KEY") or None,
        data_dir=Path(os.environ.get("YTVIEWS_DATA_DIR", "data")),
        model_dir=Path(os.environ.get("YTVIEWS_MODEL_DIR", "models")),
    )
