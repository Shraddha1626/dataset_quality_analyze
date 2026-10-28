"""Runtime settings, read from environment variables (see .env.example)."""
from __future__ import annotations
import os
from dataclasses import dataclass


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    max_upload_size_mb: int = 50
    max_rows: int = 500_000
    max_columns: int = 500
    preview_rows: int = 20
    temp_dir: str = "/tmp/dqa"
    storage_backend: str = "local"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            max_upload_size_mb=_int("MAX_UPLOAD_SIZE_MB", 50),
            max_rows=_int("MAX_ROWS", 500_000),
            max_columns=_int("MAX_COLUMNS", 500),
            preview_rows=_int("PREVIEW_ROWS", 20),
            temp_dir=os.environ.get("TEMP_DIR", "/tmp/dqa"),
            storage_backend=os.environ.get("STORAGE_BACKEND", "local"),
        )
