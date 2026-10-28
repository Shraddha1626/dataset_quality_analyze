"""Safe tabular loading with size/row/column limits. Never modifies the source file."""
from __future__ import annotations
import os
import pandas as pd
from app.core.config import Settings

SUPPORTED_TABULAR = {".csv", ".tsv", ".xlsx", ".xlsm"}


class DatasetLoadError(ValueError):
    """User-facing loading error (safe to show; contains no server paths)."""


def load_table(path: str, settings: Settings | None = None, sheet: str | int = 0) -> pd.DataFrame:
    s = settings or Settings.from_env()
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_TABULAR:
        raise DatasetLoadError(f"Unsupported tabular format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_TABULAR))}.")
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > s.max_upload_size_mb:
        raise DatasetLoadError(f"File is {size_mb:.1f} MB; the configured limit is {s.max_upload_size_mb} MB.")
    try:
        if ext in (".csv", ".tsv"):
            sep = "\t" if ext == ".tsv" else ","
            df, last = None, None
            for enc in ("utf-8", "utf-8-sig", "latin-1"):
                try:
                    df = pd.read_csv(path, sep=sep, encoding=enc, nrows=s.max_rows + 1)
                    break
                except UnicodeDecodeError as e:
                    last = e
            if df is None:
                raise DatasetLoadError(f"Could not decode the file as text ({last.__class__.__name__}).")
        else:
            df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl", nrows=s.max_rows + 1)
    except DatasetLoadError:
        raise
    except pd.errors.EmptyDataError:
        raise DatasetLoadError("The file is empty.")
    except Exception as e:  # parser errors: do not leak internals
        raise DatasetLoadError(f"The file could not be parsed as {ext[1:].upper()} ({e.__class__.__name__}).")
    if len(df) > s.max_rows:
        raise DatasetLoadError(f"Dataset exceeds the configured limit of {s.max_rows:,} rows.")
    if df.shape[1] > s.max_columns:
        raise DatasetLoadError(f"Dataset exceeds the configured limit of {s.max_columns} columns.")
    df.columns = [str(c) for c in df.columns]
    return df
