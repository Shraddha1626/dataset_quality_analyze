from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import numpy as np
import pandas as pd

SEVERITIES = ["critical", "high", "medium", "low", "info"]
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def to_native(v: Any) -> Any:
    """Convert numpy/pandas values to JSON-safe Python values (NaN/inf -> None)."""
    if isinstance(v, dict):
        return {str(k): to_native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [to_native(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (f != f or f in (float("inf"), float("-inf"))) else f
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if v is pd.NaT:
        return None
    return v


@dataclass
class Issue:
    id: str
    type: str
    category: str
    dimension: str | None
    title: str
    severity: str
    columns: list[str]
    affected_count: int
    affected_pct: float | None
    explanation: str
    impact: str
    recommendation: str
    confidence: str
    repair_available: bool
    repair_op: str | None = None
    repair_params: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    affected_rows: list = field(default_factory=list)
    status: str = "open"

    def to_dict(self) -> dict:
        return to_native(asdict(self))
