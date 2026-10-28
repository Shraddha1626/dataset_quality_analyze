"""Orchestrates analysis -> score -> recommendations -> readiness for structured data."""
from __future__ import annotations
import pandas as pd

from app.analyzers.structured import AnalysisConfig, analyze_structured
from app.analyzers.models import SEVERITY_RANK, to_native
from app.recommendations.engine import build_recommendations
from app.scoring.health import compute_health_score
from app.scoring.readiness import assess_readiness


def run_structured_analysis(df: pd.DataFrame, cfg: AnalysisConfig | None = None, weights: dict | None = None) -> dict:
    cfg = cfg or AnalysisConfig()
    raw = analyze_structured(df, cfg)
    issues = sorted((i.to_dict() for i in raw["issues"]), key=lambda i: (SEVERITY_RANK[i["severity"]], i["id"]))
    score = compute_health_score(raw["metrics"], issues, weights)
    recs = build_recommendations(issues, raw["profile"], raw["metrics"])
    readiness = assess_readiness(raw["metrics"], issues, cfg.purpose, cfg.target)
    sev_counts = {s: sum(1 for i in issues if i["severity"] == s) for s in SEVERITY_RANK}
    return to_native({
        "dataset_type": "structured", "shape": raw["shape"], "columns": raw["columns"], "profile": raw["profile"],
        "correlation": raw["correlation"], "metrics": raw["metrics"], "issues": issues, "issue_counts": sev_counts,
        "health_score": score, "recommendations": recs, "ml_readiness": readiness,
        "limitations": ["Outliers are statistical flags, not confirmed errors.",
                        "Leakage and bias checks are indicators only and can miss real problems.",
                        "The health score is a project-defined heuristic, not a predictor of model accuracy."],
    })


def compare_results(before: dict, after: dict, repair_log: list[dict]) -> dict:
    mb, ma = before["metrics"], after["metrics"]
    sb, sa = before["health_score"]["score"], after["health_score"]["score"]
    return to_native({
        "health_score": {"before": sb, "after": sa, "delta": None if sb is None or sa is None else round(sa - sb, 1),
                         "category_before": before["health_score"]["category"], "category_after": after["health_score"]["category"]},
        "rows": {"before": mb["n_rows"], "after": ma["n_rows"]},
        "missing_cells": {"before": mb["missing_cells"], "after": ma["missing_cells"]},
        "duplicate_rows": {"before": mb["dup_rows"], "after": ma["dup_rows"]},
        "outlier_cells": {"before": mb.get("outlier_cells", 0), "after": ma.get("outlier_cells", 0)},
        "class_distribution": {"before": mb.get("class_distribution"), "after": ma.get("class_distribution")},
        "records_modified_cells": sum(s["cells_modified"] for s in repair_log),
        "records_removed": sum(s["rows_removed"] for s in repair_log),
        "records_added": sum(s["rows_added"] for s in repair_log),
        "issues": {"before": len(before["issues"]), "after": len(after["issues"]),
                   "resolved_ids": sorted(set(i["id"] for i in before["issues"]) - set(i["id"] for i in after["issues"])),
                   "new_ids": sorted(set(i["id"] for i in after["issues"]) - set(i["id"] for i in before["issues"]))},
        "operations": [{"step": s["step"], "operation": s["operation"], "params": s["params"]} for s in repair_log],
        "note": "These are data-quality metrics only. They do not show that a downstream model's accuracy has improved.",
    })
