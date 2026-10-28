"""Transparent Dataset Health Score (0-100). See docs/METHODOLOGY.md.

score = sum(weight_d * dim_score_d) / sum(weight_d) over *applicable* dimensions.
Dimensions that cannot be evaluated (e.g. label quality without a class target)
are excluded and their weight is redistributed - they never receive a free 100.
The score is a project-defined heuristic, not a probability or a model-accuracy estimate.
"""
from __future__ import annotations
import math

DEFAULT_WEIGHTS = {"completeness": 0.25, "uniqueness": 0.15, "validity": 0.25, "statistical": 0.15, "label_quality": 0.20}
DEFAULT_THRESHOLDS = [(90, "Excellent"), (75, "Good"), (60, "Needs Improvement"), (40, "Poor"), (0, "Critical")]
MIN_ROWS = 5

METHODOLOGY = {
    "completeness": "70% cell-level: 100*(1-min(1, 2*missing_cell_fraction)); 30% column-level: share of columns not excessively missing.",
    "uniqueness": "100*(1-min(1, 3*duplicate_row_fraction)).",
    "validity": "100*(1 - flagged_columns/columns). Flagged: numeric-as-text, mixed types, date-as-text, inconsistent labels.",
    "statistical": "100*(1-min(1, 3*outlier_cell_fraction)) minus 5/redundant pair (max 25) minus 5/constant column (max 15).",
    "label_quality": "Only with a class target: 100 if imbalance ratio <=1.5, falling log-linearly to 20 at ratio >=20.",
    "note": "Leakage and bias indicators are reported and used in ML-readiness, but not in the score (they cannot be proven from data alone).",
    "thresholds": "Category thresholds are project-defined and configurable.",
}


def categorize(score: float, thresholds=None) -> str:
    for lo, name in sorted(thresholds or DEFAULT_THRESHOLDS, reverse=True):
        if score >= lo:
            return name
    return "Critical"


def _dimension_scores(m: dict) -> dict:
    n, cols = m["n_rows"], max(1, m["n_cols"])
    out: dict = {}
    total = max(1, m["total_cells"])
    cell = 100 * (1 - min(1, 2 * m["missing_cells"] / total))
    col = 100 * (1 - len(m.get("columns_with_excessive_missing", [])) / cols)
    out["completeness"] = (0.7 * cell + 0.3 * col, f"{m['missing_cells']} missing cells of {m['total_cells']}")
    out["uniqueness"] = (100 * (1 - min(1, 3 * m["dup_rows"] / n)), f"{m['dup_rows']} duplicate rows of {n}")
    flagged = len(m.get("validity_flagged_columns", []))
    out["validity"] = (100 * (1 - flagged / cols), f"{flagged} of {cols} columns have type/label problems")
    checked = m.get("numeric_cells_checked", 0)
    frac = m.get("outlier_cells", 0) / checked if checked else 0.0
    s = 100 * (1 - min(1, 3 * frac)) - min(25, 5 * m.get("redundant_pairs", 0)) - min(15, 5 * len(m.get("constant_columns", [])))
    out["statistical"] = (max(0.0, s), f"{m.get('outlier_cells', 0)} outlier values, {m.get('redundant_pairs', 0)} redundant pairs, {len(m.get('constant_columns', []))} constant columns")
    if m.get("target_evaluated") and m.get("imbalance_ratio") is not None:
        r = m["imbalance_ratio"]
        lq = 100.0 if r <= 1.5 else max(20.0, 100 - 80 * math.log(r / 1.5) / math.log(20 / 1.5))
        out["label_quality"] = (lq, f"imbalance ratio {r:.1f}:1")
    return out


def compute_health_score(metrics: dict, issues: list[dict], weights: dict | None = None, thresholds=None) -> dict:
    weights = weights or DEFAULT_WEIGHTS
    base = {"methodology": METHODOLOGY, "thresholds": thresholds or DEFAULT_THRESHOLDS,
            "thresholds_note": "Project-defined categories, not an industry standard."}
    if metrics.get("n_rows", 0) < MIN_ROWS:
        return {**base, "score": None, "category": "Insufficient data", "dimensions": {}, "not_evaluated": [],
                "top_reducers": [], "message": f"At least {MIN_ROWS} rows are required to compute a score."}
    dims = _dimension_scores(metrics)
    not_eval = [{"dimension": d, "reason": "Requires a class target column (classification purpose or categorical target)."}
                for d in weights if d not in dims]
    wsum = sum(weights[d] for d in dims if d in weights)
    result_dims, score = {}, 0.0
    for d, (val, note) in dims.items():
        val = max(0.0, min(100.0, val))
        w = weights.get(d, 0) / wsum
        score += w * val
        result_dims[d] = {"score": round(val, 1), "weight": round(w, 4), "points_lost": round(w * (100 - val), 2), "detail": note}
    reducers = []
    for d, info in sorted(result_dims.items(), key=lambda kv: -kv[1]["points_lost"]):
        if info["points_lost"] > 0:
            reducers.append({"dimension": d, "points_lost": info["points_lost"], "detail": info["detail"],
                             "issue_ids": [i["id"] for i in issues if i.get("dimension") == d]})
    score = round(score, 1)
    return {**base, "score": score, "category": categorize(score, thresholds), "dimensions": result_dims,
            "not_evaluated": not_eval, "top_reducers": reducers}
