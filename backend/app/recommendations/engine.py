"""Transparent rule-based recommendation engine (no external AI service required)."""
from __future__ import annotations
from app.analyzers.models import SEVERITY_RANK

_SEV_PRIORITY = {s: r for s, r in SEVERITY_RANK.items()}


def _rec(issue, action, op, params, why_helps, trade_offs, manual, why_matters=None, alternatives=None, note=""):
    return {
        "issue_id": issue["id"], "title": issue["title"], "severity": issue["severity"], "priority": _SEV_PRIORITY[issue["severity"]],
        "detected": issue["explanation"], "why_it_matters": why_matters or issue["impact"], "recommended_action": action,
        "repair_op": op, "suggested_params": params, "why_it_helps": why_helps, "trade_offs": trade_offs,
        "manual_review_recommended": manual, "manual_review_note": note, "alternatives": alternatives or [],
        "auto_apply": False,
    }


def recommend_for_issue(issue: dict, profile: dict, metrics: dict) -> dict | None:
    t, cols, p = issue["type"], issue["columns"], issue.get("repair_params", {})
    c = cols[0] if cols else None
    if t == "missing_values":
        if issue["details"].get("excessive"):
            return _rec(issue, f"Review whether '{c}' is usable; consider dropping the column.", "drop_columns", {"columns": [c]},
                        "Removes a column that is mostly empty and unlikely to be informative.",
                        "All information in the column is lost.", True, alternatives=[{"label": "Impute anyway", "op": "fill_missing", "params": p}],
                        note="Imputing a mostly-empty column invents most of its values.")
        if issue["details"]["numeric"]:
            skew = (profile.get(c) or {}).get("skew") or 0
            strat = "median" if abs(skew) >= 0.5 else "mean"
            return _rec(issue, f"Fill missing values in '{c}' with the {strat}.", "fill_missing", {"columns": [c], "strategy": strat},
                        f"The {strat} keeps the column usable without dropping rows" + (" and is robust to the skew in this column." if strat == "median" else "."),
                        "Reduces variance and can bias results if values are not missing at random. For ML, compute the statistic on the training split only.",
                        False, alternatives=[{"label": "Median", "op": "fill_missing", "params": {"columns": [c], "strategy": "median"}},
                                             {"label": "Mean", "op": "fill_missing", "params": {"columns": [c], "strategy": "mean"}},
                                             {"label": "Drop affected rows", "op": "drop_missing", "params": {"axis": "rows", "subset": [c], "max_missing_pct": 0}}])
        return _rec(issue, f"Fill missing values in '{c}' with the most frequent value, or an explicit 'Missing' category.", "fill_missing",
                    {"columns": [c], "strategy": "missing_category"},
                    "An explicit category preserves the fact that the value was absent; the mode keeps the distribution unchanged.",
                    "The mode can over-represent the majority category; a 'Missing' category adds a new level.", False,
                    alternatives=[{"label": "Mode", "op": "fill_missing", "params": {"columns": [c], "strategy": "mode"}}])
    if t == "missing_target_values":
        return _rec(issue, "Drop rows with a missing target.", "drop_missing", p, "Only labelled rows can train or evaluate a supervised model.",
                    "Rows are removed; check this does not skew class proportions.", True)
    if t == "duplicate_rows":
        return _rec(issue, "Remove exact duplicate rows, keeping the first occurrence.", "remove_duplicates", p,
                    "Prevents repeated samples from being over-weighted or appearing in both train and test sets.",
                    "Rows are permanently removed from the repaired copy; legitimate repeated observations would be lost.", True,
                    note="Confirm duplicates are errors and not genuine repeated measurements.")
    if t in ("numeric_as_string", "date_as_string"):
        to = p["to"]
        return _rec(issue, f"Convert '{c}' to {to}.", "convert_dtype", p, "Restores correct statistics, ordering and encodings.",
                    "Values that fail conversion are reported; conversion is refused if failures exceed the allowed count.", t == "date_as_string",
                    note="Confirm the date format (day-first vs month-first)." if t == "date_as_string" else "")
    if t == "mixed_types":
        return _rec(issue, f"Inspect non-numeric entries in '{c}' and decide how to treat sentinels such as 'N/A'.", None, {},
                    "Resolving sentinels lets the column be typed correctly.", "Requires a manual decision.", True)
    if t == "whitespace_inconsistency":
        return _rec(issue, f"Trim and collapse whitespace in '{c}'.", "normalize_whitespace", p,
                    "Makes identical values compare equal.", "Only spacing changes; semantic content is untouched.", False)
    if t == "inconsistent_labels":
        return _rec(issue, f"Apply the proposed label normalisation to '{c}' after review.", "map_categories", p,
                    "Merges spelling variants of the same category so classes are counted correctly.",
                    "A wrong merge silently combines different categories.", True,
                    note="Edit the mapping; remove any pair that is not truly the same category.")
    if t == "outliers":
        return _rec(issue, f"Inspect extreme values in '{c}'; cap them only if they are errors.", "handle_outliers", p,
                    "Capping limits the influence of extreme values on scale-sensitive models.",
                    "Capping alters real values; removal deletes rows. Valid rare events may be distorted.", True,
                    alternatives=[{"label": "Remove rows", "op": "handle_outliers", "params": {**p, "action": "remove"}}],
                    note="An outlier is not necessarily an error.")
    if t == "class_imbalance":
        all_num = all(((profile.get(k) or {}).get("kind") == "numeric") for k in profile if k != issue["columns"][0]) and not metrics.get("missing_cells")
        minority = min((metrics.get("class_distribution") or {"": 0}).values())
        smote_ok = all_num and minority >= 6
        alts = [{"label": "Random undersampling", "op": "balance_classes", "params": {"target": c, "method": "random_undersample"}}]
        if smote_ok:
            alts.append({"label": "SMOTE (numeric features only)", "op": "balance_classes", "params": {"target": c, "method": "smote"}})
        return _rec(issue, "Prefer class weights in the model. If resampling, apply it to the training split only.", "balance_classes", p,
                    "Gives the minority class more influence during training.",
                    "Oversampling duplicates rows (overfitting risk); undersampling discards data; SMOTE creates synthetic rows and is unsuitable for categorical, grouped or time-series data. Resampling before a train/test split leaks information.",
                    True, alternatives=alts,
                    note="SMOTE offered only when all features are numeric, complete and the minority class is large enough." if smote_ok else
                         "SMOTE is not offered: it needs all-numeric, complete features and a larger minority class.")
    if t == "constant_column":
        return _rec(issue, f"Drop constant column '{c}'.", "drop_columns", p, "Removes a feature that cannot help a model.", "None, apart from losing the column.", False)
    if t == "high_correlation":
        return _rec(issue, "Review correlated pairs and keep one feature per pair.", None, {}, "Reduces redundancy and multicollinearity.",
                    "Choosing which feature to keep needs domain judgement.", True)
    if t in ("potential_leakage", "possible_identifier"):
        return _rec(issue, f"Check '{c}' against domain knowledge; drop it if it is not available at prediction time.", "drop_columns", p,
                    "Removes a feature that may let the model 'cheat' or memorise rows.", "Dropping a legitimate strong predictor lowers model quality.", True,
                    note="Correlation alone does not prove leakage.")
    if t in ("missingness_disparity", "group_distribution_disparity"):
        return _rec(issue, "Investigate how the data was collected before correcting anything.", None, {},
                    "Understanding the cause avoids introducing bias through naive imputation or filtering.", "No automatic fix is offered.", True)
    if t in ("duplicate_ids", "rare_categories", "single_class_target"):
        return _rec(issue, "Manual review required.", None, {}, "Needs human judgement.", "No automatic fix is offered.", True)
    return None


def build_recommendations(issues: list[dict], profile: dict, metrics: dict) -> list[dict]:
    recs = [r for i in issues if (r := recommend_for_issue(i, profile, metrics))]
    return sorted(recs, key=lambda r: (r["priority"], r["issue_id"]))
