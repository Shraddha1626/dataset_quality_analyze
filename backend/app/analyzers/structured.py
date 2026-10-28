"""Structured (CSV/Excel) data-quality analysis.

Every check is computed from the data. Severity rules are documented in
docs/METHODOLOGY.md and implemented in the small `_sev_*` functions below.
Leakage and bias checks report *indicators*, never proof.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .models import Issue, to_native


@dataclass
class AnalysisConfig:
    target: str | None = None
    purpose: str = "exploratory"  # classification | regression | clustering | exploratory
    id_columns: list[str] = field(default_factory=list)
    exclude_columns: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)
    outlier_method: str = "iqr"  # iqr | zscore
    iqr_k: float = 1.5
    z_threshold: float = 3.0
    duplicate_subset: list[str] | None = None
    excessive_missing_pct: float = 40.0
    redundancy_threshold: float = 0.95
    leakage_threshold: float = 0.95
    max_correlation_columns: int = 30
    sample_limit: int = 20


# ---------------------------------------------------------------- helpers
def is_text(s: pd.Series) -> bool:
    return (s.dtype == object) or pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype)


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)


def null_mask(s: pd.Series) -> pd.Series:
    """True where a value is NaN/None or a blank (whitespace-only) string."""
    m = s.isna()
    if is_text(s):
        m = m | s.astype(object).map(lambda v: isinstance(v, str) and v.strip() == "")
    return m


def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().str.replace(",", "", regex=False), errors="coerce")


_DATE_RE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")
_ID_NAME_RE = re.compile(r"(^|_|\b)(id|uuid|guid|index|idx)$", re.I)
_TARGET_NAME_RE = re.compile(r"(^|_)(target|label|outcome|result)($|_)", re.I)


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


def _sample(idx, limit):
    return [to_native(i) for i in list(idx)[:limit]]


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))) if n else 0.0


def _eta(values: pd.Series, groups: pd.Series) -> float:
    d = pd.DataFrame({"v": values, "g": groups}).dropna()
    if d["g"].nunique() < 2 or len(d) < 20:
        return 0.0
    mean = d["v"].mean()
    sst = ((d["v"] - mean) ** 2).sum()
    if sst == 0:
        return 0.0
    gm = d.groupby("g")["v"].transform("mean")
    return float(np.sqrt(((gm - mean) ** 2).sum() / sst))


# ------------------------------------------------------ severity rules
def _sev_missing(pct: float, cfg: AnalysisConfig) -> str:
    if pct >= cfg.excessive_missing_pct:
        return "critical"
    if pct >= cfg.excessive_missing_pct / 2:
        return "high"
    return "medium" if pct >= 5 else "low"


def _sev_duplicates(pct: float) -> str:
    return "high" if pct >= 20 else "medium" if pct >= 5 else "low"


def _sev_outliers(pct: float) -> str:
    return "medium" if pct >= 10 else "low" if pct >= 1 else "info"


def _sev_imbalance(ratio: float) -> str | None:
    if ratio >= 20:
        return "critical"
    if ratio >= 10:
        return "high"
    if ratio >= 3:
        return "medium"
    return "low" if ratio >= 1.5 else None


def _sev_share(pct: float) -> str:
    return "high" if pct >= 20 else "medium" if pct >= 5 else "low"


# ---------------------------------------------------------------- profile
def _profile_numeric(s: pd.Series) -> dict:
    v = s.dropna().astype(float)
    if v.empty:
        return {"kind": "numeric", "count": 0}
    q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
    bins = int(min(20, max(1, v.nunique())))
    counts, edges = np.histogram(v, bins=bins)
    skew = float(v.skew()) if len(v) > 2 else 0.0
    return {
        "kind": "numeric", "count": int(len(v)), "mean": v.mean(), "median": med,
        "std": v.std() if len(v) > 1 else 0.0, "min": v.min(), "max": v.max(), "q1": q1, "q3": q3,
        "unique": int(v.nunique()), "skew": 0.0 if skew != skew else skew,
        "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
    }


def _profile_categorical(s: pd.Series, nulls: pd.Series) -> dict:
    v = s[~nulls].astype(str)
    vc = v.value_counts()
    return {
        "kind": "categorical", "count": int(len(v)), "unique": int(v.nunique()),
        "cardinality_ratio": round(v.nunique() / len(v), 4) if len(v) else 0.0,
        "top_values": [{"value": str(k), "count": int(c)} for k, c in vc.head(10).items()],
    }


# ------------------------------------------------------------------ main
def analyze_structured(df: pd.DataFrame, cfg: AnalysisConfig | None = None) -> dict:
    cfg = cfg or AnalysisConfig()
    unknown = [c for c in ([cfg.target] if cfg.target else []) + cfg.id_columns + cfg.group_columns + cfg.exclude_columns
               if c not in df.columns]
    if unknown:
        raise ValueError(f"Unknown column(s): {', '.join(map(str, unknown))}")
    if cfg.target and cfg.target in cfg.exclude_columns:
        raise ValueError("The target column cannot also be excluded from analysis.")

    work = df.drop(columns=cfg.exclude_columns)
    n_rows, n_cols = work.shape
    issues: list[Issue] = []
    lim = cfg.sample_limit
    metrics: dict = {"n_rows": n_rows, "n_cols": n_cols, "analysis_config": to_native(cfg.__dict__)}

    if n_rows == 0:
        metrics.update(missing_cells=0, total_cells=0, dup_rows=0)
        return {"shape": [0, n_cols], "columns": [], "issues": [], "profile": {}, "metrics": metrics, "correlation": None}

    nulls = work.apply(null_mask)
    id_cols = set(cfg.id_columns)
    target = cfg.target
    supervised = cfg.purpose in ("classification", "regression") and target is not None
    numeric_cols = [c for c in work.columns if is_numeric(work[c])]
    text_cols = [c for c in work.columns if is_text(work[c])]

    # ---- 1. missing values
    miss_per_col = nulls.sum()
    total_missing = int(miss_per_col.sum())
    metrics.update(missing_cells=total_missing, total_cells=int(n_rows * n_cols),
                   missing_per_column={c: int(v) for c, v in miss_per_col.items() if v > 0})
    excessive = []
    for c, cnt in miss_per_col.items():
        if cnt == 0:
            continue
        pct = _pct(int(cnt), n_rows)
        if pct >= cfg.excessive_missing_pct:
            excessive.append(c)
        numeric = c in numeric_cols
        if c == target:
            issues.append(Issue(
                id=f"missing_target_values:{c}", type="missing_target_values", category="Missing values",
                dimension="completeness", title=f"Target column '{c}' has missing values",
                severity="high" if pct >= 1 else "medium", columns=[c], affected_count=int(cnt), affected_pct=pct,
                explanation=f"{int(cnt)} rows ({pct}%) have no target value.",
                impact="Rows without a target cannot be used for supervised training or evaluation.",
                recommendation="Review these rows; usually drop them rather than impute the target.",
                confidence="high", repair_available=True, repair_op="drop_missing",
                repair_params={"axis": "rows", "subset": [c], "max_missing_pct": 0},
                affected_rows=_sample(work.index[nulls[c]], lim)))
            continue
        issues.append(Issue(
            id=f"missing_values:{c}", type="missing_values", category="Missing values", dimension="completeness",
            title=f"Missing values in '{c}'", severity=_sev_missing(pct, cfg), columns=[c],
            affected_count=int(cnt), affected_pct=pct,
            explanation=f"{int(cnt)} of {n_rows} values ({pct}%) are missing (blank strings count as missing).",
            impact="Many models cannot train on missing values; imputing carelessly can bias results.",
            recommendation="Impute (median/mean for numeric, mode or an explicit 'Missing' category for categorical), or drop if mostly empty.",
            confidence="high", repair_available=True, repair_op="fill_missing",
            repair_params={"columns": [c], "strategy": "median" if numeric else "mode"},
            details={"numeric": numeric, "excessive": pct >= cfg.excessive_missing_pct},
            affected_rows=_sample(work.index[nulls[c]], lim)))
    metrics["columns_with_excessive_missing"] = excessive

    # ---- 2. duplicates
    subset = cfg.duplicate_subset
    if subset:
        bad = [c for c in subset if c not in work.columns]
        if bad:
            raise ValueError(f"Unknown duplicate_subset column(s): {', '.join(bad)}")
    try:
        dup_mask = work.duplicated(subset=subset, keep="first")
    except TypeError:  # unhashable cells (lists/dicts)
        dup_mask = work.astype(str).duplicated(subset=subset, keep="first")
    dup_rows = int(dup_mask.sum())
    metrics["dup_rows"] = dup_rows
    if dup_rows:
        pct = _pct(dup_rows, n_rows)
        issues.append(Issue(
            id="duplicate_rows:dataset", type="duplicate_rows", category="Duplicates", dimension="uniqueness",
            title="Duplicate rows" + (f" (on {', '.join(subset)})" if subset else ""),
            severity=_sev_duplicates(pct), columns=list(subset or []), affected_count=dup_rows, affected_pct=pct,
            explanation=f"{dup_rows} rows ({pct}%) repeat an earlier row" + (" on the selected columns." if subset else " exactly."),
            impact="Duplicates over-weight some samples and can leak between train and test splits, inflating evaluation scores.",
            recommendation="Review, then remove duplicates keeping the first occurrence.",
            confidence="high", repair_available=True, repair_op="remove_duplicates",
            repair_params={"subset": subset, "keep": "first"}, affected_rows=_sample(work.index[dup_mask], lim)))
    for c in id_cols & set(work.columns):
        dups = int(work[c].dropna().duplicated().sum())
        if dups:
            issues.append(Issue(
                id=f"duplicate_ids:{c}", type="duplicate_ids", category="Duplicates", dimension=None,
                title=f"Duplicate identifiers in '{c}'", severity="medium", columns=[c], affected_count=dups,
                affected_pct=_pct(dups, n_rows),
                explanation=f"'{c}' was marked as a unique identifier but {dups} values repeat.",
                impact="Repeated IDs may indicate duplicated entities or conflicting records.",
                recommendation="Inspect the repeated IDs; they may be exact duplicates or distinct records sharing an ID.",
                confidence="high", repair_available=False))

    # ---- 3. data types / formats
    validity_cols: set[str] = set()
    for c in text_cols:
        nn = work.loc[~nulls[c], c]
        if len(nn) < 3:
            continue
        py_types = nn.map(lambda v: type(v).__name__).value_counts()
        has_num_objs = any(t in ("int", "float", "int64", "float64") for t in py_types.index)
        has_str_objs = any(t == "str" for t in py_types.index)
        ratio = float(_to_num(nn).notna().mean())
        if ratio >= 0.9:
            bad_n = int(round((1 - ratio) * len(nn)))
            validity_cols.add(c)
            issues.append(Issue(
                id=f"numeric_as_string:{c}", type="numeric_as_string", category="Data types", dimension="validity",
                title=f"Numeric values stored as text in '{c}'", severity="medium", columns=[c],
                affected_count=int(len(nn)), affected_pct=_pct(int(len(nn)), n_rows),
                explanation=f"{ratio:.1%} of non-missing values parse as numbers but the column is text."
                            + (f" {bad_n} value(s) do not parse." if bad_n else ""),
                impact="Text-typed numbers are treated as categories, breaking scaling, statistics and numeric models.",
                recommendation="Convert to numeric after reviewing non-parsable values.",
                confidence="high" if ratio == 1.0 else "medium", repair_available=True, repair_op="convert_dtype",
                repair_params={"column": c, "to": "numeric", "max_failures": bad_n}, details={"parse_ratio": ratio}))
            continue
        if 0.1 <= ratio < 0.9 or (has_num_objs and has_str_objs):
            validity_cols.add(c)
            issues.append(Issue(
                id=f"mixed_types:{c}", type="mixed_types", category="Data types", dimension="validity",
                title=f"Mixed value types in '{c}'", severity="medium", columns=[c],
                affected_count=int((_to_num(nn).notna()).sum()), affected_pct=_pct(int(_to_num(nn).notna().sum()), n_rows),
                explanation=f"About {ratio:.0%} of values look numeric while the rest are non-numeric text.",
                impact="Mixed types usually signal entry errors or sentinel values (e.g. 'N/A', 'unknown').",
                recommendation="Inspect the non-numeric entries; treat sentinels as missing or fix them manually.",
                confidence="medium", repair_available=False, details={"numeric_like_ratio": ratio}))
            continue
        sample = nn.astype(str).head(200)
        if sample.map(lambda v: bool(_DATE_RE.match(v))).mean() >= 0.9:
            sigs = nn.astype(str).map(lambda v: re.sub(r"\d", "d", v.strip())).value_counts()
            validity_cols.add(c)
            issues.append(Issue(
                id=f"date_as_string:{c}", type="date_as_string", category="Data types", dimension="validity",
                title=f"Date-like values stored as text in '{c}'", severity="low" if len(sigs) == 1 else "medium",
                columns=[c], affected_count=int(len(nn)), affected_pct=_pct(int(len(nn)), n_rows),
                explanation="Values look like dates but are stored as text"
                            + (f" in {len(sigs)} different formats." if len(sigs) > 1 else "."),
                impact="Text dates cannot be used for time-based features or ordering.",
                recommendation="Convert to datetime after confirming the date format (day-first vs month-first).",
                confidence="medium", repair_available=True, repair_op="convert_dtype",
                repair_params={"column": c, "to": "datetime", "max_failures": 0},
                details={"format_signatures": {k: int(v) for k, v in sigs.head(5).items()}}))

    # ---- whitespace + inconsistent labels (categorical-like text columns)
    for c in text_cols:
        nn = work.loc[~nulls[c], c]
        if nn.empty or c in validity_cols or c in id_cols:
            continue
        strs = nn.astype(str)
        ws = strs[(strs != strs.str.strip()) | strs.str.contains(r"\s{2,}", regex=True)]
        if len(ws):
            issues.append(Issue(
                id=f"whitespace_inconsistency:{c}", type="whitespace_inconsistency", category="Consistency",
                dimension="validity", title=f"Extra whitespace in '{c}'", severity="low", columns=[c],
                affected_count=int(len(ws)), affected_pct=_pct(int(len(ws)), n_rows),
                explanation=f"{len(ws)} values have leading, trailing or repeated spaces.",
                impact="'Pune' and 'Pune ' are treated as different categories.",
                recommendation="Trim and collapse whitespace.", confidence="high", repair_available=True,
                repair_op="normalize_whitespace", repair_params={"columns": [c]},
                affected_rows=_sample(ws.index, lim)))
        if strs.nunique() > max(50, int(0.05 * n_rows)):
            continue  # free text / high cardinality: label checks not meaningful
        groups: dict[str, list[str]] = {}
        for v in strs.unique():
            groups.setdefault(re.sub(r"[\W_]+", "", v.lower()), []).append(v)
        counts = strs.value_counts()
        variants = {k: vs for k, vs in groups.items() if len(vs) > 1 and k}
        if variants:
            mapping, affected, basic_only = {}, 0, True
            for vs in variants.values():
                dom = max(vs, key=lambda v: counts[v])
                for v in vs:
                    if v != dom:
                        mapping[v] = dom
                        affected += int(counts[v])
                        basic_only &= " ".join(v.lower().split()) == " ".join(dom.lower().split())
            validity_cols.add(c)
            is_t = c == target
            pct = _pct(affected, n_rows)
            issues.append(Issue(
                id=f"inconsistent_labels:{c}", type="inconsistent_labels", category="Consistency", dimension="validity",
                title=f"Inconsistent category labels in '{c}'", severity="high" if is_t else _sev_share(pct),
                columns=[c], affected_count=affected, affected_pct=pct,
                explanation=f"{len(variants)} label group(s) appear under multiple spellings (case, spacing or punctuation).",
                impact="Spelling variants split one class/category into several, distorting class counts and encodings.",
                recommendation="Review the proposed mapping and apply only the merges that are semantically correct.",
                confidence="high" if basic_only else "medium", repair_available=True, repair_op="map_categories",
                repair_params={"column": c, "mapping": mapping},
                details={"proposed_mapping": mapping, "note": "Mappings are suggestions; never auto-applied."}))
        small = [k for k, v in counts.items() if v <= 2 and v / len(strs) < 0.01]
        if small and strs.nunique() <= 30 and len(strs) >= 100:
            issues.append(Issue(
                id=f"rare_categories:{c}", type="rare_categories", category="Consistency", dimension=None,
                title=f"Rare categories in '{c}'", severity="info", columns=[c], affected_count=int(sum(counts[k] for k in small)),
                affected_pct=_pct(int(sum(counts[k] for k in small)), n_rows),
                explanation=f"{len(small)} categories occur at most twice (<1% of rows): {', '.join(map(str, small[:5]))}.",
                impact="Rare categories can be typos or cause unstable encodings and stratified-split failures.",
                recommendation="Check whether they are typos of larger categories.", confidence="low", repair_available=False))

    metrics["validity_flagged_columns"] = sorted(validity_cols)

    # ---- constant columns
    constant = [c for c in work.columns if work.loc[~nulls[c], c].nunique() <= 1 and n_rows > 1 and c not in id_cols]
    metrics["constant_columns"] = constant
    for c in constant:
        issues.append(Issue(
            id=f"constant_column:{c}", type="constant_column", category="Feature quality", dimension="statistical",
            title=f"Constant column '{c}'", severity="low", columns=[c], affected_count=n_rows, affected_pct=100.0,
            explanation="The column has at most one distinct value.", impact="It carries no information for modelling.",
            recommendation="Drop the column.", confidence="high", repair_available=True, repair_op="drop_columns",
            repair_params={"columns": [c]}))

    # ---- 4. outliers
    is_class_target = False
    if target:
        t = work[target]
        is_class_target = (cfg.purpose == "classification") or (cfg.purpose == "exploratory" and (not is_numeric(t) or t.nunique() <= 20))
    outlier_counts, out_cells, num_cells, skipped = {}, 0, 0, []
    for c in numeric_cols:
        if c in id_cols or c in constant or (c == target and (is_class_target or cfg.purpose == "regression")):
            continue
        v = work[c].dropna().astype(float)
        if len(v) < 8:
            skipped.append({"column": c, "reason": "fewer than 8 non-missing values"})
            continue
        if cfg.outlier_method == "zscore":
            sd = v.std()
            if not sd:
                skipped.append({"column": c, "reason": "zero standard deviation"}); continue
            flag = ((v - v.mean()).abs() / sd) > cfg.z_threshold
            lo, hi = v.mean() - cfg.z_threshold * sd, v.mean() + cfg.z_threshold * sd
        else:
            q1, q3 = v.quantile([0.25, 0.75]); iqr = q3 - q1
            if iqr == 0:
                skipped.append({"column": c, "reason": "IQR is zero"}); continue
            lo, hi = q1 - cfg.iqr_k * iqr, q3 + cfg.iqr_k * iqr
            flag = (v < lo) | (v > hi)
        cnt = int(flag.sum())
        num_cells += len(v)
        outlier_counts[c] = cnt
        out_cells += cnt
        if cnt:
            pct = _pct(cnt, len(v))
            issues.append(Issue(
                id=f"outliers:{c}", type="outliers", category="Outliers", dimension="statistical",
                title=f"Potential outliers in '{c}'", severity=_sev_outliers(pct), columns=[c], affected_count=cnt, affected_pct=pct,
                explanation=f"{cnt} values fall outside [{lo:.4g}, {hi:.4g}] ({cfg.outlier_method.upper()} rule).",
                impact="Extreme values can dominate scale-sensitive models; they may also be valid rare events.",
                recommendation="Inspect the records first. Cap or transform only if they are errors.",
                confidence="medium", repair_available=True, repair_op="handle_outliers",
                repair_params={"columns": [c], "method": cfg.outlier_method, "action": "cap"},
                details={"lower_bound": lo, "upper_bound": hi, "note": "An outlier is not necessarily an error."},
                affected_rows=_sample(v.index[flag], lim)))
    metrics.update(outlier_counts=outlier_counts, outlier_cells=out_cells, numeric_cells_checked=num_cells,
                   outlier_checks_skipped=skipped)

    # ---- 5. class imbalance
    metrics["target_evaluated"] = False
    metrics["class_distribution"] = None
    if target and is_class_target:
        tv = work.loc[~nulls[target], target].astype(str)
        if tv.nunique() >= 2:
            dist = tv.value_counts()
            ratio = float(dist.iloc[0] / dist.iloc[-1])
            metrics.update(target_evaluated=True, class_distribution={k: int(v) for k, v in dist.items()},
                           imbalance_ratio=ratio, majority_class=str(dist.index[0]), minority_class=str(dist.index[-1]))
            sev = _sev_imbalance(ratio)
            if sev:
                issues.append(Issue(
                    id=f"class_imbalance:{target}", type="class_imbalance", category="Class balance", dimension="label_quality",
                    title=f"Class imbalance in '{target}'", severity=sev, columns=[target],
                    affected_count=int(dist.iloc[-1]), affected_pct=_pct(int(dist.iloc[-1]), len(tv)),
                    explanation=f"Majority '{dist.index[0]}' has {int(dist.iloc[0])} rows vs minority '{dist.index[-1]}' with {int(dist.iloc[-1])} (ratio {ratio:.1f}:1).",
                    impact="Models can reach high accuracy by favouring the majority class while missing the minority.",
                    recommendation="Prefer class weights or stratified evaluation; consider resampling on the training split only.",
                    confidence="high", repair_available=True, repair_op="balance_classes",
                    repair_params={"target": target, "method": "random_oversample"},
                    details={"class_distribution": {k: int(v) for k, v in dist.items()}, "imbalance_ratio": ratio}))
        elif len(tv):
            issues.append(Issue(
                id=f"single_class_target:{target}", type="single_class_target", category="Class balance", dimension="label_quality",
                title=f"Target '{target}' has a single class", severity="critical", columns=[target], affected_count=len(tv),
                affected_pct=100.0, explanation="Only one distinct target value exists.", impact="Supervised learning is not possible.",
                recommendation="Check that the correct column is the target.", confidence="high", repair_available=False))

    # ---- 6/7. correlation
    corr_info, pairs = None, []
    corr_cols = [c for c in numeric_cols if c not in id_cols and c not in constant and not (c == target and is_class_target)]
    if len(corr_cols) >= 2:
        cc = corr_cols[: cfg.max_correlation_columns]
        cm = work[cc].astype(float).corr()
        corr_info = {"columns": cc, "matrix": to_native(cm.values.tolist()), "truncated": len(corr_cols) > len(cc)}
        for i, a in enumerate(cc):
            for b in cc[i + 1:]:
                r = cm.loc[a, b]
                if r == r and abs(r) >= cfg.redundancy_threshold and not (target and target in (a, b)):
                    pairs.append({"a": a, "b": b, "r": float(r)})
    metrics["redundant_pairs"] = len(pairs)
    if pairs:
        mx = max(abs(p["r"]) for p in pairs)
        issues.append(Issue(
            id="high_correlation:dataset", type="high_correlation", category="Feature quality", dimension="statistical",
            title="Highly correlated (potentially redundant) features", severity="medium" if mx >= 0.99 else "low",
            columns=sorted({p["a"] for p in pairs} | {p["b"] for p in pairs}), affected_count=len(pairs), affected_pct=None,
            explanation=f"{len(pairs)} feature pair(s) have |r| ≥ {cfg.redundancy_threshold}.",
            impact="Redundant features inflate dimensionality and destabilise linear-model coefficients.",
            recommendation="Keep one feature per correlated pair after checking they really measure the same thing.",
            confidence="medium", repair_available=False, details={"pairs": pairs[:20]}))

    # ---- 8. leakage indicators (never proof)
    if target and cfg.purpose != "clustering":
        t = work[target]
        t_codes = t.astype(str).factorize()[0] if (is_class_target or not is_numeric(t)) else None
        for c in work.columns:
            if c == target or c in id_cols:
                continue
            reasons, conf = [], "low"
            if _TARGET_NAME_RE.search(c) or target.lower() in c.lower():
                reasons.append("name resembles the target/outcome")
            s = work[c]
            ok = ~nulls[c] & ~nulls[target]
            if ok.sum() >= 20:
                score = 0.0
                if is_numeric(s) and c not in constant:
                    if is_class_target:
                        score = _eta(s[ok].astype(float), t[ok].astype(str))
                    elif is_numeric(t):
                        score = abs(float(np.corrcoef(s[ok].astype(float), t[ok].astype(float))[0, 1])) if s[ok].std() and t[ok].std() else 0.0
                elif is_text(s) and s[ok].nunique() <= 20 and ok.sum() / max(1, s[ok].nunique()) >= 5 and is_class_target:
                    score = _cramers_v(s[ok].astype(str), t[ok].astype(str))
                if score == score and score >= cfg.leakage_threshold:
                    reasons.append(f"very strong association with target ({score:.3f})")
                    conf = "medium"
            if reasons:
                issues.append(Issue(
                    id=f"potential_leakage:{c}", type="potential_leakage", category="Leakage indicators", dimension=None,
                    title=f"Possible target leakage via '{c}'", severity="high" if conf == "medium" else "medium", columns=[c],
                    affected_count=n_rows, affected_pct=100.0,
                    explanation="Indicator(s): " + "; ".join(reasons) + ". This is a warning, not proof of leakage.",
                    impact="If the feature is only known after the outcome, models look excellent in testing and fail in production.",
                    recommendation="Check with domain knowledge whether this value is available at prediction time; drop it if not.",
                    confidence=conf, repair_available=True, repair_op="drop_columns", repair_params={"columns": [c]},
                    details={"reasons": reasons, "limitation": "Correlation alone cannot establish leakage."}))
    flagged_leak = {i.columns[0] for i in issues if i.type == "potential_leakage"}
    for c in work.columns:
        if c in id_cols or c == target or c in validity_cols or c in flagged_leak:
            continue
        s, nn = work[c], work.loc[~nulls[c], c]
        if len(nn) < 20 or nn.nunique() / len(nn) < 0.98:
            continue
        name_hit = bool(_ID_NAME_RE.search(c))
        int_like = pd.api.types.is_integer_dtype(s.dtype)
        short_text = is_text(s) and nn.astype(str).str.len().max() <= 40
        if name_hit or int_like or short_text:
            issues.append(Issue(
                id=f"possible_identifier:{c}", type="possible_identifier", category="Leakage indicators", dimension=None,
                title=f"'{c}' looks like an identifier", severity="medium" if name_hit else "low", columns=[c],
                affected_count=n_rows, affected_pct=100.0,
                explanation="Nearly every value is unique, which is typical of row identifiers.",
                impact="Identifiers can be memorised by flexible models and carry no generalisable signal.",
                recommendation="Exclude it from the feature set (mark it as an identifier column).",
                confidence="medium" if name_hit else "low", repair_available=True, repair_op="drop_columns",
                repair_params={"columns": [c]}))

    # ---- 9. bias indicators (limited, indicative)
    if target and is_class_target:
        t = work[target].astype(str)
        sizes = t[~nulls[target]].value_counts()
        big = [k for k, v in sizes.items() if v >= 20]
        if len(big) >= 2:
            for c, cnt in miss_per_col.items():
                if cnt == 0 or c == target:
                    continue
                rates = {k: float(nulls.loc[(t == k) & ~nulls[target], c].mean()) for k in big}
                if max(rates.values()) - min(rates.values()) >= 0.2:
                    issues.append(Issue(
                        id=f"missingness_disparity:{c}", type="missingness_disparity", category="Bias indicators", dimension=None,
                        title=f"Missingness in '{c}' differs across target classes", severity="medium", columns=[c],
                        affected_count=int(cnt), affected_pct=_pct(int(cnt), n_rows),
                        explanation="Missing rate by class: " + ", ".join(f"{k}: {v:.0%}" for k, v in rates.items()) + ".",
                        impact="If missingness is related to the label, imputation or row-dropping can bias the model.",
                        recommendation="Investigate why values are missing before imputing; consider a missing-indicator feature.",
                        confidence="low", repair_available=False,
                        details={"rates": rates, "limitation": "Only compares groups with ≥20 rows; does not prove bias or fairness."}))
        majority = sizes.index[0] if len(sizes) else None
        for g in cfg.group_columns:
            if g == target or majority is None:
                continue
            sub = pd.DataFrame({"g": work[g].astype(str), "t": t})[~nulls[g] & ~nulls[target]]
            share = {k: float((v["t"] == majority).mean()) for k, v in sub.groupby("g") if len(v) >= 30}
            if len(share) >= 2 and max(share.values()) - min(share.values()) >= 0.2:
                issues.append(Issue(
                    id=f"group_distribution_disparity:{g}", type="group_distribution_disparity", category="Bias indicators",
                    dimension=None, title=f"Class distribution differs across groups of '{g}'", severity="medium", columns=[g, target],
                    affected_count=int(len(sub)), affected_pct=_pct(int(len(sub)), n_rows),
                    explanation=f"Share of class '{majority}' by group: " + ", ".join(f"{k}: {v:.0%}" for k, v in share.items()) + ".",
                    impact="Large differences may encode sampling bias or real differences; a model can inherit them.",
                    recommendation="Review how the data was collected for each group and evaluate the model per group.",
                    confidence="low", repair_available=False,
                    details={"shares": share, "limitation": "A difference is an indicator only. Absence of a flag does not mean the data is fair."}))

    # ---- 10. profiling
    profile = {}
    for c in work.columns:
        if c in numeric_cols:
            p = _profile_numeric(work[c])
        else:
            p = _profile_categorical(work[c], nulls[c])
        p.update(dtype=str(work[c].dtype), missing=int(miss_per_col[c]), missing_pct=_pct(int(miss_per_col[c]), n_rows))
        profile[c] = to_native(p)

    columns = [{"name": c, "dtype": str(work[c].dtype), "missing": int(miss_per_col[c]),
                "unique": int(work[c].nunique(dropna=True)), "role": "target" if c == target else "id" if c in id_cols else "feature"}
               for c in work.columns]
    return {"shape": [n_rows, n_cols], "columns": columns, "issues": issues, "profile": profile,
            "metrics": to_native(metrics), "correlation": corr_info}
