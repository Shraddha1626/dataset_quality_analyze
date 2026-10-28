"""Structured-data repair engine.

* The original DataFrame is never mutated (RepairSession works on copies).
* Every applied step is logged with its parameters; `revert` replays the remaining
  steps from the original, so repairs are reproducible.
* Destructive operations require explicit confirmation.
"""
from __future__ import annotations
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app.analyzers.models import to_native
from app.analyzers.structured import is_numeric, is_text, null_mask


class RepairError(ValueError):
    """User-facing repair error."""


def _cols(df, cols):
    if not cols:
        raise RepairError("No columns were specified.")
    bad = [c for c in cols if c not in df.columns]
    if bad:
        raise RepairError(f"Unknown column(s): {', '.join(map(str, bad))}")
    return list(cols)


def _blank_to_nan(s: pd.Series) -> pd.Series:
    return s.mask(null_mask(s)) if is_text(s) else s


# Each op: (df, params, seed) -> (new_df, extra_info)
def op_remove_duplicates(df, p, seed):
    subset = p.get("subset") or None
    if subset:
        _cols(df, subset)
    try:
        out = df[~df.duplicated(subset=subset, keep=p.get("keep", "first"))]
    except TypeError:
        out = df[~df.astype(str).duplicated(subset=subset, keep=p.get("keep", "first"))]
    return out, {}


def op_fill_missing(df, p, seed):
    cols, strat = _cols(df, p.get("columns")), p.get("strategy")
    out, warns = df.copy(), ["Statistics are computed on the whole dataset; for ML, fit them on the training split only."]
    for c in cols:
        s = _blank_to_nan(out[c])
        if strat in ("mean", "median"):
            if not is_numeric(s):
                raise RepairError(f"'{c}' is not numeric; use 'mode' or 'missing_category'.")
            if s.notna().sum() == 0:
                raise RepairError(f"'{c}' has no non-missing values to compute a {strat} from.")
            val = s.mean() if strat == "mean" else s.median()
        elif strat == "mode":
            m = s.dropna().mode()
            if m.empty:
                raise RepairError(f"'{c}' has no non-missing values.")
            val = m.iloc[0]
        elif strat == "missing_category":
            val = p.get("constant", "Missing")
        elif strat == "constant":
            if "constant" not in p:
                raise RepairError("Strategy 'constant' needs a 'constant' value.")
            val = p["constant"]
        else:
            raise RepairError("strategy must be one of: mean, median, mode, constant, missing_category.")
        if is_numeric(s) and not pd.api.types.is_float_dtype(s.dtype) and isinstance(val, float) and val != int(val):
            s = s.astype(float)
        out[c] = s.fillna(val)
    return out, {"warnings": warns}


def op_drop_missing(df, p, seed):
    axis, thr = p.get("axis", "rows"), float(p.get("max_missing_pct", 50))
    if axis == "rows":
        sub = _cols(df, p["subset"]) if p.get("subset") else list(df.columns)
        pct = df[sub].apply(null_mask).mean(axis=1) * 100
        return df[pct <= thr] if thr == 0 else df[pct < thr], {}
    if axis == "columns":
        pct = df.apply(null_mask).mean() * 100
        drop = [c for c in df.columns if pct[c] > thr]
        return df.drop(columns=drop), {"dropped_columns": drop}
    raise RepairError("axis must be 'rows' or 'columns'.")


def op_drop_columns(df, p, seed):
    cols = _cols(df, p.get("columns"))
    if len(cols) >= df.shape[1]:
        raise RepairError("Cannot drop every column.")
    return df.drop(columns=cols), {"dropped_columns": cols}


def op_normalize_whitespace(df, p, seed):
    cols = _cols(df, p["columns"]) if p.get("columns") else [c for c in df.columns if is_text(df[c])]
    out = df.copy()
    for c in cols:
        s = out[c]
        if not is_text(s):
            raise RepairError(f"'{c}' is not a text column.")
        s = s.astype(object).map(lambda v: " ".join(v.split()) if isinstance(v, str) else v)
        out[c] = s.mask(s.map(lambda v: isinstance(v, str) and v == "")) if p.get("blank_to_nan", True) else s
    return out, {}


def op_map_categories(df, p, seed):
    col, mapping = _cols(df, [p.get("column")])[0], p.get("mapping")
    if not isinstance(mapping, dict) or not mapping:
        raise RepairError("A non-empty 'mapping' of old -> new values is required (it must be user-approved).")
    out = df.copy()
    present = set(out[col].dropna().astype(str))
    missing = [k for k in mapping if str(k) not in present]
    out[col] = out[col].astype(object).map(lambda v: mapping.get(v, v) if isinstance(v, str) else v)
    return out, {"warnings": [f"Mapping keys not found in data: {missing[:5]}"] if missing else []}


def op_convert_dtype(df, p, seed):
    col, to = _cols(df, [p.get("column")])[0], p.get("to")
    s = _blank_to_nan(df[col])
    before_null = s.isna()
    if to == "numeric":
        conv = pd.to_numeric(s.astype(str).str.strip().str.replace(",", "", regex=False).where(~before_null), errors="coerce")
    elif to == "datetime":
        conv = pd.to_datetime(s.where(~before_null), errors="coerce", format="mixed", dayfirst=bool(p.get("dayfirst", False)))
    elif to == "string":
        conv = s.astype(object).where(before_null, s.astype(str))
    elif to == "category":
        conv = s.astype("category")
    else:
        raise RepairError("'to' must be one of: numeric, datetime, string, category.")
    failed = conv.isna() & ~before_null
    allowed = int(p.get("max_failures", 0))
    if failed.sum() > allowed:
        ex = df.loc[failed, col].astype(str).head(5).tolist()
        raise RepairError(f"{int(failed.sum())} value(s) in '{col}' cannot be converted to {to} (e.g. {ex}); allowed {allowed}. "
                          "Fix them first or raise max_failures to coerce them to missing.")
    if to == "numeric" and conv.dropna().apply(float.is_integer).all() and conv.notna().all():
        conv = conv.astype("int64")
    out = df.copy()
    out[col] = conv
    return out, {"cells_modified": int((~before_null).sum()), "coerced_to_missing": int(failed.sum())}


def _bounds(v: pd.Series, method, k, z):
    if method == "zscore":
        sd = v.std()
        return (v.mean() - z * sd, v.mean() + z * sd) if sd else (-np.inf, np.inf)
    q1, q3 = v.quantile([0.25, 0.75]); iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr) if iqr else (-np.inf, np.inf)


def op_handle_outliers(df, p, seed):
    cols, method, action = _cols(df, p.get("columns")), p.get("method", "iqr"), p.get("action", "cap")
    if method not in ("iqr", "zscore") or action not in ("cap", "remove"):
        raise RepairError("method must be iqr|zscore and action cap|remove.")
    out, drop = df.copy(), pd.Series(False, index=df.index)
    for c in cols:
        if not is_numeric(out[c]):
            raise RepairError(f"'{c}' is not numeric.")
        lo, hi = _bounds(out[c].dropna().astype(float), method, float(p.get("iqr_k", 1.5)), float(p.get("z_threshold", 3.0)))
        if action == "cap":
            out[c] = out[c].astype(float).clip(lo, hi) if out[c].dtype.kind == "f" else out[c].clip(np.ceil(lo) if np.isfinite(lo) else None, np.floor(hi) if np.isfinite(hi) else None)
        else:
            drop |= (out[c] < lo) | (out[c] > hi)
    return out[~drop], {}


def op_balance_classes(df, p, seed):
    tgt, method = _cols(df, [p.get("target")])[0], p.get("method", "random_oversample")
    if df[tgt].isna().any():
        raise RepairError("The target has missing values; handle them before balancing.")
    counts = df[tgt].value_counts()
    if len(counts) < 2:
        raise RepairError("Balancing needs at least two classes.")
    warns = ["Apply resampling to the TRAINING split only; resampling before splitting leaks information into evaluation."]
    rng = np.random.default_rng(seed)
    if method in ("random_oversample", "random_undersample"):
        goal = counts.max() if method == "random_oversample" else counts.min()
        parts = []
        for cls, grp in df.groupby(tgt, sort=False):
            if method == "random_oversample" and len(grp) < goal:
                extra = grp.iloc[rng.choice(len(grp), goal - len(grp), replace=True)]
                parts.append(pd.concat([grp, extra]))
            elif method == "random_undersample" and len(grp) > goal:
                parts.append(grp.iloc[np.sort(rng.choice(len(grp), goal, replace=False))])
            else:
                parts.append(grp)
        return pd.concat(parts).sort_index(kind="stable"), {"warnings": warns}
    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise RepairError("SMOTE needs the optional 'imbalanced-learn' package, which is not installed.")
        feats = [c for c in df.columns if c != tgt]
        if not all(is_numeric(df[c]) for c in feats):
            raise RepairError("SMOTE is only offered when every feature is numeric.")
        if df[feats].isna().any().any():
            raise RepairError("SMOTE requires no missing feature values.")
        k = int(p.get("k_neighbors", 5))
        if counts.min() <= k:
            raise RepairError(f"The smallest class has {counts.min()} rows; SMOTE needs more than k_neighbors={k}.")
        X, y = SMOTE(k_neighbors=k, random_state=seed).fit_resample(df[feats], df[tgt])
        out = pd.concat([X, y.rename(tgt)], axis=1)
        out.index = list(df.index) + [f"synthetic_{i}" for i in range(len(out) - len(df))]
        return out, {"warnings": warns + ["New rows are synthetic and carry 'synthetic_*' row ids."]}
    raise RepairError("method must be random_oversample, random_undersample or smote.")


OPERATIONS = {
    "remove_duplicates": (op_remove_duplicates, True),
    "fill_missing": (op_fill_missing, False),
    "drop_missing": (op_drop_missing, True),
    "drop_columns": (op_drop_columns, True),
    "normalize_whitespace": (op_normalize_whitespace, False),
    "map_categories": (op_map_categories, False),
    "convert_dtype": (op_convert_dtype, False),
    "handle_outliers": (op_handle_outliers, False),  # destructive only when action == 'remove'
    "balance_classes": (op_balance_classes, True),
}


def _is_destructive(op: str, params: dict) -> bool:
    if op == "handle_outliers":
        return params.get("action", "cap") == "remove"
    return OPERATIONS[op][1]


def _diff(before: pd.DataFrame, after: pd.DataFrame, limit=10) -> dict:
    info = {"rows_before": len(before), "rows_after": len(after), "columns_before": before.shape[1], "columns_after": after.shape[1],
            "rows_removed": 0, "rows_added": 0, "cells_modified": 0, "sample_changes": [], "sample_removed_row_ids": []}
    if before.index.is_unique and after.index.is_unique:
        removed = before.index.difference(after.index)
        added = after.index.difference(before.index)
        info["rows_removed"], info["rows_added"] = len(removed), len(added)
        info["sample_removed_row_ids"] = [to_native(i) for i in removed[:limit]]
        common, cols = after.index.intersection(before.index), [c for c in after.columns if c in before.columns]
        a, b = after.loc[common, cols].astype(object), before.loc[common, cols].astype(object)
        changed = (a != b) & ~(a.isna() & b.isna())
        info["cells_modified"] = int(changed.values.sum())
        for r, c in list(zip(*np.where(changed.values)))[:limit]:
            info["sample_changes"].append({"row_id": to_native(common[r]), "column": cols[c],
                                           "before": to_native(b.iloc[r, c]), "after": to_native(a.iloc[r, c])})
    else:
        info["rows_added"] = max(0, len(after) - len(before))
        info["rows_removed"] = max(0, len(before) - len(after))
    info["columns_changed"] = sorted(set(before.columns) ^ set(after.columns))
    return info


class RepairSession:
    """Holds an immutable original and an ordered, replayable list of repair steps."""

    def __init__(self, original: pd.DataFrame, seed: int = 42):
        self._original = original.copy(deep=True)
        self.seed = seed
        self.steps: list[dict] = []
        self.working = self._original.copy()

    @property
    def original(self) -> pd.DataFrame:
        return self._original.copy()

    def _run(self, df, op, params):
        if op not in OPERATIONS:
            raise RepairError(f"Unknown repair operation '{op}'. Available: {', '.join(OPERATIONS)}.")
        out, extra = OPERATIONS[op][0](df, dict(params), self.seed)
        return out, extra

    @staticmethod
    def _check_op(op: str) -> None:
        if op not in OPERATIONS:
            raise RepairError(f"Unknown repair operation '{op}'. Available: {', '.join(OPERATIONS)}.")

    def preview(self, op: str, params: dict) -> dict:
        self._check_op(op)
        out, extra = self._run(self.working, op, params)
        d = _diff(self.working, out)
        d["cells_modified"] = extra.get("cells_modified", d["cells_modified"])
        return {"operation": op, "params": to_native(params), "destructive": _is_destructive(op, params),
                "requires_confirmation": _is_destructive(op, params), "warnings": extra.get("warnings", []), **d}

    def apply(self, op: str, params: dict, confirm: bool = False) -> dict:
        self._check_op(op)
        if _is_destructive(op, params) and not confirm:
            raise RepairError(f"'{op}' removes data from the repaired copy and requires explicit confirmation (confirm=true).")
        out, extra = self._run(self.working, op, params)
        d = _diff(self.working, out)
        d["cells_modified"] = extra.get("cells_modified", d["cells_modified"])
        entry = {"step": len(self.steps) + 1, "operation": op, "params": to_native(params), "destructive": _is_destructive(op, params),
                 "confirmed": bool(confirm), "applied_at": datetime.now(timezone.utc).isoformat(), "seed": self.seed,
                 "warnings": extra.get("warnings", []), "status": "accepted",
                 **{k: d[k] for k in ("rows_before", "rows_after", "rows_removed", "rows_added", "cells_modified", "columns_changed")}}
        self.working = out
        self.steps.append(entry)
        return entry

    def revert(self, step: int) -> None:
        """Remove a step and replay the others from the original. Atomic: state is unchanged on failure."""
        if not 1 <= step <= len(self.steps):
            raise RepairError(f"No step {step}.")
        remaining = [s for s in self.steps if s["step"] != step]
        df, rebuilt = self._original.copy(), []
        for i, s in enumerate(remaining, 1):
            try:
                df, extra = self._run(df, s["operation"], s["params"])
            except RepairError as e:
                raise RepairError(f"Cannot revert step {step}: step {s['step']} ({s['operation']}) depends on it ({e}).")
            rebuilt.append({**s, "step": i})
        self.working, self.steps = df, rebuilt

    def log(self) -> list[dict]:
        return to_native(self.steps)

    def export_frame(self, include_row_id: bool = False) -> pd.DataFrame:
        out = self.working.copy()
        if include_row_id:
            out.insert(0, "_row_id", [str(i) for i in out.index])
        return out
