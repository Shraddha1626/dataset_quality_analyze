"""ML-readiness assessment from explicit, documented checks. Never claims universal readiness."""
from __future__ import annotations

CATEGORIES = ["Ready for initial experimentation", "Ready after recommended fixes",
              "Requires substantial cleaning", "Insufficient information to assess"]


def assess_readiness(metrics: dict, issues: list[dict], purpose: str, target: str | None) -> dict:
    def of(*types):
        return [i for i in issues if i["type"] in types and i["status"] == "open"]

    checks = []

    def add(name, status, detail, related=()):
        checks.append({"check": name, "status": status, "detail": detail, "issue_ids": [i["id"] for i in related]})

    supervised = purpose in ("classification", "regression")
    if supervised:
        miss_t = of("missing_target_values")
        if not target:
            add("Target availability", "fail", "A supervised purpose was selected but no target column was set.")
        elif miss_t:
            add("Target availability", "fail", "Target has missing values.", miss_t)
        else:
            add("Target availability", "pass", f"Target '{target}' is set and complete.")
    else:
        add("Target availability", "not_evaluated", "Not required for this purpose.")

    mv = of("missing_values")
    if not mv:
        add("Missing values", "pass", "No missing values.")
    elif any(i["severity"] in ("critical", "high") for i in mv):
        add("Missing values", "fail", "Columns with substantial missingness.", mv)
    else:
        add("Missing values", "warn", "Some missing values need imputation.", mv)

    ty = of("numeric_as_string", "mixed_types", "date_as_string")
    add("Data types", "fail" if ty else "pass", f"{len(ty)} column(s) with type problems." if ty else "No type problems detected.", ty)

    dup = of("duplicate_rows")
    add("Duplicate records", ("fail" if dup and dup[0]["affected_pct"] >= 1 else "warn") if dup else "pass",
        f"{dup[0]['affected_count']} duplicate rows." if dup else "No duplicate rows.", dup)

    if metrics.get("target_evaluated"):
        r = metrics["imbalance_ratio"]
        imb = of("class_imbalance")
        add("Class distribution", "fail" if r >= 10 else "warn" if r >= 3 else "pass", f"Imbalance ratio {r:.1f}:1.", imb)
    else:
        add("Class distribution", "not_evaluated", "No class target available.")

    fq = of("constant_column", "high_correlation")
    add("Feature quality", "warn" if fq else "pass", "Constant or redundant features present." if fq else "No constant/redundant features.", fq)

    lk = of("potential_leakage", "possible_identifier")
    add("Leakage indicators", "warn" if lk else "pass",
        "Indicators found; verify with domain knowledge (not proof)." if lk else "No indicators found (this does not rule out leakage).", lk)

    lb = of("inconsistent_labels")
    add("Label consistency", ("fail" if any(target in i["columns"] for i in lb) else "warn") if lb else "pass",
        f"{len(lb)} column(s) with inconsistent labels." if lb else "No inconsistent labels detected.", lb)

    crit = [i for i in issues if i["severity"] == "critical" and i["status"] == "open"]
    add("Unresolved critical issues", "fail" if crit else "pass", f"{len(crit)} critical issue(s) open." if crit else "None.", crit)

    fails = [c for c in checks if c["status"] == "fail"]
    if metrics.get("n_rows", 0) < 30 or (supervised and not target):
        cat = CATEGORIES[3]
    elif not fails:
        cat = CATEGORIES[0]
    elif crit or len(fails) >= 4:
        cat = CATEGORIES[2]
    else:
        cat = CATEGORIES[1]
    steps = [i["recommendation"] for c in fails + [c for c in checks if c["status"] == "warn"] for i in issues if i["id"] in c["issue_ids"]][:6]
    return {"category": cat, "checks": checks,
            "passed": [c["check"] for c in checks if c["status"] == "pass"],
            "failed": [c["check"] for c in fails],
            "warnings": [c["check"] for c in checks if c["status"] == "warn"],
            "next_steps": list(dict.fromkeys(steps)),
            "disclaimer": "Readiness is judged against generic criteria; it does not guarantee suitability for every model or task."}
