"""Synthetic datasets with KNOWN, injected quality problems (for tests and demos)."""
import numpy as np
import pandas as pd


def make_problem_df(seed: int = 0) -> pd.DataFrame:
    """200 base rows + 10 exact duplicates of rows 0-9 = 210 rows.

    Injected (expected) facts:
      age:       exactly 20 missing (rows 20-39); 3 extreme outliers (rows 50-52)
      notes:     5 blank strings (counted as missing)
      score_str: numeric values stored as text
      date_str:  ISO dates stored as text (30 distinct values)
      mixed:     half numeric strings, half words
      city:      16 non-dominant spelling variants (15 Pune-variants + 1 Mumbai-variant); 5 with trailing spaces
      label:     200 'yes' / 10 'no' -> imbalance ratio 20.0
      label_code: 0/1 copy of the label (leakage indicator)
      income_k:  income / 1000 (perfectly correlated -> redundant)
      const:     constant column
      duplicates: 10
    """
    rng = np.random.default_rng(seed)
    n = 200
    df = pd.DataFrame({
        "id": np.arange(n),
        "age": rng.normal(40, 5, n).round(1),
        "income": rng.normal(50000, 8000, n).round(0),
        "city": [["Pune", "Mumbai", "Nashik"][i % 3] for i in range(n)],
        "score_str": [f"{v:.1f}" for v in rng.uniform(0, 100, n)],
        "date_str": [f"2024-01-{(i % 30) + 1:02d}" for i in range(n)],
        "mixed": [str(i) if i % 2 == 0 else "unknown" for i in range(n)],
        "notes": ["ok"] * n,
        "label": ["yes"] * n,
        "const": 1,
    })
    df.loc[100:109, "label"] = "no"
    df["label_code"] = (df["label"] == "yes").astype(int)
    df["income_k"] = df["income"] / 1000
    df.loc[20:39, "age"] = np.nan
    df.loc[50:52, "age"] = 500.0
    df.loc[10:14, "notes"] = "  "   # not in the duplicated rows 0-9
    df.loc[60:69, "city"] = "pune"
    df.loc[70:74, "city"] = "PUNE "
    df.loc[80, "city"] = "mumbai"
    return pd.concat([df, df.iloc[:10]], ignore_index=True)


def make_clean_df(seed: int = 1, n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n), "x1": rng.uniform(0, 1, n), "x2": rng.uniform(0, 1, n),
        "grp": rng.choice(["a", "b", "c"], n), "y": ["pos", "neg"] * (n // 2),
    })


PROBLEM_CFG = dict(target="label", purpose="classification", id_columns=["id"])
