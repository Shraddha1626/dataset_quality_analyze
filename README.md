# AI-Powered Dataset Quality Analyzer and Intelligent Data Repair Assistant
Final-year Computer Engineering project (PVG's COE & SSDIOM, Nashik; SPPU, 2025-26).

## Current status (read this first)
| Component | Status |
|---|---|
| Structured-data analysis engine (missing, duplicates, types, outliers, imbalance, labels, correlation, leakage/bias *indicators*, profiling) | **Implemented and tested** |
| Dataset Health Score, ML-readiness, rule-based recommendations | **Implemented and tested** |
| Repair engine (9 operations, confirmation, replayable log, revert) + before/after comparison | **Implemented and tested** |
| JSON / CSV / PDF reports | **Implemented and tested** (PDF checked for validity only, not visually) |
| Safe CSV/Excel loader with limits | **Implemented and tested** |
| Image, text, audio analyzers | Not yet implemented |
| FastAPI layer, PostgreSQL/Alembic, history, storage | Not yet implemented |
| React frontend | Not yet implemented |
| Deployment config, Docker-free hosting docs | Not yet written; nothing has been deployed |

SMOTE is implemented but untested here (`imbalanced-learn` was unavailable offline); it fails with a clear message if the package is missing.

## Run the engine tests
```bash
cd backend
pip install pandas numpy scipy scikit-learn openpyxl reportlab pillow opencv-python-headless
python -m unittest discover -s tests -v      # 41 tests
```
Last run (Python 3.12.3, pandas 3.0.2): **41 tests, OK**.

## Quick use
```python
import pandas as pd
from app.analyzers.structured import AnalysisConfig
from app.services.analysis_service import run_structured_analysis
df = pd.read_csv("sample_datasets/sample_problematic_synthetic.csv")
res = run_structured_analysis(df, AnalysisConfig(target="label", purpose="classification", id_columns=["id"]))
print(res["health_score"]["score"], res["ml_readiness"]["category"])
```
See `docs/METHODOLOGY.md` for scoring and severity rules. Sample datasets are **synthetic**.
