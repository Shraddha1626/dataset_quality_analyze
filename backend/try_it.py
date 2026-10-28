import pandas as pd
from app.analyzers.structured import AnalysisConfig
from app.services.analysis_service import run_structured_analysis, compare_results
from app.repair.structured import RepairSession
from app.reports import builder

df = pd.read_csv("../sample_datasets/sample_problematic_synthetic.csv")   # or your own CSV
cfg = AnalysisConfig(target="label", purpose="classification", id_columns=["id"])

# Analyze
before = run_structured_analysis(df, cfg)
print("Score:", before["health_score"]["score"], before["health_score"]["category"])
print("Readiness:", before["ml_readiness"]["category"])
for i in before["issues"]:
    print(f'[{i["severity"]}] {i["title"]} ({i["affected_count"]} affected)')

# Repair (the original df is never modified)
s = RepairSession(df)
s.apply("remove_duplicates", {}, confirm=True)
s.apply("fill_missing", {"columns": ["age"], "strategy": "median"})
s.apply("normalize_whitespace", {"columns": ["city"]})
s.apply("map_categories", {"column": "city", "mapping": {"pune": "Pune", "PUNE": "Pune", "mumbai": "Mumbai"}})

# Compare and export
after = run_structured_analysis(s.working, cfg)
cmp = compare_results(before, after, s.log())
print("Score:", cmp["health_score"]["before"], "->", cmp["health_score"]["after"])

s.export_frame().to_csv("cleaned.csv", index=False)
report = builder.build_report("sample.csv", after, cmp, s.log())
open("report.json", "w", encoding="utf-8").write(builder.to_json(report))
open("issues.csv", "w", encoding="utf-8").write(builder.issues_csv(report))
open("report.pdf", "wb").write(builder.to_pdf(report))