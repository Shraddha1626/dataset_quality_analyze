import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from app.analyzers.structured import AnalysisConfig
from app.core.config import Settings
from app.recommendations.engine import build_recommendations
from app.repair.structured import RepairError, RepairSession
from app.reports import builder
from app.scoring.health import categorize, compute_health_score
from app.services.analysis_service import compare_results, run_structured_analysis
from app.services.loader import DatasetLoadError, load_table
from synthetic import PROBLEM_CFG, make_clean_df, make_problem_df


def issue(res, id_):
    return next((i for i in res["issues"] if i["id"] == id_), None)


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = make_problem_df()
        cls.res = run_structured_analysis(cls.df, AnalysisConfig(**PROBLEM_CFG))

    def test_shape(self):
        self.assertEqual(self.res["shape"], [210, 12])

    def test_missing_counts(self):
        self.assertEqual(issue(self.res, "missing_values:age")["affected_count"], 20)
        self.assertEqual(issue(self.res, "missing_values:notes")["affected_count"], 5)  # blanks count
        self.assertEqual(self.res["metrics"]["missing_cells"], 25)

    def test_duplicates(self):
        i = issue(self.res, "duplicate_rows:dataset")
        self.assertEqual(i["affected_count"], 10)
        self.assertEqual(i["affected_pct"], round(100 * 10 / 210, 2))

    def test_outliers_include_injected(self):
        i = issue(self.res, "outliers:age")
        self.assertGreaterEqual(i["affected_count"], 3)
        self.assertTrue({50, 51, 52} <= set(i["affected_rows"]))
        self.assertIsNone(issue(self.res, "outliers:id"))  # id column excluded

    def test_zscore_method(self):
        r = run_structured_analysis(self.df, AnalysisConfig(**PROBLEM_CFG, outlier_method="zscore"))
        self.assertGreaterEqual(issue(r, "outliers:age")["affected_count"], 3)

    def test_class_imbalance(self):
        i = issue(self.res, "class_imbalance:label")
        self.assertEqual(self.res["metrics"]["class_distribution"], {"yes": 200, "no": 10})
        self.assertAlmostEqual(self.res["metrics"]["imbalance_ratio"], 20.0)
        self.assertEqual(i["severity"], "critical")

    def test_types(self):
        self.assertEqual(issue(self.res, "numeric_as_string:score_str")["confidence"], "high")
        self.assertIsNotNone(issue(self.res, "date_as_string:date_str"))
        self.assertIsNotNone(issue(self.res, "mixed_types:mixed"))
        self.assertIsNone(issue(self.res, "numeric_as_string:mixed"))

    def test_labels_and_whitespace(self):
        i = issue(self.res, "inconsistent_labels:city")
        self.assertEqual(i["affected_count"], 16)
        self.assertEqual(i["repair_params"]["mapping"], {"pune": "Pune", "PUNE ": "Pune", "mumbai": "Mumbai"})
        self.assertEqual(issue(self.res, "whitespace_inconsistency:city")["affected_count"], 5)

    def test_redundancy_constant_leakage(self):
        pairs = issue(self.res, "high_correlation:dataset")["details"]["pairs"]
        self.assertTrue(any({p["a"], p["b"]} == {"income", "income_k"} for p in pairs))
        self.assertIsNotNone(issue(self.res, "constant_column:const"))
        lk = issue(self.res, "potential_leakage:label_code")
        self.assertIn("not proof", lk["explanation"])
        self.assertEqual(lk["confidence"], "medium")

    def test_profile_and_correlation(self):
        self.assertAlmostEqual(self.res["profile"]["income"]["median"], self.df["income"].median())
        self.assertIn("histogram", self.res["profile"]["income"])
        self.assertIn("matrix", self.res["correlation"])

    def test_json_serialisable(self):
        json.dumps(self.res)

    def test_unknown_column_rejected(self):
        with self.assertRaises(ValueError):
            run_structured_analysis(self.df, AnalysisConfig(target="nope"))

    def test_excluded_column_ignored(self):
        r = run_structured_analysis(self.df, AnalysisConfig(**PROBLEM_CFG, exclude_columns=["age"]))
        self.assertIsNone(issue(r, "missing_values:age"))

    def test_empty_and_tiny(self):
        r = run_structured_analysis(self.df.iloc[:3])
        self.assertIsNone(r["health_score"]["score"])
        self.assertEqual(r["health_score"]["category"], "Insufficient data")
        run_structured_analysis(self.df.iloc[0:0])

    def test_original_unchanged_by_analysis(self):
        before = make_problem_df()
        pd.testing.assert_frame_equal(self.df, before)


class ScoreTests(unittest.TestCase):
    def test_clean_dataset_scores_100(self):
        r = run_structured_analysis(make_clean_df(), AnalysisConfig(target="y", purpose="classification", id_columns=["id"]))
        self.assertEqual(r["issues"], [])
        self.assertEqual(r["health_score"]["score"], 100.0)
        self.assertEqual(r["health_score"]["category"], "Excellent")

    def test_problem_dataset_lower_and_bounded(self):
        r = run_structured_analysis(make_problem_df(), AnalysisConfig(**PROBLEM_CFG))
        s = r["health_score"]
        self.assertTrue(0 <= s["score"] < 100)
        for d in s["dimensions"].values():
            self.assertTrue(0 <= d["score"] <= 100)
        self.assertAlmostEqual(sum(d["weight"] for d in s["dimensions"].values()), 1.0, places=3)
        self.assertTrue(s["top_reducers"])

    def test_inapplicable_dimension_not_free_100(self):
        r = run_structured_analysis(make_clean_df(), AnalysisConfig(id_columns=["id"]))
        s = r["health_score"]
        self.assertNotIn("label_quality", s["dimensions"])
        self.assertEqual(s["not_evaluated"][0]["dimension"], "label_quality")
        self.assertAlmostEqual(sum(d["weight"] for d in s["dimensions"].values()), 1.0, places=3)

    def test_categories(self):
        self.assertEqual([categorize(x) for x in (95, 80, 65, 45, 10)],
                         ["Excellent", "Good", "Needs Improvement", "Poor", "Critical"])


class RecommendationTests(unittest.TestCase):
    def test_every_recommendation_explains(self):
        r = run_structured_analysis(make_problem_df(), AnalysisConfig(**PROBLEM_CFG))
        self.assertTrue(r["recommendations"])
        for rec in r["recommendations"]:
            for k in ("detected", "why_it_matters", "why_it_helps", "trade_offs", "manual_review_recommended"):
                self.assertIn(k, rec)
            self.assertFalse(rec["auto_apply"])
        by = {x["issue_id"]: x for x in r["recommendations"]}
        self.assertEqual(by["missing_values:age"]["suggested_params"]["strategy"] in ("median", "mean"), True)
        self.assertEqual(by["missing_values:notes"]["suggested_params"]["strategy"], "missing_category")
        self.assertTrue(by["class_imbalance:label"]["manual_review_recommended"])

    def test_readiness(self):
        r = run_structured_analysis(make_problem_df(), AnalysisConfig(**PROBLEM_CFG))
        self.assertEqual(r["ml_readiness"]["category"], "Requires substantial cleaning")
        c = run_structured_analysis(make_clean_df(), AnalysisConfig(target="y", purpose="classification", id_columns=["id"]))
        self.assertEqual(c["ml_readiness"]["category"], "Ready for initial experimentation")
        nt = run_structured_analysis(make_clean_df(), AnalysisConfig(purpose="classification", id_columns=["id"]))
        self.assertEqual(nt["ml_readiness"]["category"], "Insufficient information to assess")


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.df = make_problem_df()
        self.snapshot = self.df.copy(deep=True)
        self.s = RepairSession(self.df)

    def tearDown(self):
        pd.testing.assert_frame_equal(self.df, self.snapshot)        # caller's frame untouched
        pd.testing.assert_frame_equal(self.s.original, self.snapshot)  # session original untouched

    def test_destructive_requires_confirmation(self):
        with self.assertRaises(RepairError):
            self.s.apply("remove_duplicates", {})
        self.assertEqual(len(self.s.working), 210)

    def test_remove_duplicates(self):
        prev = self.s.preview("remove_duplicates", {})
        self.assertEqual(prev["rows_removed"], 10)
        self.assertEqual(len(self.s.working), 210)  # preview does not apply
        e = self.s.apply("remove_duplicates", {}, confirm=True)
        self.assertEqual((e["rows_before"], e["rows_after"], e["rows_removed"]), (210, 200, 10))
        self.assertEqual(self.s.working.duplicated().sum(), 0)
        self.assertEqual(list(self.s.working.index[:3]), [0, 1, 2])  # row ids preserved

    def test_fill_missing(self):
        e = self.s.apply("fill_missing", {"columns": ["age"], "strategy": "median"})
        self.assertEqual(self.s.working["age"].isna().sum(), 0)
        self.assertEqual(e["cells_modified"], 20)
        self.s.apply("fill_missing", {"columns": ["notes"], "strategy": "missing_category"})
        self.assertEqual((self.s.working["notes"] == "Missing").sum(), 5)

    def test_fill_missing_errors(self):
        with self.assertRaises(RepairError):
            self.s.apply("fill_missing", {"columns": ["city"], "strategy": "mean"})
        with self.assertRaises(RepairError):
            self.s.apply("fill_missing", {"columns": ["nope"], "strategy": "mean"})

    def test_normalize_and_map(self):
        self.s.apply("normalize_whitespace", {"columns": ["city"]})
        self.assertNotIn("PUNE ", set(self.s.working["city"]))
        with self.assertRaises(RepairError):
            self.s.apply("map_categories", {"column": "city", "mapping": {}})
        self.s.apply("map_categories", {"column": "city", "mapping": {"pune": "Pune", "PUNE": "Pune", "mumbai": "Mumbai"}})
        self.assertEqual(set(self.s.working["city"]), {"Pune", "Mumbai", "Nashik"})

    def test_convert_dtype(self):
        self.s.apply("convert_dtype", {"column": "score_str", "to": "numeric"})
        self.assertTrue(pd.api.types.is_float_dtype(self.s.working["score_str"]))
        with self.assertRaises(RepairError) as cm:
            self.s.apply("convert_dtype", {"column": "mixed", "to": "numeric"})
        self.assertIn("cannot be converted", str(cm.exception))
        self.s.apply("convert_dtype", {"column": "mixed", "to": "numeric", "max_failures": 105})
        self.assertEqual(self.s.working["mixed"].isna().sum(), 105)
        self.s.apply("convert_dtype", {"column": "date_str", "to": "datetime"})
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.s.working["date_str"]))

    def test_outliers(self):
        self.s.apply("handle_outliers", {"columns": ["age"], "action": "cap"})
        self.assertLess(self.s.working["age"].max(), 100)
        self.assertEqual(len(self.s.working), 210)
        with self.assertRaises(RepairError):
            self.s.apply("handle_outliers", {"columns": ["age"], "action": "remove"})
        e = self.s.apply("handle_outliers", {"columns": ["age"], "action": "remove"}, confirm=True)
        self.assertEqual(e["rows_removed"], 0)  # already capped

    def test_balance_random(self):
        with self.assertRaises(RepairError):
            self.s.apply("balance_classes", {"target": "label"})
        self.s.apply("balance_classes", {"target": "label", "method": "random_oversample"}, confirm=True)
        self.assertEqual(self.s.working["label"].value_counts().to_dict(), {"yes": 200, "no": 200})
        s2 = RepairSession(self.df)
        s2.apply("balance_classes", {"target": "label", "method": "random_undersample"}, confirm=True)
        self.assertEqual(s2.working["label"].value_counts().to_dict(), {"yes": 10, "no": 10})

    def test_balance_is_reproducible(self):
        a, b = RepairSession(self.df, seed=7), RepairSession(self.df, seed=7)
        for s in (a, b):
            s.apply("balance_classes", {"target": "label", "method": "random_oversample"}, confirm=True)
        pd.testing.assert_frame_equal(a.working, b.working)

    def test_smote_graceful(self):
        try:
            import imblearn  # noqa: F401
            self.skipTest("imbalanced-learn installed; unavailable-path not testable")
        except ImportError:
            with self.assertRaises(RepairError) as cm:
                self.s.apply("balance_classes", {"target": "label", "method": "smote"}, confirm=True)
            self.assertIn("not installed", str(cm.exception))

    def test_revert_replays(self):
        self.s.apply("remove_duplicates", {}, confirm=True)
        self.s.apply("fill_missing", {"columns": ["age"], "strategy": "mean"})
        self.s.revert(1)
        self.assertEqual(len(self.s.working), 210)
        self.assertEqual(self.s.working["age"].isna().sum(), 0)
        self.assertEqual([x["operation"] for x in self.s.log()], ["fill_missing"])
        self.assertEqual(self.s.log()[0]["step"], 1)

    def test_revert_dependent_fails_atomically(self):
        self.s.apply("convert_dtype", {"column": "score_str", "to": "numeric"})
        self.s.apply("handle_outliers", {"columns": ["score_str"], "action": "cap"})
        with self.assertRaises(RepairError):
            self.s.revert(1)
        self.assertEqual(len(self.s.steps), 2)

    def test_unknown_operation(self):
        with self.assertRaises(RepairError):
            self.s.apply("rm_rf", {})

    def test_export_with_row_ids(self):
        self.s.apply("remove_duplicates", {}, confirm=True)
        self.assertEqual(self.s.export_frame(include_row_id=True).columns[0], "_row_id")
        self.assertNotIn("_row_id", self.s.export_frame().columns)


class ComparisonAndReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = AnalysisConfig(**PROBLEM_CFG)
        cls.df = make_problem_df()
        cls.before = run_structured_analysis(cls.df, cls.cfg)
        s = RepairSession(cls.df)
        s.apply("remove_duplicates", {}, confirm=True)
        s.apply("fill_missing", {"columns": ["age"], "strategy": "median"})
        s.apply("fill_missing", {"columns": ["notes"], "strategy": "missing_category"})
        s.apply("normalize_whitespace", {"columns": ["city"]})
        s.apply("map_categories", {"column": "city", "mapping": {"pune": "Pune", "PUNE": "Pune", "mumbai": "Mumbai"}})
        s.apply("convert_dtype", {"column": "score_str", "to": "numeric"})
        s.apply("drop_columns", {"columns": ["const", "label_code", "income_k"]}, confirm=True)
        cls.log = s.log()
        cls.after = run_structured_analysis(s.working, cls.cfg)
        cls.cmp = compare_results(cls.before, cls.after, cls.log)

    def test_metrics_calculated_from_data(self):
        c = self.cmp
        self.assertEqual((c["missing_cells"]["before"], c["missing_cells"]["after"]), (25, 0))
        self.assertEqual((c["duplicate_rows"]["before"], c["duplicate_rows"]["after"]), (10, 0))
        self.assertEqual(c["records_removed"], 10)
        self.assertGreater(c["health_score"]["after"], c["health_score"]["before"])
        self.assertIn("missing_values:age", c["issues"]["resolved_ids"])
        self.assertNotIn("outliers:age", c["issues"]["resolved_ids"])  # outliers were NOT repaired
        self.assertIn("do not show", c["note"])

    def test_json_and_csv_reports(self):
        rep = builder.build_report("demo.csv", self.after, self.cmp, self.log)
        data = json.loads(builder.to_json(rep))
        self.assertEqual(data["dataset_name"], "demo.csv")
        self.assertEqual(len(data["repair_operations"]), 7)
        rows = builder.issues_csv(builder.build_report("demo.csv", self.before)).strip().splitlines()
        self.assertEqual(len(rows), 1 + len(self.before["issues"]))
        self.assertTrue(rows[0].startswith("id,severity"))

    def test_pdf_report(self):
        pdf = builder.to_pdf(builder.build_report("demo.csv", self.before, self.cmp, self.log))
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 3000)


def write(path, text):
    with open(path, "w") as f:
        f.write(text)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_xlsx_roundtrip(self):
        df = make_clean_df()
        df.to_csv(os.path.join(self.d, "a.csv"), index=False)
        df.to_excel(os.path.join(self.d, "a.xlsx"), index=False)
        self.assertEqual(load_table(os.path.join(self.d, "a.csv")).shape, df.shape)
        self.assertEqual(load_table(os.path.join(self.d, "a.xlsx")).shape, df.shape)

    def test_latin1_csv(self):
        p = os.path.join(self.d, "l.csv")
        with open(p, "wb") as f:
            f.write("name,v\ncaf\xe9,1\n".encode("latin-1"))
        self.assertEqual(load_table(p)["name"][0], "café")

    def test_rejections(self):
        p = os.path.join(self.d, "x.exe")
        write(p, "x")
        with self.assertRaises(DatasetLoadError):
            load_table(p)
        big = os.path.join(self.d, "b.csv")
        make_clean_df(n=500).to_csv(big, index=False)
        with self.assertRaises(DatasetLoadError) as cm:
            load_table(big, Settings(max_rows=100))
        self.assertNotIn(self.d, str(cm.exception))  # no server paths leaked
        with self.assertRaises(DatasetLoadError):
            load_table(big, Settings(max_upload_size_mb=0))
        empty = os.path.join(self.d, "e.csv")
        write(empty, "")
        with self.assertRaises(DatasetLoadError):
            load_table(empty)
        bad = os.path.join(self.d, "bad.xlsx")
        write(bad, "not excel")
        with self.assertRaises(DatasetLoadError):
            load_table(bad)


if __name__ == "__main__":
    unittest.main()
