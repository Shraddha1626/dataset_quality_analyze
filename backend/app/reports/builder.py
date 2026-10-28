"""Server-side report generation: JSON, CSV (issues) and PDF."""
from __future__ import annotations
import csv, io, json
from datetime import datetime, timezone
from xml.sax.saxutils import escape

from app.analyzers.models import to_native


def build_report(dataset_name: str, analysis: dict, comparison: dict | None = None, repair_log: list | None = None,
                 remaining_issues: list | None = None) -> dict:
    return to_native({
        "title": "Dataset Quality Report", "dataset_name": dataset_name, "dataset_type": analysis["dataset_type"],
        "generated_at": datetime.now(timezone.utc).isoformat(), "dimensions": {"rows": analysis["shape"][0], "columns": analysis["shape"][1]},
        "health_score": analysis["health_score"], "issues": analysis["issues"], "recommendations": analysis["recommendations"],
        "ml_readiness": analysis["ml_readiness"], "repair_operations": repair_log or [], "comparison": comparison,
        "remaining_issues": remaining_issues if remaining_issues is not None else analysis["issues"], "limitations": analysis["limitations"],
    })


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False)


ISSUE_COLS = ["id", "severity", "category", "title", "columns", "affected_count", "affected_pct", "confidence", "repair_available", "recommendation"]


def issues_csv(report: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(ISSUE_COLS)
    for i in report["issues"]:
        w.writerow([("; ".join(i[c]) if c == "columns" else i[c]) for c in ISSUE_COLS])
    return buf.getvalue()


def to_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    ss = getSampleStyleSheet()
    P = lambda t, st="BodyText": Paragraph(escape(str(t)), ss[st])
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                            title=f"Dataset Quality Report - {report['dataset_name']}")
    hs = report["health_score"]
    st = [P(report["title"], "Title"), P(f"Dataset: {report['dataset_name']} ({report['dataset_type']})"),
          P(f"Generated: {report['generated_at']}"),
          P(f"Dimensions: {report['dimensions']['rows']} rows x {report['dimensions']['columns']} columns"), Spacer(1, 10),
          P(f"Dataset Health Score: {hs['score'] if hs['score'] is not None else 'n/a'} / 100 - {hs['category']}", "Heading2"),
          P("Project-defined heuristic, not a probability or a prediction of model accuracy.", "Italic")]

    def table(rows, widths):
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                               ("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        return t

    if hs["dimensions"]:
        st += [Spacer(1, 6), table([["Dimension", "Score", "Weight", "Points lost", "Detail"]] +
               [[Paragraph(escape(d), ss["BodyText"]), v["score"], v["weight"], v["points_lost"], Paragraph(escape(v["detail"]), ss["BodyText"])]
                for d, v in hs["dimensions"].items()], [3 * cm, 1.6 * cm, 1.6 * cm, 2 * cm, 8 * cm])]
    st += [P("Detected issues", "Heading2")]
    if report["issues"]:
        st.append(table([["Severity", "Issue", "Columns", "Affected"]] +
                        [[i["severity"], Paragraph(escape(i["title"]), ss["BodyText"]), Paragraph(escape(", ".join(i["columns"])), ss["BodyText"]),
                          f"{i['affected_count']}" + (f" ({i['affected_pct']}%)" if i["affected_pct"] is not None else "")] for i in report["issues"]],
                        [2 * cm, 7 * cm, 4.5 * cm, 3 * cm]))
    else:
        st.append(P("No issues detected by the implemented checks (this does not guarantee the data is flawless)."))
    st.append(P("Recommendations", "Heading2"))
    for r in report["recommendations"][:25]:
        st.append(P(f"[{r['severity']}] {r['title']}: {r['recommended_action']}"))
        st.append(P(f"   Why: {r['why_it_helps']} Trade-offs: {r['trade_offs']}", "Italic"))
    rd = report["ml_readiness"]
    st += [P("ML-readiness assessment", "Heading2"), P(f"Category: {rd['category']}"), P(f"Passed: {', '.join(rd['passed']) or '-'}"),
           P(f"Failed: {', '.join(rd['failed']) or '-'}"), P(f"Warnings: {', '.join(rd['warnings']) or '-'}"), P(rd["disclaimer"], "Italic")]
    if report["repair_operations"]:
        st.append(P("Applied repair operations", "Heading2"))
        for s in report["repair_operations"]:
            st.append(P(f"{s['step']}. {s['operation']} {json.dumps(s['params'])} - removed {s['rows_removed']}, added {s['rows_added']}, cells modified {s['cells_modified']}"))
    c = report.get("comparison")
    if c:
        st += [P("Before / after", "Heading2"),
               table([["Metric", "Before", "After"], ["Health score", c["health_score"]["before"], c["health_score"]["after"]],
                      ["Rows", c["rows"]["before"], c["rows"]["after"]], ["Missing cells", c["missing_cells"]["before"], c["missing_cells"]["after"]],
                      ["Duplicate rows", c["duplicate_rows"]["before"], c["duplicate_rows"]["after"]],
                      ["Outlier values", c["outlier_cells"]["before"], c["outlier_cells"]["after"]], ["Issues", c["issues"]["before"], c["issues"]["after"]]],
                     [6 * cm, 4 * cm, 4 * cm]), P(c["note"], "Italic")]
    st.append(P("Limitations", "Heading2"))
    st += [P("- " + l) for l in report["limitations"]]
    doc.build(st)
    return buf.getvalue()
