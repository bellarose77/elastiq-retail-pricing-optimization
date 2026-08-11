"""Generate the share-ready ELASTIQ team overview and technical system guide."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/elastiq-report-mpl")

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
TMP = ROOT / "tmp" / "pdfs" / "team-reports"
DATA = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
TMP.mkdir(parents=True, exist_ok=True)

OVERVIEW_PDF = OUT / "ELASTIQ_Team_Overview.pdf"
TECHNICAL_PDF = OUT / "ELASTIQ_Technical_System_Guide.pdf"

NAVY = colors.HexColor("#172338")
BLUE = colors.HexColor("#2F66FF")
MID_BLUE = colors.HexColor("#5D84FF")
PALE_BLUE = colors.HexColor("#EBF0FF")
GREEN = colors.HexColor("#078763")
PALE_GREEN = colors.HexColor("#E8F7F1")
AMBER = colors.HexColor("#D08A00")
PALE_AMBER = colors.HexColor("#FFF5DD")
RED = colors.HexColor("#C2344B")
PALE_RED = colors.HexColor("#FCECEF")
INK = colors.HexColor("#202B3D")
MUTED = colors.HexColor("#66738A")
LINE = colors.HexColor("#D9E0EA")
PALE = colors.HexColor("#F5F7FA")
WHITE = colors.white


def register_fonts():
    regular = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    bold = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("Elastiq", str(regular)))
        pdfmetrics.registerFont(TTFont("Elastiq-Bold", str(bold)))
        return "Elastiq", "Elastiq-Bold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverKicker", fontName=FONT_BOLD, fontSize=9, leading=12, textColor=colors.HexColor("#93ADFF"), spaceAfter=8))
styles.add(ParagraphStyle(name="CoverTitle", fontName=FONT_BOLD, fontSize=27, leading=32, textColor=WHITE, spaceAfter=12))
styles.add(ParagraphStyle(name="CoverSub", fontName=FONT, fontSize=11, leading=17, textColor=colors.HexColor("#DCE5F6"), spaceAfter=10))
styles.add(ParagraphStyle(name="H1", fontName=FONT_BOLD, fontSize=20, leading=25, textColor=NAVY, spaceAfter=7))
styles.add(ParagraphStyle(name="H2", fontName=FONT_BOLD, fontSize=13, leading=17, textColor=NAVY, spaceBefore=7, spaceAfter=5))
styles.add(ParagraphStyle(name="H3", fontName=FONT_BOLD, fontSize=9.5, leading=13, textColor=BLUE, spaceBefore=4, spaceAfter=3))
styles.add(ParagraphStyle(name="Body", fontName=FONT, fontSize=8.7, leading=13, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="Small", fontName=FONT, fontSize=7.1, leading=9.6, textColor=MUTED, spaceAfter=3))
styles.add(ParagraphStyle(name="Tiny", fontName=FONT, fontSize=6.2, leading=8, textColor=INK))
styles.add(ParagraphStyle(name="TableHead", fontName=FONT_BOLD, fontSize=7.1, leading=9, textColor=WHITE))
styles.add(ParagraphStyle(name="TableBody", fontName=FONT, fontSize=7, leading=9.2, textColor=INK))
styles.add(ParagraphStyle(name="LayerTitle", fontName=FONT_BOLD, fontSize=9.3, leading=12, textColor=NAVY, spaceAfter=3))
styles.add(ParagraphStyle(name="LayerBody", fontName=FONT, fontSize=7.6, leading=10.8, textColor=INK))
styles.add(ParagraphStyle(name="Metric", fontName=FONT_BOLD, fontSize=18, leading=21, textColor=GREEN, alignment=TA_CENTER))
styles.add(ParagraphStyle(name="MetricLabel", fontName=FONT_BOLD, fontSize=6.6, leading=8.2, textColor=MUTED, alignment=TA_CENTER))


def p(text, style="Body"):
    return Paragraph(str(text), styles[style])


def title(text, eyebrow=None):
    out = []
    if eyebrow:
        out.append(p(eyebrow.upper(), "H3"))
    out += [p(text, "H1"), HRFlowable(width="100%", thickness=.7, color=LINE), Spacer(1, 4 * mm)]
    return out


def table(rows, widths, small=False, header=True):
    body_style = "Tiny" if small else "TableBody"
    content = []
    for i, row in enumerate(rows):
        content.append([p(cell, "TableHead" if header and i == 0 else body_style) for cell in row])
    t = Table(content, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY if header else WHITE),
        ("GRID", (0, 0), (-1, -1), .35, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(2 if header else 1, len(rows), 2):
        commands.append(("BACKGROUND", (0, i), (-1, i), PALE))
    t.setStyle(TableStyle(commands))
    return t


def callout(text, tone="blue"):
    palettes = {
        "blue": (PALE_BLUE, BLUE), "green": (PALE_GREEN, GREEN),
        "amber": (PALE_AMBER, AMBER), "red": (PALE_RED, RED),
    }
    bg, border = palettes[tone]
    t = Table([[p(text, "Body")]], colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX", (0, 0), (-1, -1), .7, border),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def bullets(items, style="Body"):
    return [p(f"<bullet>&bull;</bullet>{item}", style) for item in items]


def metric_cards(cards):
    cells = [[p(value, "Metric"), p(label.upper(), "MetricLabel")] for value, label in cards]
    t = Table([cells], colWidths=[43.5 * mm] * len(cards))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE),
        ("BOX", (0, 0), (-1, -1), .5, LINE),
        ("INNERGRID", (0, 0), (-1, -1), .5, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def page_frame(canvas, doc):
    canvas.saveState()
    width, height = doc.pagesize
    if doc.page > 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 15 * mm, width, 15 * mm, stroke=0, fill=1)
        canvas.setFont(FONT_BOLD, 8)
        canvas.setFillColor(WHITE)
        canvas.drawString(18 * mm, height - 9.6 * mm, "ELASTIQ | RETAIL PRICING DECISION STUDIO")
        canvas.setStrokeColor(LINE)
        canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
        canvas.setFont(FONT, 7)
        canvas.setFillColor(MUTED)
        canvas.drawString(18 * mm, 9 * mm, "Team-ready decision support documentation")
        canvas.drawRightString(width - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def cover(canvas, doc):
    canvas.saveState()
    width, height = doc.pagesize
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, stroke=0, fill=1)
    canvas.setFillColor(BLUE)
    canvas.circle(width - 10 * mm, height - 12 * mm, 46 * mm, stroke=0, fill=1)
    canvas.setFillColor(colors.HexColor("#2949B9"))
    canvas.circle(width - 5 * mm, 8 * mm, 55 * mm, stroke=0, fill=1)
    canvas.restoreState()


def make_visuals():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    pipeline = TMP / "pipeline.png"
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")
    labels = ["Validate", "Explore", "Elasticity", "Causal test", "Promotion", "Forecast", "Market", "Optimize"]
    cols = ["#8796AC", "#8796AC", "#2F66FF", "#2F66FF", "#2F66FF", "#2F66FF", "#C2344B", "#C2344B"]
    for i, (lab, col) in enumerate(zip(labels, cols)):
        x = .02 + i * .122
        ax.add_patch(plt.Rectangle((x, .34), .106, .3, facecolor=col, edgecolor="none", transform=ax.transAxes))
        ax.text(x + .053, .49, f"{i+1:02d}\n{lab}", ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
        if i < 7:
            ax.annotate("", xy=(x + .121, .49), xytext=(x + .106, .49), xycoords=ax.transAxes, arrowprops=dict(arrowstyle="->", color="#61708A", lw=1.4))
    ax.text(.02, .76, "WHAT IS TRUE?", color="#61708A", fontweight="bold", transform=ax.transAxes)
    ax.text(.265, .76, "WHAT WILL HAPPEN?", color="#2F66FF", fontweight="bold", transform=ax.transAxes)
    ax.text(.755, .76, "WHAT SHOULD WE DO?", color="#C2344B", fontweight="bold", transform=ax.transAxes)
    ax.text(.5, .16, "Every stage produces evidence or a gate. Only Optimize selects a price.", ha="center", color="#202B3D", transform=ax.transAxes)
    fig.savefig(pipeline, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    evidence = TMP / "evidence.png"
    forecast = pd.read_csv(DATA / "xgboost_naive_baseline_comparison.csv")
    actions = pd.read_csv(DATA / "price_optimization_action_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    names = ["Lag 1", "Lag 7", "Rolling 7", "XGBoost"]
    axes[0].barh(names, forecast["mae"], color=["#B4BECC", "#97A5B7", "#76879E", "#2F66FF"])
    axes[0].invert_yaxis()
    axes[0].set_title("Forecast error on held-out dates")
    axes[0].set_xlabel("MAE in units - lower is better")
    for i, value in enumerate(forecast["mae"]):
        axes[0].text(value + .2, i, f"{value:.2f}", va="center")
    labels = ["Increase", "Hold", "Decrease"]
    axes[1].bar(labels, actions["item_count"], color=["#2F66FF", "#8796AC", "#078763"])
    axes[1].set_title("Illustrative action mix")
    axes[1].set_ylabel("Priced store-product rows")
    for i, value in enumerate(actions["item_count"]):
        axes[1].text(i, value + .5, str(int(value)), ha="center", fontweight="bold")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    fig.savefig(evidence, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    gates = TMP / "gates.png"
    fig, ax = plt.subplots(figsize=(8.7, 4.1))
    ax.axis("off")
    gate_labels = [
        ("Data ready", .92, "#8796AC"), ("Causal evidence", .78, "#2F66FF"),
        ("Forecast ready", .64, "#426FFF"), ("Promotion ready", .50, "#5D84FF"),
        ("Price feasible", .36, "#C2344B"), ("Independent audit", .22, "#078763"),
    ]
    for i, (lab, width, col) in enumerate(gate_labels):
        y = .83 - i * .13
        left = (1 - width) / 2
        ax.add_patch(plt.Rectangle((left, y), width, .09, facecolor=col, edgecolor="none", transform=ax.transAxes))
        ax.text(.5, y + .045, lab, color="white", ha="center", va="center", fontweight="bold", transform=ax.transAxes)
    ax.text(.5, .06, "If a required gate fails, the item is held with a reason.", ha="center", color="#202B3D", fontweight="bold", transform=ax.transAxes)
    fig.savefig(gates, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pipeline, evidence, gates


PIPELINE_IMG, EVIDENCE_IMG, GATES_IMG = make_visuals()


def build_overview():
    doc = SimpleDocTemplate(str(OVERVIEW_PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=22 * mm, bottomMargin=18 * mm,
                            title="ELASTIQ Team Overview", author="ELASTIQ", subject="High-level retail pricing decision support overview")
    s = []
    s += [Spacer(1, 58 * mm), p("TEAM OVERVIEW", "CoverKicker"), p("ELASTIQ retail pricing decision studio", "CoverTitle"),
          p("A visual, plain-language guide to how the application turns retail history and market evidence into controlled next-period pricing recommendations.", "CoverSub"),
          Spacer(1, 17 * mm)]
    cover_cells = Table([
        [p("AUDIENCE", "MetricLabel"), p("PRIMARY DECISION", "MetricLabel"), p("OPERATING MODE", "MetricLabel")],
        [p("Commercial and technical teams", "CoverSub"), p("Raise, hold, or lower price", "CoverSub"), p("Reviewed decision support", "CoverSub")],
    ], colWidths=[58 * mm] * 3)
    cover_cells.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#21304A")), ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#435475")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]))
    s += [cover_cells, PageBreak()]

    s += title("What ELASTIQ does", "The application in one page")
    s += [callout("ELASTIQ recommends the most suitable next-period price for each store-product decision unit, or explicitly holds the current price when the evidence is not strong enough.", "green"), Spacer(1, 4 * mm),
          metric_cards([("8", "Analytical stages"), ("6", "Optimization methods"), ("50-2,000", "Test-case size"), ("5", "Visible run steps")]),
          p("ELASTIQ combines five questions that are normally answered separately:"),
          table([
              ["Question", "How the application answers it"],
              ["Can we trust the input?", "Validation checks completeness, consistency, types, ranges, duplicates, and business rules."],
              ["How does price affect demand?", "Descriptive and causal elasticity models estimate the demand response to a price change."],
              ["Will a promotion create incremental demand?", "Promotion uplift is estimated and allowed into pricing only when readiness gates pass."],
              ["What will demand be next period?", "A time-safe forecast creates the do-nothing quantity baseline."],
              ["What price is commercially feasible?", "Optimization evaluates candidate prices against margin, demand, competitor, inventory, and implementation controls."],
          ], [58 * mm, 116 * mm]),
          Spacer(1, 4 * mm), callout("The application supports review and approval. It does not automatically publish a price.", "blue"), PageBreak()]

    s += title("Understand the input before running", "Initial data analysis")
    s += [callout("Every decision starts with a profile of the exact rows entering the optimizer. This makes data gaps, unusual commercial structure, and evidence readiness visible before recommendations are calculated.", "green"),
          table([
              ["Input group", "Example fields", "What the initial analysis explains"],
              ["Identity", "Item, product, store, category, region", "How many distinct decisions are loaded and how they are distributed."],
              ["Current economics", "Current price, unit cost, competitor price", "Price range, average margin, and average premium or discount to competitors."],
              ["Demand and capacity", "Baseline forecast, available inventory", "Total and median demand plus the share of rows under inventory pressure."],
              ["Price evidence", "Elasticity, source, causal flag, confidence", "Elasticity range and how much of the portfolio has decision-ready causal evidence."],
              ["Promotion / market", "Uplift, readiness, market signal", "Promotion readiness and the bounded market context available to candidate scenarios."],
          ], [39 * mm, 58 * mm, 77 * mm], small=True),
          Spacer(1, 4 * mm),
          metric_cards([("100%", "Numeric completeness"), ("0", "Invalid rows"), ("6", "Category views"), ("Live", "Edit feedback")]),
          p("The Data page updates the profile immediately after an edit, row addition, or CSV import. A category table then shows price, margin, forecast demand, inventory risk, causal coverage, and promotion readiness side by side."),
          callout("A high completeness score is necessary but not sufficient. Causal coverage, forecast provenance, feasible margin, inventory, and promotion readiness still determine whether a row may be repriced.", "amber"), PageBreak()]

    s += title("The complete journey", "From raw evidence to a decision")
    s += [Image(str(PIPELINE_IMG), width=174 * mm, height=56 * mm), Spacer(1, 3 * mm),
          table([
              ["Act", "Stages", "What the team receives"],
              ["1. Establish truth", "01 Validation, 02 Exploration", "A trusted analytical dataset and clear commercial context."],
              ["2. Estimate response", "03 Elasticity, 04 Causal test, 05 Promotion, 06 Forecast", "Causal price sensitivity, promotion readiness, and next-period baseline demand."],
              ["3. Make a decision", "07 Market evidence, 08 Optimization", "A bounded recommendation with expected results, constraints, provenance, and status."],
          ], [40 * mm, 60 * mm, 74 * mm]),
          p("Every item remains visible throughout the flow. If a required input or evidence gate fails, the final output records a hold and a machine-readable reason instead of silently dropping the item."), PageBreak()]

    s += title("How one price is selected", "Decision logic")
    s += [Image(str(GATES_IMG), width=158 * mm, height=75 * mm),
          table([
              ["Step", "Decision made"],
              ["1. Establish the baseline", "Use the forecast quantity at the current price for the next selling period."],
              ["2. Generate candidates", "Create allowed prices inside configured increase and decrease limits."],
              ["3. Estimate response", "Apply causal elasticity and only eligible bounded promotion and market adjustments."],
              ["4. Calculate economics", "Cap realized sales at inventory and calculate comparable revenue and profit."],
              ["5. Enforce controls", "Reject candidates that break margin, quantity, profit, competitor, inventory, or step rules."],
              ["6. Select and audit", "Choose the best feasible candidate, apply a no-harm fallback, and recompute every control independently."],
          ], [42 * mm, 132 * mm]), PageBreak()]

    s += title("What the team sees", "Outputs and responsibilities")
    s += [Image(str(EVIDENCE_IMG), width=174 * mm, height=63 * mm),
          table([
              ["Role", "What they use", "Typical action"],
              ["Pricing / category manager", "Recommended price, expected demand, profit change, binding constraints, evidence status", "Approve, hold, or document an override"],
              ["Revenue / finance", "Portfolio and category summaries, margin and profit controls", "Confirm the commercial objective and downside limits"],
              ["Data science", "Elasticity diagnostics, forecast metrics, promotion readiness, evidence provenance", "Monitor quality, recalibrate, and investigate holds"],
              ["Engineering", "Pipeline status, schemas, exports, parity and regression tests", "Operate releases, monitor failures, and preserve traceability"],
              ["Leadership", "Action mix, portfolio deltas, readiness and exception counts", "Approve pilot scope and governance"],
          ], [42 * mm, 76 * mm, 56 * mm], small=True),
          callout("Illustrative results use the packaged synthetic dataset. Live commercial performance must be measured with a controlled rollout and holdout group.", "amber"), PageBreak()]

    s += title("Using the application", "Simple operating workflow")
    s += [table([
        ["Moment", "User action", "Application response"],
        ["Start", "Run start.bat on Windows or ./start.sh on Linux/macOS.", "Installs web dependencies on first use and opens the workbench."],
        ["Load", "Use packaged data or import a compatible CSV.", "Validates columns, values, and evidence provenance."],
        ["Configure", "Choose objective, technique, price limits, margin, competitor, inventory, promotion, and implementation controls.", "Shows the active decision policy before optimization."],
        ["Run", "Execute the pricing analysis.", "Returns item recommendations, scenario comparisons, portfolio KPIs, and control status."],
        ["Review", "Inspect large moves, holds, binding constraints, and evidence quality.", "Preserves reasons and provenance for every decision row."],
        ["Export", "Download CSV or PDF.", "Creates a review-ready decision package."],
    ], [28 * mm, 72 * mm, 74 * mm]),
          p("Optional: `start.bat --refresh-data` or `./start.sh --refresh-data` creates an isolated Python environment, reruns all eight analytical stages, updates the browser dataset, and then launches the application."), PageBreak()]

    s += title("Run and watch live experiments", "Deep Analysis Runner")
    s += [callout("The Deep Analysis Runner is the application's opening workflow. It executes refinement, stress re-optimization, uncertainty simulation and validation against the exact generated rows and active policy. Recommendations are not precomputed or replayed.", "green"),
          table([
              ["1. Choose the problem", "2. Choose the approach", "3. Watch and review"],
              ["Small (50), Medium (250), Large (1,000), or custom 5-2,000 decision units. Select balanced, inventory-constrained, promotion-intensive, or volatile conditions; every dataset has a reproducible seed.",
               "All six methods are selected by default. Choose Standard, Deep analysis, or Research workload. Deep is the default; Hybrid routes feasible items by inventory, uncertainty, promotion, and objective context.",
               "Follow Start/profile, optimize/refine, stress test, risk simulation, validation, and Finish. Watch exact optimizer and candidate counters, runtime, events, live charts, risk percentiles, consensus and top decisions."],
          ], [58 * mm, 58 * mm, 58 * mm], small=True),
          p("The workload runs in a dedicated browser worker, keeping the page responsive. Deep analysis uses three price-grid resolutions, size-adjusted shocked-market re-optimization and 1,000 Monte Carlo draws per method. Research uses four resolutions, more stress states and 3,000 draws. There is no artificial wait. A completed run retains the frozen configuration, convergence, stress distribution, risk percentiles, exact work counters, decisions and analysis report."),
          table([
              ["Deep-mode QA case", "Measured compute time", "Candidate evaluations"],
              ["Small / 50 units / six methods", "5.4 seconds", "16.2 million"],
              ["Medium / 250 units / six methods", "9.2 seconds", "28.6 million"],
              ["Large / 1,000 units / six methods", "26.9 seconds", "62.7 million"],
          ], [62 * mm, 54 * mm, 58 * mm], small=True),
          p("Times were measured during release QA on the build environment and will vary by computer. Exact operation counts are reported by every live run."),
          table([
              ["Approach", "Use it when", "Result"],
              ["Single", "You need one transparent answer or want to test a chosen technique.", "One measured method result."],
              ["Compare", "You want a fair benchmark under the same rows and policy.", "Side-by-side profit, revenue, runtime, movement, and readiness."],
              ["Hybrid", "Different items may benefit from different optimization logic.", "One feasible decision per item plus a method-contribution breakdown."],
          ], [32 * mm, 72 * mm, 70 * mm]),
          callout("Run history is available for the current browser session. Any completed run can be reopened with its charts and analysis or downloaded as a full JSON record or two-page Deep Analysis Run PDF for sharing and audit.", "blue"), PageBreak()]

    s += title("Controls, expectations, and next steps", "Responsible use")
    s += [table([
        ["The application is ready for", "Additional controls required for live publishing"],
        ["Team demonstrations and training", "Authenticated source connections and schema contracts"],
        ["Reviewed scenario analysis", "User roles, approval workflow, and audit retention"],
        ["Synthetic and shadow-data validation", "Model registry, monitoring, drift alerts, and rollback"],
        ["Controlled assortment pilots", "Retailer-specific causal validation and legal/policy review"],
        ["CSV/PDF decision packages", "Controlled price publishing integration"],
    ], [87 * mm, 87 * mm]),
          Spacer(1, 5 * mm), callout("Recommended team practice: treat every recommendation as evidence-backed advice, preserve all holds and overrides, and measure outcomes against a predefined control group.", "green"),
          p("For implementation details, formulas, schemas, testing, and step-by-step functionality, use the companion ELASTIQ Technical System Guide."),
          Spacer(1, 12 * mm), HRFlowable(width="100%", thickness=.7, color=LINE), Spacer(1, 4 * mm), p("ELASTIQ Team Overview | Final team document", "Small")]
    doc.build(s, onFirstPage=cover, onLaterPages=page_frame)


FUNCTIONS = [
    {
        "title": "Application startup and runtime",
        "purpose": "Give Windows, Linux, and macOS users one predictable way to start the workbench and an optional way to refresh analytical outputs.",
        "level1": "Run `start.bat` on Windows or `./start.sh` on Linux/macOS. The browser opens at localhost and the terminal remains the application control window.",
        "level2": "The launcher checks Node.js and npm, creates a local runtime folder, installs exact web dependencies only when needed, starts Vite on 127.0.0.1, and opens the default browser. `--refresh-data` also provisions a Python virtual environment and executes the eight stages.",
        "level3": "Runtime state is isolated under `.runtime`; npm uses a project-local cache. `scripts/run_pipeline.py` calls every stage with the active interpreter and prepends the repository root to PYTHONPATH. A non-zero subprocess status stops the refresh. The launcher does not expose the service beyond localhost.",
        "io": ("Node.js 18+; optional Python 3.10+", "Local Vite workbench; optionally refreshed CSV/model artifacts"),
        "controls": "Dependency checks, exact `npm ci`, local-only host binding, fail-fast process status, Ctrl+C shutdown.",
    },
    {
        "title": "Input data profiling and initial analysis",
        "purpose": "Explain the exact decision rows before optimization and surface structural, commercial, capacity, or evidence-readiness concerns early.",
        "level1": "Shows what data is loaded, whether it is complete, and what the portfolio looks like before a price is calculated.",
        "level2": "Profiles decision-unit identity, categories, stores, products, numeric completeness, invalid relationships, duplicate IDs, price distribution, demand, margin, competitor position, inventory pressure, elasticity range, causal coverage, and promotion readiness. A category view identifies where demand, risk, and evidence are concentrated.",
        "level3": "`analyzeInput` evaluates nine core numeric fields, finite-value coverage, price/cost and sign rules, unique `itemId` values, and evidence provenance. It calculates min, median, mean, and total statistics using finite values only. Inventory pressure means available stock is at or below 105% of baseline demand. Causal readiness recognizes the explicit flag or approved IV source; category aggregates retain items, average price and margin, demand, inventory-risk rate, causal rate, and promotion-readiness rate. React memoization recalculates the profile whenever input rows change.",
        "io": ("Packaged, generated, edited, or CSV-imported decision rows", "Portfolio and category input profile, completeness and readiness indicators"),
        "controls": "Finite-value checks, required commercial relationships, duplicate detection, explicit causal provenance, transparent readiness definitions.",
    },
    {
        "title": "01 - Data validation",
        "purpose": "Prevent missing, duplicate, malformed, or commercially inconsistent records from contaminating models and decisions.",
        "level1": "Checks whether transaction and market-note data are complete and sensible before any analysis begins.",
        "level2": "Validates required fields, numeric types, missing values, duplicates, date coverage, price and cost relationships, inventory consistency, revenue arithmetic, and permitted ranges. It writes a clean dataset plus a quality summary and consistency audit.",
        "level3": "The stage loads configured raw sources, applies column aliases, coerces dates and numerics, and separates critical checks from review warnings. Cross-field rules include selling price versus regular price, units versus inventory, unit cost versus selling price, and recomputed revenue/profit tolerances. A manifest records the stage inputs and outputs.",
        "io": ("Raw retail transactions and market notes", "Clean retail data, clean notes, overview, quality summary, audit, manifest"),
        "controls": "Required-schema checks, critical/review severity, deterministic output paths, explicit row counts and date range.",
    },
    {
        "title": "02 - Exploratory analysis",
        "purpose": "Describe where demand, revenue, margin, discounting, promotion, and stock risk are concentrated before modelling.",
        "level1": "Turns validated rows into business KPIs and portfolio views that explain what is happening.",
        "level2": "Calculates overall KPIs and summaries by category, product, store, region, discount band, and date. It visualizes revenue, units, margin, price position, promotion response, stockouts, and calendar patterns.",
        "level3": "Feature engineering adds margin, discount, calendar, price-position, and stockout fields. Aggregations retain observations, quantities, revenue, costs, profit, and rates. Correlation outputs are descriptive only and are not passed as causal price sensitivity. Figures and CSV summaries are reproducible from the validated dataset.",
        "io": ("Validated retail dataset", "KPI tables, segment summaries, correlations, and commercial figures"),
        "controls": "Stable grouping keys, safe divisions, transparent aggregation definitions, descriptive-versus-causal separation.",
    },
    {
        "title": "03 - Descriptive price elasticity",
        "purpose": "Measure the observed relationship between price and demand and provide diagnostics for the causal stage.",
        "level1": "Estimates how much demand moved when price changed in the historical data.",
        "level2": "Fits log-log regressions by category and product, classifies elastic versus inelastic response, reports confidence and significance, shrinks noisy product estimates toward category baselines, and simulates price scenarios.",
        "level3": "The coefficient on log price is elasticity: a value of -1.5 means a 1% price rise is associated with about a 1.5% demand decline. Controls reduce observed confounding but do not solve price endogeneity. Product estimates carry status, observations, standard errors, confidence intervals, p-values, reliability, and shrinkage weight. This stage is diagnostic; executable pricing requires the causal gate.",
        "io": ("Clean demand, price, product, category, store, promotion, and calendar fields", "Category/product elasticity estimates and scenario simulations"),
        "controls": "Minimum observations and price variation, finite logs, confidence diagnostics, plausible-sign flags, no direct executable pricing.",
    },
    {
        "title": "04 - Causal price analysis",
        "purpose": "Estimate demand response caused by price rather than price changes that reacted to expected demand.",
        "level1": "Checks whether the price effect is credible enough to support a recommendation.",
        "level2": "Uses two-stage least squares: cost-side instruments predict price in the first stage, then instrumented price estimates demand response in the second. It produces pooled and product-level estimates with strength and reliability diagnostics.",
        "level3": "The endogenous regressor is log selling price; the outcome is log units. Declared observed cost instruments include supplier and shipping indices, while exogenous controls and store effects account for measured context. Product models drop constant controls to avoid singular matrices. Reliability requires sufficient observations, an economically plausible sign, statistical support, and a strong first stage. Missing instruments cause an explicit failure; they are never fabricated.",
        "io": ("Validated price, demand, controls, and observed instruments", "IV estimates, OLS comparison, first-stage diagnostics, reliability fields"),
        "controls": "Observed-instrument requirement, first-stage F and partial R-squared, confidence intervals, per-product failure rows, causal provenance.",
    },
    {
        "title": "05 - Promotion uplift and readiness",
        "purpose": "Estimate incremental demand from promotions without treating normal buyers or leaked treatment outcomes as uplift.",
        "level1": "Determines whether a promotion is likely to create extra demand and whether the evidence is safe to use in pricing.",
        "level2": "Builds a promotion propensity model, restricts analysis to common support, estimates an inverse-propensity weighted treatment effect, trains a two-model uplift learner, evaluates ranking on future dates, and produces segment summaries and opportunities.",
        "level3": "Only pretreatment features enter the model; selling price, competitor price, inventory, and stockout outcomes are excluded. Evaluation uses complete-date holdout partitions. Readiness combines propensity overlap, observed stockout censoring, ranking direction, and configured thresholds. Model scores can be stored for analysis even when the pricing-eligibility flag is false. Step 08 then applies zero promotion uplift.",
        "io": ("Promotion flag, pretreatment product/store/calendar/context features, demand outcome", "Propensity diagnostics, uplift predictions, deciles, metrics, readiness flag"),
        "controls": "Pretreatment-only schema, common support, date holdout, censoring threshold, ranking-quality gate, action-eligibility flag.",
    },
    {
        "title": "06 - Demand forecasting",
        "purpose": "Create the next-period quantity baseline against which every candidate price is compared.",
        "level1": "Forecasts how many units are expected if the current pricing plan continues.",
        "level2": "Creates lags, rolling histories, price/promotion/calendar features, splits complete dates into train, validation, and test periods, tunes and trains XGBoost, compares it with naive forecasts, and generates one future row per store-product decision unit.",
        "level3": "All rows sharing a calendar date remain in one partition. Lags and rolling means are backward looking. Internal model selection uses only training/validation dates, while the final report uses a later untouched test block. Metrics include MAE, RMSE, R-squared, MAPE, WAPE, and SMAPE. Next-period construction advances the date explicitly, clears stale promotions, uses trailing or known context, and records forecast provenance.",
        "io": ("Clean chronological demand history and known/trailing covariates", "Test predictions, metrics, baselines, feature importance, next-period forecast"),
        "controls": "Unique-date splits, no future-derived features, naive benchmarks, explicit forecast date/source, complete store-product coverage.",
    },
    {
        "title": "07 - Time-safe market evidence",
        "purpose": "Convert recent competitor and customer notes into structured context without using future information or letting text directly choose a price.",
        "level1": "Adds relevant recent market context beside each next-period decision.",
        "level2": "Cleans and chunks notes, indexes them with TF-IDF, retrieves the most relevant evidence for each decision row, extracts structured signals, and merges them with the forecast dataset.",
        "level3": "Eligibility requires note date <= decision date, a configured 120-day lookback, and exact normalized category and region matches when metadata is supplied. Similarity filtering and top-k retrieval limit evidence volume. Structured fields summarize sentiment, competitor position, sensitivity, promotion interest, trends, counts, and evidence dates. A bounded weighted market adjustment may affect demand; free text never selects price directly. Variance diagnostics flag constant features.",
        "io": ("Dated market notes plus next-period decision rows", "Document index, retrieved evidence, structured features, coverage and configuration"),
        "controls": "As-of-time boundary, lookback, exact metadata filters, minimum similarity, bounded adjustment, evidence provenance.",
    },
    {
        "title": "08 - Price optimization",
        "purpose": "Select the best feasible next-period price under the configured commercial objective and guardrails.",
        "level1": "Compares allowed prices and recommends raise, hold, lower, or not scored.",
        "level2": "Combines forecast baseline, causal elasticity, unit cost, inventory, competitor context, eligible promotion uplift, market adjustment, and constraints. It estimates candidate demand, revenue, and profit, rejects infeasible candidates, selects the strongest objective value, and produces item and portfolio summaries.",
        "level3": "Demand follows Q(P)=Q0*(P/P0)^e. Realized sales are capped at available inventory in cap mode. Profit is (P-cost)*realized sales minus promotion cost. Current and candidate cases use identical inventory and promotion semantics. Causal elasticity precedence is explicit and non-causal rows hold by default. Cross-price effects are excluded until a true joint portfolio solver is used. If the current price is feasible, no method may return a worse configured objective.",
        "io": ("Forecasts, causal elasticity, cost, inventory, competitor, promotion readiness, market features, policy", "Recommendations, action/category/portfolio summaries, sensitivity scenarios"),
        "controls": "Price bounds, margin, quantity, profit, competitor, inventory, implementation step, causal gate, no-harm fallback.",
    },
    {
        "title": "Independent recommendation validation",
        "purpose": "Recompute the selected decision outside the optimization search and make any control failure visible before review.",
        "level1": "Checks that every priced row really follows all configured rules.",
        "level2": "Recalculates price movement, demand response, margin, profit, competitor position, inventory treatment, implementation caps, evidence status, and objective performance for the final selected price.",
        "level3": "Validation distinguishes priced rows from explicit holds. Each control receives its own boolean field, while `meets_all_constraints` is the conjunction for scored rows. Tolerances handle numerical rounding without masking material failures. The validator verifies objective no-harm against a feasible current-price baseline and summarizes total, scored, held, passed, and failed rows. Release logic should stop when a scored row fails.",
        "io": ("Final recommendation rows plus the same policy and source evidence", "Row-level control fields and portfolio validation summary"),
        "controls": "Independent recomputation, explicit not-scored count, per-rule status, pass-rate summary, release-stop signal.",
    },
    {
        "title": "Browser decision workbench",
        "purpose": "Give commercial users an interactive surface to load data, configure policy, compare methods, review decisions, and export results.",
        "level1": "A local browser application for running and explaining retail pricing scenarios.",
        "level2": "The React/Vite workbench presents data, configuration, recommendations, portfolio KPIs, action queues, scenario comparisons, category contributions, and evidence/constraint status. Users can switch among six optimization techniques under one common policy.",
        "level3": "The JavaScript engine independently implements the demand and economics model. Methods include grid, closed form, robust, Bayesian, multi-objective, and Lagrangian approaches. Shared fixtures enforce parity with Python for candidate demand, price grids, and rounding. Technique validation requires plausible economics, bounds, feasibility, no harm, and distinct price vectors. Manual rows have non-causal provenance and are held unless the policy explicitly changes.",
        "io": ("Packaged demo data or validated CSV plus user configuration", "Interactive decisions, summaries, scenarios, CSV and PDF exports"),
        "controls": "Causal provenance gate, shared guardrails, cross-language parity, method distinctness, explicit error handling.",
    },
    {
        "title": "Live scalable experiments and hybrid execution",
        "purpose": "Run real optimization workloads of different sizes, expose their execution process, and compare or combine methods under one frozen configuration.",
        "level1": "Generates live examples, performs substantial real analysis, shows each computation stage and keeps every completed run available for review.",
        "level2": "The application opens on the Deep Analysis Runner. Users select Small (50), Medium (250), Large (1,000), or 5-2,000 custom units; choose a scenario profile, analysis depth and methods; and run comparison or hybrid execution. The six-stage monitor follows Start/profile, Optimize/refine, Stress test, Simulate risk, Validate, and Finish. It shows exact engine counters, measured time, events, charts, stress/risk tables, consensus, controls, top decisions and history.",
        "level3": "`scenarioGenerator.js` creates deterministic causal-provenance rows. `deepAnalysis.js` freezes a workload plan. Deep uses 1%, 0.5% and 0.25% grids, size-adjusted shocked-market reruns and 1,000 draws per method; Research adds 0.125%, more stress states and 3,000 draws. Stress states perturb demand, cost, elasticity, inventory, competitor price and market signal, then fully re-optimize every selected technique. Monte Carlo keeps each recommendation fixed while drawing portfolio and item uncertainty, producing P05/P50/P95, volatility and positive-uplift probability. `engine.js` directly counts portfolio optimizations, candidate evaluations, demand/profit evaluations and capacity iterations. Worker yields occur only after completed work; no delay is inserted. Release QA measured Deep all-method cases at 5.4 seconds/16.2M candidates for 50 units, 9.2 seconds/28.6M for 250, and 26.9 seconds/62.7M for 1,000 on the build environment. The final record retains convergence, all stress scenarios, risk percentiles, consensus, counters, method/hybrid decisions, configuration and timing; JSON and embedded-font PDF exports support review.",
        "io": ("Generated size/profile/seed, analysis depth, selected techniques, active policy", "Live computation stages, exact work counts, stress/risk distributions, decisions, JSON and PDF reports"),
        "controls": "Seed reproducibility, frozen workload, identical comparison input, exact instrumentation, worker isolation, row-level validation, deterministic hybrid tie-break, cancellable execution.",
    },
    {
        "title": "CSV import, reporting, and provenance",
        "purpose": "Move data and decisions in and out of the workbench without corrupting fields or losing the evidence behind a price.",
        "level1": "Imports retail rows and exports review-ready CSV and PDF decision packages.",
        "level2": "The importer supports quoted CSV values and pipeline column aliases, validates required numerics and business relationships, and labels data origin. Exports include recommendation, scenario, category, configuration, control, and governance sections.",
        "level3": "The CSV parser handles escaped quotes, embedded commas, carriage returns, and blank lines. Aliases map pipeline forecast, cost, inventory, competitor, elasticity, promotion, market, and provenance fields into the browser schema. Validation rejects non-finite or impossible values and upward-sloping actionable elasticity. PDF libraries load only when requested, keeping the main bundle smaller. Exports preserve causal source, forecast source, promotion readiness, market evidence, status, and constraint flags.",
        "io": ("CSV files or in-memory decisions and configuration", "Validated browser rows, CSV recommendations, two-page decision PDF"),
        "controls": "Quote-aware parsing, aliases, strict numeric checks, provenance defaults, export error handling, lazy PDF loading.",
    },
    {
        "title": "Testing, build, and operational assurance",
        "purpose": "Detect mathematical, behavioural, integration, reporting, and dependency regressions before the application is shared.",
        "level1": "Runs automated checks for the analytical pipeline and browser application.",
        "level2": "Python tests cover configuration, features, models, optimization, plots, text processing, validation, and known regressions. JavaScript tests exercise all six techniques, controls, advanced features, CSV behavior, provenance, method differentiation, multi-size generation, input profiling, live execution, hybrid selection, and Python parity. The frontend production build and dependency audit complete the release check.",
        "level3": "The current suite contains 358 Python tests. Regression cases include complete-date isolation, time-safe retrieval, instrument availability, known-truth elasticity recovery, inventory-baseline consistency, non-causal hold behavior, and objective no harm. Browser checks generate 50, 250, and 1,000 unit scenarios reproducibly; execute multiple techniques; validate timing and row controls; and confirm the hybrid method mix covers every item. Manual benchmarks run all six methods plus hybrid on Small, Medium, and Large cases. Shared fixtures compare four candidate cases, three grids, and four rounding cases. `npm run build` verifies bundling, Web Worker, and dynamic PDF chunks; `npm audit` checks the lockfile dependency graph.",
        "io": ("Source code, fixtures, synthetic data, package lock", "Pass/fail evidence, parity output, production bundle, audit status"),
        "controls": "Fail-fast exit codes, committed fixtures, behavioural assertions, known-truth tests, build and dependency checks.",
    },
]


def layered_page(item, index, total):
    level_rows = []
    for label, body, bg, border in [
        ("LEVEL 1 - AT A GLANCE", item["level1"], PALE_BLUE, BLUE),
        ("LEVEL 2 - HOW IT WORKS", item["level2"], PALE_GREEN, GREEN),
        ("LEVEL 3 - FULL TECHNICAL DETAIL", item["level3"], PALE_AMBER, AMBER),
    ]:
        content = [p(label, "LayerTitle"), p(body, "LayerBody")]
        box = Table([[content]], colWidths=[174 * mm])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), .7, border),
            ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        level_rows += [box, Spacer(1, 3.2 * mm)]
    return title(item["title"], f"Functionality {index} of {total}") + [p(item["purpose"])] + level_rows + [
        table([
            ["Inputs", "Outputs"],
            [item["io"][0], item["io"][1]],
        ], [87 * mm, 87 * mm]),
        Spacer(1, 3 * mm), callout("<b>Primary controls:</b> " + item["controls"], "blue"), PageBreak(),
    ]


def build_technical():
    doc = SimpleDocTemplate(str(TECHNICAL_PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=22 * mm, bottomMargin=18 * mm,
                            title="ELASTIQ Technical System Guide", author="ELASTIQ", subject="Layered technical reference for the retail pricing decision studio")
    s = []
    s += [Spacer(1, 55 * mm), p("TECHNICAL SYSTEM GUIDE", "CoverKicker"), p("ELASTIQ retail pricing decision studio", "CoverTitle"),
          p("A step-by-step reference for analysts, data scientists, engineers, and technical stakeholders. Every functionality is explained in three layers: at a glance, how it works, and full technical detail.", "CoverSub"),
          Spacer(1, 15 * mm), callout("Use Level 1 for orientation, Level 2 for implementation understanding, and Level 3 for model, data, control, and engineering details.", "blue"), PageBreak()]

    s += title("System orientation", "How to use this guide")
    s += [Image(str(PIPELINE_IMG), width=174 * mm, height=56 * mm),
          p("The system has a Python evidence pipeline and a React/Vite decision workbench. The pipeline produces validated model evidence. The workbench allows reviewed scenario execution, comparison, and export. The final price is selected only after causal, forecast, feasibility, and validation controls are applied."),
          table([
              ["Layer", "Purpose", "Main implementation"],
              ["Source and validation", "Establish reliable analytical inputs", "CSV sources, pandas, schema and business-rule checks"],
              ["Evidence models", "Estimate causal price response, promotion readiness, and future demand", "statsmodels, linearmodels, scikit-learn, XGBoost"],
              ["Market context", "Structure recent relevant unstructured evidence", "TF-IDF retrieval, metadata and date filters"],
              ["Decision engine", "Evaluate feasible prices and commercial objectives", "NumPy/SciPy Python engine and independent JavaScript engine"],
              ["Review surface", "Configure, compare, explain, and export", "React, Vite, CSV, jsPDF"],
              ["Assurance", "Detect regressions and unsafe outputs", "pytest, browser validation suite, parity fixtures, build and audit"],
          ], [36 * mm, 68 * mm, 70 * mm]),
          callout("Design rule: evidence and decision are separated. A model may produce a score, but only a passed readiness gate may allow that score to affect price.", "green"), PageBreak()]

    s += title("Repository map and execution order", "Developer orientation")
    s += [table([
        ["Location", "Responsibility"],
        ["start.bat / start.sh", "One-click startup and optional full data refresh"],
        ["scripts/run_pipeline.py", "Cross-platform ordered execution of stages 01 through 08"],
        ["src/pipelines", "Stage orchestration, input/output wiring, and saved artifacts"],
        ["src/models", "Elasticity, causal, uplift, forecasting, and retrieval logic"],
        ["src/optimization", "Demand response, constraints, price search, and decision validation"],
        ["src/data / src/features", "Splitting, validation, cleaning, aliases, and feature construction"],
        ["app/frontend/src", "React workbench, JavaScript engine, live lab, input profiler, reports, and demo data"],
        ["app/frontend/src/workers", "Isolated live experiment execution and progress events"],
        ["tests", "Python unit, regression, plotting, validation, and parity fixtures"],
        ["data/processed", "Stage outputs and final recommendation evidence"],
        ["output/pdf", "Share-ready high-level and technical documentation"],
    ], [57 * mm, 117 * mm]),
          p("Execution is strictly ordered because each stage consumes saved products of an earlier stage. The final stage checks its upstream dependencies before optimization. Browser data can be refreshed only after the Python outputs and parity fixture are regenerated."),
          callout("Normal team use does not require Python. Run the standard launcher with packaged data. Use --refresh-data only when the analytical outputs must be recomputed.", "amber"), PageBreak()]

    for i, item in enumerate(FUNCTIONS, start=1):
        s += layered_page(item, i, len(FUNCTIONS))

    s += title("Core schemas and decision provenance", "Reference")
    s += [table([
        ["Domain", "Required or important fields", "Purpose"],
        ["Decision identity", "decision_date, store_id, product_id, category, region", "Defines the store-product-time unit and retrieval filters"],
        ["Commercial baseline", "current_price, unit_cost, baseline_quantity, available_inventory", "Defines current economics and next-period capacity"],
        ["Causal evidence", "elasticity, elasticity_source, elasticity_is_causal, first_stage_f", "Defines demand response and whether pricing is allowed"],
        ["Forecast evidence", "forecast_quantity, forecast_date, forecast_source", "Defines Q0 at P0 for the next period"],
        ["Promotion evidence", "promotion_uplift, promotion_readiness, promotion_cost", "Allows or suppresses incremental demand and cost"],
        ["Market evidence", "market_adjustment, evidence_count, evidence dates/filters", "Provides bounded, time-safe context"],
        ["Policy", "objective, price bands, margin, quantity, profit, competitor, inventory, step limits", "Defines the feasible decision set"],
        ["Decision", "recommended_price, expected_quantity, revenue, profit, deltas, action", "Communicates the selected scenario"],
        ["Audit", "status, reason, per-constraint flags, meets_all_constraints", "Preserves failure behavior and release readiness"],
    ], [37 * mm, 75 * mm, 62 * mm], small=True),
          p("Every exported row should be self-describing: what was recommended, what baseline and model evidence supported it, which policy was active, which constraints bound, and whether independent validation passed."), PageBreak()]

    s += title("Mathematical reference", "Core equations")
    s += [p("<b>Constant-elasticity demand</b>"), p("Q(P) = Q0 x (P / P0)^e, where Q0 is the next-period baseline at current price P0 and e is causal price elasticity."),
          p("<b>Inventory realization</b>"), p("Q_realized(P) = min(Q(P), available inventory) in cap mode. Constraint mode instead rejects scenarios whose unconstrained demand exceeds the permitted inventory condition."),
          p("<b>Revenue and profit</b>"), p("Revenue(P) = P x Q_realized(P). Profit(P) = (P - unit cost) x Q_realized(P) - promotion cost."),
          p("<b>Two-stage least squares</b>"), p("Stage 1 predicts log price from observed instruments and controls. Stage 2 regresses log quantity on predicted log price and controls. The log-price coefficient is the IV elasticity."),
          p("<b>Inverse propensity weighting</b>"), p("Promotion observations receive weights based on treatment propensity so the weighted promoted and control groups are more comparable inside common support."),
          p("<b>Forecast metrics</b>"), p("MAE averages absolute unit error. RMSE emphasizes large errors. MAPE averages row-level percentage error. WAPE divides total absolute error by total actual demand. SMAPE symmetrizes percentage error."),
          p("<b>No-harm rule</b>"), p("When the current price is feasible, objective(recommended) must be greater than or equal to objective(current), within numerical tolerance."),
          callout("Point estimates are used for the packaged demonstration. A live deployment should propagate forecast and elasticity uncertainty into downside-aware objectives and monitoring.", "amber"), PageBreak()]

    s += title("Operational runbook", "Start, validate, stop, and troubleshoot")
    s += [table([
        ["Task", "Windows", "Linux/macOS"],
        ["Start with packaged data", "Double-click start.bat", "chmod +x start.sh; ./start.sh"],
        ["Refresh models and start", "start.bat --refresh-data", "./start.sh --refresh-data"],
        ["Start without opening browser", "start.bat --no-browser", "./start.sh --no-browser"],
        ["Stop", "Press Ctrl+C in the command window", "Press Ctrl+C in the terminal"],
        ["Python tests", "python -m pytest", "python3 -m pytest"],
        ["Browser tests", "cd app\\frontend && npm test", "cd app/frontend && npm test"],
        ["Production build", "cd app\\frontend && npm run build", "cd app/frontend && npm run build"],
    ], [50 * mm, 62 * mm, 62 * mm]),
          p("First-launch installation requires access to the npm registry. Data refresh also requires Python package installation. Runtime caches and the Python environment remain inside `.runtime`, which can be removed to force a clean setup."),
          table([
              ["Symptom", "Likely cause", "Resolution"],
              ["Node.js is not installed", "Node/npm unavailable on PATH", "Install Node.js 18+ and reopen the terminal"],
              ["Port 5173 is busy", "Another Vite process is running", "Stop the earlier terminal or use the displayed alternate port"],
              ["Python refresh fails", "Python or build dependency missing", "Run without --refresh-data or install Python 3.10+ and retry"],
              ["CSV is rejected", "Missing, nonnumeric, impossible, or non-causal fields", "Review the import error and correct the identified column"],
              ["Item is held", "Evidence or feasibility gate failed", "Read status/reason; do not force a price without resolving the evidence"],
          ], [43 * mm, 58 * mm, 73 * mm], small=True), PageBreak()]

    s += title("Release and governance checklist", "Team handoff")
    s += [table([
        ["Area", "Ready condition"],
        ["Application", "One-click launcher starts locally; browser engine validation and production build pass"],
        ["Data", "Source contract, freshness, row counts, dates, critical rules, and lineage pass"],
        ["Causal model", "Observed instruments, sufficient strength, plausible sign, stability, and business exclusion logic pass"],
        ["Forecast", "Unique-date evaluation, naive comparison, segment errors, freshness, and drift thresholds pass"],
        ["Promotion", "Overlap, censoring, ranking, and incremental-value thresholds pass before uplift is enabled"],
        ["Market evidence", "As-of-time, lookback, metadata matching, variance, and provenance pass"],
        ["Optimization", "Policy approved; current/candidate semantics match; no-harm and independent audit pass"],
        ["Operations", "Roles, approvals, audit retention, monitoring, alerting, rollback, and incident owners are defined"],
        ["Pilot", "Eligible scope and holdout are predefined; realized customer and financial outcomes are measured"],
    ], [48 * mm, 126 * mm]),
          callout("The packaged system is ready for demonstration, training, reviewed scenario analysis, and controlled pilot preparation. Direct live price publishing requires the operational controls listed above.", "green"),
          Spacer(1, 10 * mm), HRFlowable(width="100%", thickness=.7, color=LINE), Spacer(1, 4 * mm), p("ELASTIQ Technical System Guide | Final team document", "Small")]
    doc.build(s, onFirstPage=cover, onLaterPages=page_frame)


def main():
    build_overview()
    build_technical()
    print(OVERVIEW_PDF)
    print(TECHNICAL_PDF)


if __name__ == "__main__":
    main()
