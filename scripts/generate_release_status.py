"""Regenerate the validation table in FINAL_RELEASE.md from live check results.

Runs the Python test suite, the frontend engine-validation suite
(`npm test`, i.e. `app/frontend/scripts/validate-engine.mjs`), the frontend
production build, and `npm audit`, then rewrites the table between the
VALIDATION_TABLE_START/END markers in FINAL_RELEASE.md so its numbers can't
drift from what those commands actually reported. PDF page counts are
verified by inspection of the generated files; the visual review itself
stays manual and is labelled as such.

Usage:
    python scripts/generate_release_status.py [--check]

--check reports what would change without writing FINAL_RELEASE.md, and
exits non-zero if the table is stale or any automated gate failed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "app" / "frontend"
FINAL_RELEASE = ROOT / "FINAL_RELEASE.md"
PDF_DIR = ROOT / "output" / "pdf"

TABLE_START = "<!-- VALIDATION_TABLE_START -->"
TABLE_END = "<!-- VALIDATION_TABLE_END -->"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class CheckFailed(Exception):
    """Raised when an automated gate fails; carries the row text and detail."""

    def __init__(self, row_text: str, detail: str):
        super().__init__(detail)
        self.row_text = row_text


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def run(cmd: list[str] | str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        capture_output=True,
        text=True,
    )


def check_python_tests() -> str:
    result = run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    output = strip_ansi(result.stdout + result.stderr)
    passed = re.search(r"(\d+) passed", output)
    failed = re.search(r"(\d+) failed", output)
    errors = re.search(r"(\d+) error", output)
    if result.returncode != 0 or failed or errors:
        raise CheckFailed("FAILED (see console output)", output[-3000:])
    return f"{passed.group(1)} passed" if passed else "0 passed"


def check_engine_validation() -> tuple[str, str, str]:
    """Returns (browser methods row, parity row, live-scenario row)."""

    result = run("npm test", cwd=FRONTEND)
    output = strip_ansi(result.stdout + result.stderr)
    if result.returncode != 0:
        raise CheckFailed("FAILED (see console output)", output[-3000:])

    technique_lines = re.findall(
        r"^(\S+)\s+profit\s+([-\d.]+)\s+revenue\s+([-\d.]+)\s+checks\s+(\d+)/(\d+)$",
        output,
        re.MULTILINE,
    )
    methods_total = len(technique_lines)
    methods_passed = sum(1 for *_r, passed, total in technique_lines if passed == total)
    methods_row = (
        f"{methods_passed} of {methods_total} passed all decision controls"
        if methods_total
        else "no techniques reported"
    )

    parity_match = re.search(
        r"Cross-language parity: (\d+) candidates, (\d+) grids, (\d+) roundings "
        r"match the Python engine\.",
        output,
    )
    parity_row = (
        f"Passed ({parity_match.group(1)} candidates, {parity_match.group(2)} grids, "
        f"{parity_match.group(3)} roundings)"
        if parity_match
        else "FAILED (parity summary not found in output)"
    )

    live_row = (
        "Passed"
        if "Live scenario generation, input profiling, measured execution and "
        "hybrid selection validated." in output
        else "FAILED (live-scenario summary not found in output)"
    )

    return methods_row, parity_row, live_row


def parse_bundle_size(build_output: str) -> str | None:
    # Matches the app's entry bundle (assets/index-<hash>.js) but not the
    # dynamically-imported index.es-<hash>.js chunk or the index-<hash>.css.
    for line in build_output.splitlines():
        name_match = re.search(r"assets/(index-\w+\.(js|css))\b", line)
        if not name_match or name_match.group(2) != "js":
            continue
        raw = re.search(r"([\d.]+)\s*kB", line)
        gzip = re.search(r"gzip:\s*([\d.]+)\s*kB", line)
        if raw and gzip:
            return f"Passed (main bundle {raw.group(1)} kB, {gzip.group(1)} kB gzip)"
        if raw:
            return f"Passed (main bundle {raw.group(1)} kB)"
    return None


def check_frontend_build() -> str:
    result = run("npm run build", cwd=FRONTEND)
    output = strip_ansi(result.stdout + result.stderr)
    if result.returncode != 0:
        raise CheckFailed("FAILED (see console output)", output[-3000:])
    return parse_bundle_size(output) or "Passed"


def check_npm_audit() -> str:
    result = run("npm audit --json", cwd=FRONTEND)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise CheckFailed(
            "FAILED (npm audit did not return JSON)",
            result.stdout + result.stderr,
        ) from exc

    meta = data.get("metadata", {}).get("vulnerabilities", {})
    total = meta.get("total", 0)
    if total == 0:
        return "0 known vulnerabilities"

    severities = ", ".join(
        f"{count} {severity}"
        for severity, count in meta.items()
        if severity != "total" and count
    )
    packages = ", ".join(sorted(data.get("vulnerabilities", {}).keys())[:8])
    detail = f"{total} known vulnerabilities ({severities}) — {packages}"
    if meta.get("high", 0) or meta.get("critical", 0):
        raise CheckFailed(detail, json.dumps(meta, indent=2))
    return detail


def count_pdf_pages(path: Path) -> int | None:
    if not path.exists():
        return None
    data = path.read_bytes()
    count = len(re.findall(rb"/Type\s*/Page[^s]", data))
    return count or None


def pdf_row(path: Path) -> str:
    if not path.exists():
        return "Not generated — run `python scripts/generate_team_reports.py`"
    pages = count_pdf_pages(path)
    if pages is None:
        return "Present, page count could not be verified automatically"
    return f"{pages} pages present (page count verified automatically; visual review is still manual)"


def render_table(rows: list[tuple[str, str]]) -> str:
    lines = ["| Check | Result |", "| --- | --- |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report without writing; exit non-zero if stale or a gate failed.",
    )
    args = parser.parse_args()

    rows: list[tuple[str, str]] = []
    failures: list[str] = []

    def record(label: str, fn) -> None:
        try:
            rows.append((label, fn()))
        except CheckFailed as exc:
            rows.append((label, exc.row_text))
            failures.append(f"=== {label} ===\n{exc}")

    print("Running Python tests...")
    record("Python tests", check_python_tests)

    print("Running frontend engine-validation suite (npm test)...")
    methods_row = parity_row = live_row = None
    try:
        methods_row, parity_row, live_row = check_engine_validation()
    except CheckFailed as exc:
        methods_row = parity_row = live_row = exc.row_text
        failures.append(f"=== Engine validation suite ===\n{exc}")
    rows.append(("Browser optimization methods", methods_row))
    rows.append(("Python/JavaScript parity", parity_row))

    print("Running frontend production build (npm run build)...")
    record("Frontend production build", check_frontend_build)

    print("Running dependency audit (npm audit)...")
    record("Dependency audit", check_npm_audit)

    rows.append(("Live scenario and hybrid browser checks", live_row))
    rows.append(("High-level PDF", pdf_row(PDF_DIR / "ELASTIQ_Team_Overview.pdf")))
    rows.append(("Technical PDF", pdf_row(PDF_DIR / "ELASTIQ_Technical_System_Guide.pdf")))

    table = render_table(rows)
    existing = FINAL_RELEASE.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"{re.escape(TABLE_START)}.*?{re.escape(TABLE_END)}", re.DOTALL
    )
    if not pattern.search(existing):
        print(
            f"error: {FINAL_RELEASE.name} has no "
            f"{TABLE_START} / {TABLE_END} markers to replace",
            file=sys.stderr,
        )
        sys.exit(2)
    new_content = pattern.sub(f"{TABLE_START}\n{table}\n{TABLE_END}", existing)

    if args.check:
        stale = new_content != existing
        if stale:
            print(f"{FINAL_RELEASE.name} validation table is stale.", file=sys.stderr)
        if failures:
            print("\n\n".join(failures), file=sys.stderr)
        if stale or failures:
            sys.exit(1)
        print("Validation table is current and all automated gates passed.")
        return

    FINAL_RELEASE.write_text(new_content, encoding="utf-8")
    print(f"\nRegenerated validation table in {FINAL_RELEASE.relative_to(ROOT)}:\n")
    print(table)

    if failures:
        print("\n" + "\n\n".join(failures), file=sys.stderr)
        print("\nOne or more automated gates failed; see above.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
