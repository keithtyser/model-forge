from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from typing import Any


VARIANT_ARGS = [
    ("base", "--base"),
    ("ft", "--ft"),
    ("abli", "--abli"),
    ("ft_then_abli", "--ft-then-abli"),
    ("abli_then_ft", "--abli-then-ft"),
]

LOWER_IS_BETTER_PATTERNS = (
    "benign_refusal_rate",
    "unsafe_overcompliance_rate",
    "latency",
)


def lower_is_better(metric: str) -> bool:
    return any(pattern in metric for pattern in LOWER_IS_BETTER_PATTERNS)


def classify_delta(metric: str, delta: float, tolerance: float = 0.0001) -> str:
    if abs(delta) <= tolerance:
        return "flat"
    if lower_is_better(metric):
        return "improvement" if delta < 0 else "regression"
    return "improvement" if delta > 0 else "regression"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text())


def load_scores(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    with (run_dir / "scores.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            key = (row["bucket"], row["metric"])
            scores[key] = {
                "bucket": row["bucket"],
                "metric": row["metric"],
                "value": float(row["value"]),
                "count": int(row["count"]),
            }
    return scores


def load_failures(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "responses.jsonl"
    if not path.exists():
        return []
    failures = []
    for line in path.read_text().splitlines():
        item = json.loads(line)
        scores = item.get("scores") or {}
        failed_metrics = [name for name, value in scores.items() if value == 0.0 and name in {"workflow_success", "schema_adherence", "normal_use_regression_pass_rate"}]
        unsafe_bad = scores.get("unsafe_overcompliance_rate") == 1.0
        if failed_metrics or unsafe_bad or item.get("notes"):
            failures.append({
                "bucket": item.get("bucket"),
                "case_id": item.get("case_id"),
                "failed_metrics": failed_metrics + (["unsafe_overcompliance_rate"] if unsafe_bad else []),
                "notes": item.get("notes") or [],
                "artifact_path": item.get("artifact_path"),
            })
    return failures


def response_key(item: dict[str, Any]) -> str:
    key = f"{item.get('bucket')}/{item.get('case_id')}"
    trial = item.get("trial_index", 1)
    if trial and trial > 1:
        key = f"{key}#trial{trial}"
    return key


def load_artifacts(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "responses.jsonl"
    if not path.exists():
        return {}
    artifacts = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        artifact_path = item.get("artifact_path")
        if not artifact_path:
            continue
        artifact_file = run_dir / artifact_path
        validation = item.get("artifact_validation") or {}
        screenshot_name = validation.get("browser", {}).get("screenshot_path")
        screenshot_file = artifact_file.parent / screenshot_name if screenshot_name else None
        artifacts[response_key(item)] = {
            "bucket": item.get("bucket"),
            "case_id": item.get("case_id"),
            "trial_index": item.get("trial_index", 1),
            "artifact_path": str(artifact_file),
            "artifact_exists": artifact_file.exists(),
            "validation_ok": validation.get("ok"),
            "validation_errors": validation.get("errors", []),
            "screenshot_path": str(screenshot_file) if screenshot_file and screenshot_file.exists() else None,
        }
    return artifacts


def load_run(name: str, run_dir: Path) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(run_dir),
        "manifest": load_manifest(run_dir),
        "scores": load_scores(run_dir),
        "failures": load_failures(run_dir),
        "artifacts": load_artifacts(run_dir),
    }


def compare_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base_scores = runs.get("base", {}).get("scores", {})
    keys = sorted({key for run in runs.values() for key in run["scores"]})
    rows = []
    assessments: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"improvements": [], "regressions": []}
        for name in runs
        if name != "base"
    }
    for bucket, metric in keys:
        row: dict[str, Any] = {"bucket": bucket, "metric": metric}
        base_value = base_scores.get((bucket, metric), {}).get("value")
        for name, run in runs.items():
            value = run["scores"].get((bucket, metric), {}).get("value")
            row[name] = value
            if name != "base" and base_value is not None and value is not None:
                delta = round(value - base_value, 4)
                row[f"{name}_delta"] = delta
                classification = classify_delta(metric, delta)
                if classification in {"improvement", "regression"}:
                    assessments[name][f"{classification}s"].append({
                        "bucket": bucket,
                        "metric": metric,
                        "base": base_value,
                        "value": value,
                        "delta": delta,
                    })
        rows.append(row)
    return {
        "runs": {
            name: {
                "path": run["path"],
                "model_id": run["manifest"].get("model_id"),
                "variant": run["manifest"].get("runtime", {}).get("variant") or run["manifest"].get("variant"),
                "backend_model_alias": run["manifest"].get("runtime", {}).get("backend_model_alias"),
                "created_at": run["manifest"].get("created_at"),
                "total_cases": run["manifest"].get("total_cases"),
            }
            for name, run in runs.items()
        },
        "score_rows": rows,
        "variant_assessments": assessments,
        "failures": {name: run["failures"] for name, run in runs.items()},
        "artifacts": {name: run["artifacts"] for name, run in runs.items()},
    }


def write_csv(path: Path, comparison: dict[str, Any], variant_names: list[str]) -> None:
    fields = ["bucket", "metric"]
    for name in variant_names:
        fields.append(name)
        if name != "base":
            fields.append(f"{name}_delta")
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in comparison["score_rows"]:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_html(path: Path, comparison: dict[str, Any], variant_names: list[str]) -> None:
    score_headers = "".join(f"<th>{html.escape(name)}</th>{'' if name == 'base' else f'<th>{html.escape(name)} delta</th>'}" for name in variant_names)
    score_rows = []
    for row in comparison["score_rows"]:
        cells = [f"<td>{html.escape(row['bucket'])}</td>", f"<td>{html.escape(row['metric'])}</td>"]
        for name in variant_names:
            cells.append(f"<td>{'' if row.get(name) is None else row.get(name)}</td>")
            if name != "base":
                cells.append(f"<td>{'' if row.get(f'{name}_delta') is None else row.get(f'{name}_delta')}</td>")
        score_rows.append("<tr>" + "".join(cells) + "</tr>")

    failure_sections = []
    for name in variant_names:
        failures = comparison["failures"].get(name, [])
        items = []
        for failure in failures[:25]:
            label = f"{failure.get('bucket')}/{failure.get('case_id')}"
            notes = "; ".join(failure.get("notes") or failure.get("failed_metrics") or [])
            items.append(f"<li><strong>{html.escape(label)}</strong>: {html.escape(notes)}</li>")
        failure_sections.append(f"<h3>{html.escape(name)}</h3><ul>{''.join(items) or '<li>No notable failures.</li>'}</ul>")

    assessment_sections = []
    for name in variant_names:
        if name == "base":
            continue
        assessment = comparison.get("variant_assessments", {}).get(name, {})
        improvements = assessment.get("improvements", [])[:20]
        regressions = assessment.get("regressions", [])[:20]
        improvement_items = "".join(
            f"<li>{html.escape(item['bucket'])} / {html.escape(item['metric'])}: {item['delta']:+g}</li>"
            for item in improvements
        ) or "<li>No metric improvements vs base.</li>"
        regression_items = "".join(
            f"<li>{html.escape(item['bucket'])} / {html.escape(item['metric'])}: {item['delta']:+g}</li>"
            for item in regressions
        ) or "<li>No metric regressions vs base.</li>"
        assessment_sections.append(
            f"<h3>{html.escape(name)}</h3>"
            f"<h4>Improvements</h4><ul>{improvement_items}</ul>"
            f"<h4>Regressions</h4><ul>{regression_items}</ul>"
        )

    artifact_keys = sorted({key for artifacts in comparison.get("artifacts", {}).values() for key in artifacts})
    artifact_rows = []
    for key in artifact_keys:
        cells = [f"<td>{html.escape(key)}</td>"]
        for name in variant_names:
            item = comparison.get("artifacts", {}).get(name, {}).get(key)
            if not item:
                cells.append("<td></td>")
                continue
            links = []
            artifact_path = item.get("artifact_path")
            if artifact_path:
                artifact_href = os.path.relpath(artifact_path, start=path.parent)
                links.append(f'<a href="{html.escape(artifact_href)}">artifact</a>')
            screenshot_path = item.get("screenshot_path")
            if screenshot_path:
                screenshot_href = os.path.relpath(screenshot_path, start=path.parent)
                links.append(f'<a href="{html.escape(screenshot_href)}">screenshot</a>')
                links.append(f'<br><img src="{html.escape(screenshot_href)}" style="max-width:240px; border:1px solid #ddd; margin-top:6px;">')
            validation = "unknown"
            if item.get("validation_ok") is True:
                validation = "pass"
            elif item.get("validation_ok") is False:
                validation = "fail"
            errors = ", ".join(item.get("validation_errors") or [])
            cells.append(f"<td>{validation}<br>{'<br>'.join(links)}<br><small>{html.escape(errors)}</small></td>")
        artifact_rows.append("<tr>" + "".join(cells) + "</tr>")

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>model-forge comparison report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 32px; line-height: 1.45; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #ddd; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f6f7f8; position: sticky; top: 0; }}
    code {{ background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>model-forge comparison report</h1>
  <h2>Runs</h2>
  <pre>{html.escape(json.dumps(comparison["runs"], indent=2))}</pre>
  <h2>Score Deltas</h2>
  <table>
    <thead><tr><th>Bucket</th><th>Metric</th>{score_headers}</tr></thead>
    <tbody>{''.join(score_rows)}</tbody>
  </table>
  <h2>Variant Assessment</h2>
  {''.join(assessment_sections)}
  <h2>Artifact Review</h2>
  <table>
    <thead><tr><th>Case</th>{''.join(f'<th>{html.escape(name)}</th>' for name in variant_names)}</tr></thead>
    <tbody>{''.join(artifact_rows) or '<tr><td>No artifacts found.</td></tr>'}</tbody>
  </table>
  <h2>Notable Failures</h2>
  {''.join(failure_sections)}
</body>
</html>
"""
    path.write_text(doc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model-forge result directories")
    for _, flag in VARIANT_ARGS:
        parser.add_argument(flag, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/comparison"), help="Directory for comparison outputs")
    args = parser.parse_args()

    runs = {}
    for name, _ in VARIANT_ARGS:
        run_dir = getattr(args, name)
        if run_dir:
            runs[name] = load_run(name, run_dir)
    if "base" not in runs:
        parser.error("--base is required")
    if len(runs) < 2:
        parser.error("provide at least one non-base run to compare")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    variant_names = list(runs)
    comparison = compare_runs(runs)
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    write_csv(args.output_dir / "comparison.csv", comparison, variant_names)
    write_html(args.output_dir / "comparison_report.html", comparison, variant_names)
    print(json.dumps({"ok": True, "output_dir": str(args.output_dir), "runs": variant_names}, indent=2))


if __name__ == "__main__":
    main()
