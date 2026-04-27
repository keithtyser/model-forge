from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any


VARIANT_ARGS = [
    ("base", "--base"),
    ("ft", "--ft"),
    ("abli", "--abli"),
    ("ft_then_abli", "--ft-then-abli"),
    ("abli_then_ft", "--abli-then-ft"),
]

ARTIFACT_VARIANT_ARGS = [
    (name, f"--artifact-{flag.removeprefix('--')}")
    for name, flag in VARIANT_ARGS
]

LOWER_IS_BETTER_PATTERNS = (
    "benign_refusal_rate",
    "unsafe_overcompliance_rate",
    "latency",
)

CRITICAL_REGRESSION_METRICS = {
    "workflow_success",
    "schema_adherence",
    "normal_use_regression_pass_rate",
    "unsafe_overcompliance_rate",
}


def color(code: str, text: str) -> str:
    if os.getenv("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def green(text: str) -> str:
    return color("32", text)


def red(text: str) -> str:
    return color("31", text)


def yellow(text: str) -> str:
    return color("33", text)


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


def load_run(name: str, run_dir: Path, artifact_dir: Path | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(run_dir),
        "artifact_path": str(artifact_dir) if artifact_dir else None,
        "manifest": load_manifest(run_dir),
        "scores": load_scores(run_dir),
        "failures": load_failures(run_dir),
        "artifacts": load_artifacts(artifact_dir or run_dir),
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
    recommendations = build_recommendations(assessments)
    return {
        "runs": {
            name: {
                "path": run["path"],
                "artifact_path": run.get("artifact_path"),
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
        "recommendations": recommendations,
        "failures": {name: run["failures"] for name, run in runs.items()},
        "artifacts": {name: run["artifacts"] for name, run in runs.items()},
    }


def build_recommendations(assessments: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, dict[str, Any]]:
    recommendations = {}
    for name, assessment in assessments.items():
        regressions = assessment.get("regressions", [])
        improvements = assessment.get("improvements", [])
        critical_regressions = [
            item for item in regressions
            if item["metric"] in CRITICAL_REGRESSION_METRICS
        ]
        refusal_improvements = [
            item for item in improvements
            if item["metric"] == "benign_refusal_rate"
        ]
        capability_improvements = [
            item for item in improvements
            if item["metric"] in {"workflow_success", "schema_adherence", "normal_use_regression_pass_rate"}
        ]
        if critical_regressions:
            decision = "reject_or_investigate"
            reason = "critical regressions detected"
        elif improvements:
            decision = "promote_candidate"
            reason = "improvements without critical regressions"
        else:
            decision = "flat"
            reason = "no material metric movement"
        recommendations[name] = {
            "decision": decision,
            "reason": reason,
            "critical_regressions": critical_regressions,
            "refusal_improvements": refusal_improvements,
            "capability_improvements": capability_improvements,
        }
    return recommendations


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
        recommendation = comparison.get("recommendations", {}).get(name, {})
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
            f"<p><strong>Recommendation:</strong> {html.escape(recommendation.get('decision', 'unknown'))} - {html.escape(recommendation.get('reason', ''))}</p>"
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
  <p><a href="artifact_compare.html">Open side-by-side artifact comparison</a></p>
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


def artifact_link(path: Path, target: str | None, label: str) -> str:
    if not target:
        return ""
    href = os.path.relpath(target, start=path.parent)
    return f'<a href="{html.escape(href)}">{html.escape(label)}</a>'


def write_artifact_compare_html(path: Path, comparison: dict[str, Any], variant_names: list[str]) -> None:
    artifact_keys = sorted({key for artifacts in comparison.get("artifacts", {}).values() for key in artifacts})
    sections = []
    for key in artifact_keys:
        cards = []
        for name in variant_names:
            item = comparison.get("artifacts", {}).get(name, {}).get(key)
            if not item:
                cards.append(f'<section class="card missing"><h3>{html.escape(name)}</h3><p>No artifact for this case.</p></section>')
                continue
            validation = "unknown"
            status_class = "unknown"
            if item.get("validation_ok") is True:
                validation = "pass"
                status_class = "pass"
            elif item.get("validation_ok") is False:
                validation = "fail"
                status_class = "fail"
            screenshot_path = item.get("screenshot_path")
            screenshot = ""
            if screenshot_path:
                screenshot_href = os.path.relpath(screenshot_path, start=path.parent)
                screenshot = f'<a href="{html.escape(screenshot_href)}"><img src="{html.escape(screenshot_href)}" alt="{html.escape(name)} screenshot"></a>'
            errors = ", ".join(item.get("validation_errors") or [])
            cards.append(
                f'<section class="card {status_class}">'
                f'<div class="card-head"><h3>{html.escape(name)}</h3><span>{html.escape(validation)}</span></div>'
                f'<div class="preview">{screenshot or "<p>No screenshot captured.</p>"}</div>'
                f'<p class="links">{artifact_link(path, item.get("artifact_path"), "open artifact")} {artifact_link(path, screenshot_path, "open screenshot")}</p>'
                f'<p class="errors">{html.escape(errors)}</p>'
                f'</section>'
            )
        sections.append(
            f'<section class="case"><h2>{html.escape(key)}</h2><div class="grid">'
            + "".join(cards)
            + "</div></section>"
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>model-forge artifact comparison</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fa; --panel:#fff; --text:#15171a; --muted:#626b77; --line:#dfe4ea; --ok:#0f8a5f; --bad:#b42318; }}
    body {{ margin:0; padding:28px; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }}
    header {{ max-width:1200px; margin:0 auto 28px; }}
    h1 {{ margin:0 0 6px; font-size:28px; }}
    p {{ color:var(--muted); }}
    .case {{ max-width:1400px; margin:0 auto 32px; }}
    .case h2 {{ font-size:18px; margin:0 0 12px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:16px; align-items:start; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
    .card-head {{ display:flex; align-items:center; justify-content:space-between; padding:10px 12px; border-bottom:1px solid var(--line); }}
    .card h3 {{ margin:0; font-size:15px; }}
    .card span {{ font-size:12px; font-weight:700; text-transform:uppercase; }}
    .pass span {{ color:var(--ok); }}
    .fail span {{ color:var(--bad); }}
    .preview {{ min-height:220px; background:#eef1f5; display:flex; align-items:center; justify-content:center; }}
    img {{ display:block; width:100%; height:auto; }}
    .links, .errors {{ margin:10px 12px; font-size:13px; }}
    .links a {{ margin-right:12px; }}
    .errors {{ color:var(--bad); }}
    .missing {{ opacity:.65; }}
  </style>
</head>
<body>
  <header>
    <h1>Artifact comparison</h1>
    <p>Side-by-side screenshots and artifact links for each generated artifact prompt.</p>
  </header>
  {''.join(sections) or '<p>No artifacts found.</p>'}
</body>
</html>
"""
    path.write_text(doc)


def metric_label(bucket: str, metric: str) -> str:
    return f"{bucket}/{metric}"


def print_terminal_results(comparison: dict[str, Any], variant_names: list[str], output_dir: Path) -> None:
    print()
    print(f"{green('OK')} Comparison report refreshed")
    print(f"  output:    {output_dir}")
    print(f"  html:      {output_dir / 'comparison_report.html'}")
    print(f"  artifacts: {output_dir / 'artifact_compare.html'}")
    print(f"  runs:      {', '.join(variant_names)}")

    print()
    print("Results")
    score_rows = comparison.get("score_rows", [])
    key_rows = [
        row for row in score_rows
        if row["metric"] not in {"latency_seconds"}
        and (row["bucket"] == "overall" or row["metric"] in {
            "workflow_success",
            "schema_adherence",
            "normal_use_regression_pass_rate",
            "benign_refusal_rate",
            "unsafe_overcompliance_rate",
            "tokens_per_second",
            "latency_seconds_median",
        })
    ]
    for row in key_rows[:32]:
        values = []
        for name in variant_names:
            value = row.get(name)
            if value is None:
                continue
            cell = f"{name}={value:g}"
            if name != "base" and row.get(f"{name}_delta") is not None:
                delta = row[f"{name}_delta"]
                marker = green(f"{delta:+g}") if classify_delta(row["metric"], delta) == "improvement" else red(f"{delta:+g}") if classify_delta(row["metric"], delta) == "regression" else f"{delta:+g}"
                cell = f"{cell} ({marker})"
            values.append(cell)
        print(f"  {metric_label(row['bucket'], row['metric'])}: " + " | ".join(values))

    if len(variant_names) > 1:
        print()
        print("Recommendations")
        for name in variant_names:
            if name == "base":
                continue
            recommendation = comparison.get("recommendations", {}).get(name, {})
            decision = recommendation.get("decision", "unknown")
            rendered = red(decision) if decision == "reject_or_investigate" else green(decision) if decision == "promote_candidate" else yellow(decision)
            print(f"  {name}: {rendered} - {recommendation.get('reason', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model-forge result directories")
    for _, flag in VARIANT_ARGS:
        parser.add_argument(flag, type=Path)
    for _, flag in ARTIFACT_VARIANT_ARGS:
        parser.add_argument(flag, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/comparison"), help="Directory for comparison outputs")
    args = parser.parse_args()

    runs = {}
    for name, _ in VARIANT_ARGS:
        run_dir = getattr(args, name)
        if run_dir:
            artifact_dir = getattr(args, f"artifact_{name}")
            runs[name] = load_run(name, run_dir, artifact_dir=artifact_dir)
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
    write_artifact_compare_html(args.output_dir / "artifact_compare.html", comparison, variant_names)
    print_terminal_results(comparison, variant_names, args.output_dir)


if __name__ == "__main__":
    main()
