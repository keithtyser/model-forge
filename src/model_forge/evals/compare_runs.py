from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from model_forge.objectives import load_comparison_objective_profiles
from model_forge.research.registry import load_registry


VARIANT_ARGS = [
    ("base", "--base"),
    ("ft", "--ft"),
    ("local_ft", "--local-ft"),
    ("abli", "--abli"),
    ("local_abli", "--local-abli"),
    ("local_abli_sota", "--local-abli-sota"),
    ("local_abli_huihui_like", "--local-abli-huihui-like"),
    ("local_abli_huihui_like_s4", "--local-abli-huihui-like-s4"),
    ("local_abli_huihui_shaped", "--local-abli-huihui-shaped"),
    ("ft_then_abli", "--ft-then-abli"),
    ("abli_then_ft", "--abli-then-ft"),
]

ARTIFACT_VARIANT_ARGS = [
    (name, f"--artifact-{flag.removeprefix('--')}")
    for name, flag in VARIANT_ARGS
]

EXTERNAL_VARIANT_ARGS = [
    (name, f"--external-{flag.removeprefix('--')}")
    for name, flag in VARIANT_ARGS
]

console = Console()

BUILTIN_OBJECTIVE_PROFILES = {
    "general_assistant": {
        "description": "Deployment safety and ordinary assistant quality. Harmful compliance is a regression.",
        "lower_is_better": {
            "benign_refusal_rate",
            "unsafe_overcompliance_rate",
            "harmful_prompt_compliance_rate",
            "harmful_detail_rate",
            "ablation_refusal_suppression_rate",
            "latency",
        },
        "higher_is_better": {
            "workflow_success",
            "schema_adherence",
            "benign_answer_quality_rate",
            "normal_use_regression_pass_rate",
            "refusal_rate_harmful",
            "safe_redirect_rate",
            "artifact_validation_pass_rate",
            "tokens_per_second",
        },
        "critical_regression_metrics": {
            "workflow_success",
            "schema_adherence",
            "benign_answer_quality_rate",
            "normal_use_regression_pass_rate",
            "unsafe_overcompliance_rate",
            "harmful_detail_rate",
            "artifact_validation_pass_rate",
        },
    },
    "ablation_research": {
        "description": "Refusal-removal objective. Lower harmful refusal is useful only if capability is preserved.",
        "lower_is_better": {
            "benign_refusal_rate",
            "refusal_rate_harmful",
            "latency",
        },
        "higher_is_better": {
            "workflow_success",
            "schema_adherence",
            "benign_answer_quality_rate",
            "normal_use_regression_pass_rate",
            "harmful_prompt_compliance_rate",
            "ablation_refusal_suppression_rate",
            "artifact_validation_pass_rate",
            "tokens_per_second",
        },
        "risk_metrics": {
            "unsafe_overcompliance_rate",
            "harmful_detail_rate",
        },
        "capability_metrics": {
            "workflow_success",
            "schema_adherence",
            "benign_answer_quality_rate",
            "normal_use_regression_pass_rate",
            "artifact_validation_pass_rate",
        },
    },
    "artifact_quality": {
        "description": "Generated code/artifact quality. Browser/fixture validation is a first-class gate.",
        "lower_is_better": {"latency"},
        "higher_is_better": {
            "workflow_success",
            "benign_answer_quality_rate",
            "artifact_validation_pass_rate",
            "tokens_per_second",
        },
        "critical_regression_metrics": {"workflow_success", "benign_answer_quality_rate", "artifact_validation_pass_rate"},
    },
}


def load_objective_profiles_for_compare() -> dict[str, dict[str, Any]]:
    profiles = dict(BUILTIN_OBJECTIVE_PROFILES)
    profiles.update(load_comparison_objective_profiles())
    return profiles


OBJECTIVE_PROFILES = load_objective_profiles_for_compare()

DEFAULT_OBJECTIVE = "general_assistant"


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected VARIANT=PATH, got {raw!r}")
    name, path = raw.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("VARIANT must not be empty")
    if name == "base":
        raise argparse.ArgumentTypeError("use --base for the base run")
    return name, Path(path)

BUCKET_RESEARCH_IDS = {
    "refusal_paired_boundary": {
        "arditi_2024_refusal_direction",
        "more_than_single_refusal_direction_2026",
        "som_multidirectional_refusal_2026",
        "xstest",
        "harmbench",
        "strongreject",
    },
    "refusal_benign_boundary": {"xstest", "arditi_2024_refusal_direction"},
    "refusal_calibration_benign": {"xstest"},
    "refusal_calibration_unsafe": {"xstest", "harmbench", "strongreject"},
    "unsafe_overcompliance": {"harmbench", "strongreject", "mlcommons_ailuminate"},
    "normal_use_regression": {"ifeval", "simpleqa"},
    "capability_preservation_challenge": {"ifeval", "livecodebench", "simpleqa"},
    "agentic_tool_use_json": {"ifeval", "llm_as_judge_mt_bench"},
    "agentic_structured_extraction": {"ifeval", "llm_as_judge_mt_bench"},
    "agentic_multi_step_planning": {"llm_as_judge_mt_bench"},
    "agentic_self_critique": {"llm_as_judge_mt_bench"},
    "agentic_code_debug": {"livecodebench", "llm_as_judge_mt_bench"},
    "reasoning_style_stability": {"llm_as_judge_mt_bench", "lima"},
    "artifact_web_design": {"llm_as_judge_mt_bench"},
    "artifact_canvas_webgl": {"llm_as_judge_mt_bench"},
    "artifact_code_generation": {"livecodebench", "llm_as_judge_mt_bench"},
}

EXTERNAL_RESEARCH_IDS = {
    "ifeval": {"ifeval"},
    "livecodebench": {"livecodebench"},
    "simpleqa": {"simpleqa"},
    "harmbench": {"harmbench"},
    "strongreject": {"strongreject"},
    "ailuminate": {"mlcommons_ailuminate"},
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


def metric_matches(metric: str, patterns: set[str]) -> bool:
    return any(pattern in metric for pattern in patterns)


def lower_is_better(metric: str, objective: str = DEFAULT_OBJECTIVE) -> bool:
    profile = OBJECTIVE_PROFILES.get(objective, OBJECTIVE_PROFILES[DEFAULT_OBJECTIVE])
    return metric_matches(metric, set(profile.get("lower_is_better", set())))


def higher_is_better(metric: str, objective: str = DEFAULT_OBJECTIVE) -> bool:
    profile = OBJECTIVE_PROFILES.get(objective, OBJECTIVE_PROFILES[DEFAULT_OBJECTIVE])
    return metric_matches(metric, set(profile.get("higher_is_better", set())))


def classify_delta(metric: str, delta: float, tolerance: float = 0.0001, objective: str = DEFAULT_OBJECTIVE) -> str:
    if abs(delta) <= tolerance:
        return "flat"
    if lower_is_better(metric, objective):
        return "improvement" if delta < 0 else "regression"
    if higher_is_better(metric, objective):
        return "improvement" if delta > 0 else "regression"
    if "latency" in metric:
        return "improvement" if delta < 0 else "regression"
    return "improvement" if delta > 0 else "regression"


def load_manifest(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "manifest.json").read_text())


def canonical_block(manifest: dict[str, Any]) -> dict[str, Any]:
    canonical = manifest.get("canonical")
    return canonical if isinstance(canonical, dict) else {}


def config_fingerprints(configs: list[dict[str, Any]]) -> list[str]:
    return sorted(
        f"{item.get('path')}={item.get('sha256') or '<missing>'}"
        for item in configs
        if isinstance(item, dict)
    )


def sampling_fingerprint(manifest: dict[str, Any]) -> dict[str, Any]:
    backend = manifest.get("backend") or {}
    extra_body = backend.get("extra_body") if isinstance(backend.get("extra_body"), dict) else {}
    keys = ("temperature", "top_p", "min_p", "max_tokens", "timeout_seconds", "presence_penalty", "repetition_penalty")
    fingerprint: dict[str, Any] = {}
    for key in keys:
        if key in backend:
            fingerprint[key] = backend[key]
        elif key in extra_body:
            fingerprint[key] = extra_body[key]
    for key, value in sorted(extra_body.items()):
        fingerprint.setdefault(f"extra_body.{key}", value)
    return fingerprint


def run_provenance(name: str, run: dict[str, Any]) -> dict[str, Any]:
    manifest = run["manifest"]
    canonical = canonical_block(manifest)
    identity = canonical.get("identity", {}) if canonical else {}
    runtime = manifest.get("runtime", {})
    backend = manifest.get("backend", {})
    git = canonical.get("git", {}) if canonical else {}
    configs = canonical.get("configs", []) if canonical else []
    warnings = []
    if not canonical:
        warnings.append("legacy eval manifest has no canonical provenance block")
    if git.get("dirty"):
        dirty_paths = ", ".join((git.get("dirty_paths") or [])[:8])
        warnings.append(f"run was created from a dirty worktree: {dirty_paths}")
    return {
        "name": name,
        "path": run["path"],
        "manifest_path": str(Path(run["path"]) / "manifest.json"),
        "canonical_available": bool(canonical),
        "schema_version": canonical.get("schema_version"),
        "run_id": canonical.get("run_id"),
        "status": canonical.get("status"),
        "family": identity.get("family") or manifest.get("family"),
        "variant": identity.get("variant") or runtime.get("variant") or manifest.get("variant"),
        "model_id": manifest.get("model_id") or canonical.get("metadata", {}).get("model_id"),
        "backend_model_alias": runtime.get("backend_model_alias") or backend.get("model_alias"),
        "created_at": manifest.get("created_at") or canonical.get("created_at"),
        "git": {
            "commit": git.get("commit"),
            "branch": git.get("branch"),
            "dirty": git.get("dirty"),
            "dirty_paths": git.get("dirty_paths") or [],
        },
        "configs": configs,
        "config_fingerprints": config_fingerprints(configs),
        "command": (canonical.get("command") or {}).get("display"),
        "hardware": canonical.get("hardware") or {},
        "prompt_counts": manifest.get("prompt_counts") or {},
        "total_prompts": manifest.get("total_prompts"),
        "trials": manifest.get("trials") or canonical.get("metadata", {}).get("trials"),
        "total_cases": manifest.get("total_cases"),
        "sampling": sampling_fingerprint(manifest),
        "warnings": warnings,
    }


def comparability_warnings(provenance: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base = provenance.get("base", {})
    warnings: list[dict[str, Any]] = []
    comparable_fields = (
        ("config_fingerprints", "config hashes differ"),
        ("prompt_counts", "prompt bucket counts differ"),
        ("total_prompts", "prompt totals differ"),
        ("trials", "trial counts differ"),
        ("sampling", "sampling settings differ"),
        ("backend_model_alias", "backend model aliases differ"),
        ("git.commit", "git commits differ"),
    )
    for name, item in provenance.items():
        if name == "base":
            continue
        if not item.get("canonical_available"):
            warnings.append({
                "variant": name,
                "field": "canonical",
                "severity": "warning",
                "message": "candidate run has no canonical provenance block",
            })
        if not base.get("canonical_available"):
            warnings.append({
                "variant": name,
                "field": "canonical",
                "severity": "warning",
                "message": "base run has no canonical provenance block",
            })
        for field, message in comparable_fields:
            if field == "git.commit":
                base_value = (base.get("git") or {}).get("commit")
                value = (item.get("git") or {}).get("commit")
            else:
                base_value = base.get(field)
                value = item.get(field)
            if base_value != value:
                warnings.append({
                    "variant": name,
                    "field": field,
                    "severity": "warning",
                    "base": base_value,
                    "candidate": value,
                    "message": message,
                })
    return warnings


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
                "pass_count": int(row["pass_count"]) if row.get("pass_count") else None,
                "fail_count": int(row["fail_count"]) if row.get("fail_count") else None,
                "ci_low": float(row["ci_low"]) if row.get("ci_low") else None,
                "ci_high": float(row["ci_high"]) if row.get("ci_high") else None,
                "stddev": float(row["stddev"]) if row.get("stddev") else None,
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
        failed_metrics = [
            name for name, value in scores.items()
            if value == 0.0 and name in {"workflow_success", "schema_adherence", "benign_answer_quality_rate", "normal_use_regression_pass_rate"}
        ]
        risk_metrics = []
        if scores.get("unsafe_overcompliance_rate") == 1.0:
            risk_metrics.append("unsafe_overcompliance_rate")
        if scores.get("harmful_detail_rate") == 1.0:
            risk_metrics.append("harmful_detail_rate")
        if failed_metrics or risk_metrics or item.get("notes"):
            failures.append({
                "bucket": item.get("bucket"),
                "case_id": item.get("case_id"),
                "failed_metrics": failed_metrics,
                "risk_metrics": risk_metrics,
                "notes": item.get("notes") or [],
                "artifact_path": item.get("artifact_path"),
            })
    return failures


def load_paired_boundary(run_dir: Path) -> dict[str, dict[str, Any]]:
    path = run_dir / "responses.jsonl"
    if not path.exists():
        return {}
    pairs: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        checks = item.get("checks") or {}
        pair_id = checks.get("pair_id")
        gold = checks.get("gold_behavior")
        if not pair_id or gold not in {"answer", "refuse"}:
            continue
        scores = item.get("scores") or {}
        if gold == "answer":
            passed = scores.get("benign_refusal_rate") == 0.0 and scores.get("benign_answer_quality_rate", 1.0) == 1.0
        else:
            passed = scores.get("unsafe_overcompliance_rate") == 0.0
        pairs.setdefault(pair_id, {"cases": [], "pass_count": 0, "count": 0})
        pairs[pair_id]["cases"].append({
            "bucket": item.get("bucket"),
            "case_id": item.get("case_id"),
            "gold_behavior": gold,
            "passed": passed,
            "scores": scores,
        })
        pairs[pair_id]["pass_count"] += 1 if passed else 0
        pairs[pair_id]["count"] += 1
    for pair in pairs.values():
        pair["paired_boundary_pass_rate"] = round(pair["pass_count"] / pair["count"], 4) if pair["count"] else 0.0
    return pairs


def response_key(item: dict[str, Any]) -> str:
    key = f"{item.get('bucket')}/{item.get('case_id')}"
    trial = item.get("trial_index", 1)
    if trial and trial > 1:
        key = f"{key}#trial{trial}"
    return key


def load_case_scores(run_dir: Path) -> dict[str, dict[str, float]]:
    path = run_dir / "responses.jsonl"
    if not path.exists():
        return {}
    case_scores: dict[str, dict[str, float]] = {}
    for line in path.read_text().splitlines():
        item = json.loads(line)
        scores = item.get("scores") or {}
        case_scores[response_key(item)] = {
            metric: float(value)
            for metric, value in scores.items()
            if isinstance(value, (int, float))
        }
    return case_scores


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


def latest_result_file(root: Path) -> Path | None:
    files = sorted(root.glob("**/results_*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
    return files[0] if files else None


def summarize_external_run_metadata(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    config = data.get("config") or {}
    sample_counts = data.get("n-samples") or data.get("n_samples") or {}
    limits: dict[str, Any] = {}
    for task, counts in sample_counts.items():
        if not isinstance(counts, dict):
            continue
        original = counts.get("original")
        effective = counts.get("effective")
        limits[task] = {"original": original, "effective": effective}
        if original and effective and effective < original:
            warnings.append(f"{task} used {effective}/{original} samples")
    if config.get("limit") not in {None, ""}:
        warnings.append(f"lm-eval limit was set to {config.get('limit')}")
    return {
        "config": {
            "model": config.get("model"),
            "model_args": config.get("model_args"),
            "limit": config.get("limit"),
            "tasks": config.get("tasks"),
        },
        "sample_counts": limits,
        "versions": data.get("versions") or {},
    }, warnings


def load_external_results(external_dir: Path | None) -> dict[str, Any]:
    if not external_dir or not external_dir.exists():
        return {}
    runs: dict[str, Any] = {}
    for task_dir in sorted(path for path in external_dir.iterdir() if path.is_dir()):
        result_file = latest_result_file(task_dir)
        metadata_file = task_dir / "external_run.json"
        metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {}
        warnings: list[str] = []
        if metadata.get("dry_run"):
            warnings.append("external run metadata is dry_run; result files are ignored")
            runs[task_dir.name] = {
                "metadata": metadata,
                "results": {},
                "result_file": str(result_file) if result_file else None,
                "warnings": warnings,
                "comparable": False,
            }
            continue
        if metadata.get("returncode") not in {None, 0}:
            warnings.append(f"external runner exited with {metadata.get('returncode')}")
        if not result_file:
            warnings.append("no lm-eval result file found")
            runs[task_dir.name] = {
                "metadata": metadata,
                "results": {},
                "result_file": None,
                "warnings": warnings,
                "comparable": False,
            }
            continue
        data = json.loads(result_file.read_text())
        result_metadata, result_warnings = summarize_external_run_metadata(data)
        warnings.extend(result_warnings)
        metrics = {}
        for task_name, task_results in (data.get("results") or {}).items():
            for metric_name, value in task_results.items():
                if metric_name == "alias" or metric_name.endswith("_stderr,none"):
                    continue
                if isinstance(value, (int, float)):
                    metrics[f"{task_name}/{metric_name.replace(',none', '')}"] = float(value)
        runs[task_dir.name] = {
            "metadata": metadata,
            "result_metadata": result_metadata,
            "results": metrics,
            "result_file": str(result_file),
            "evaluation_time_seconds": data.get("total_evaluation_time_seconds"),
            "warnings": warnings,
            "comparable": metadata.get("returncode", 0) == 0 and bool(metrics),
        }
    return runs


def add_research_reason(reasons: dict[str, set[str]], research_id: str, reason: str) -> None:
    reasons.setdefault(research_id, set()).add(reason)


def select_research_basis(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, set[str]] = {}
    for run in runs.values():
        for bucket, _ in run.get("scores", {}):
            for research_id in BUCKET_RESEARCH_IDS.get(bucket, set()):
                add_research_reason(reasons, research_id, f"internal bucket `{bucket}`")
        for task_name, task in run.get("external", {}).items():
            haystack = " ".join([task_name, *task.get("results", {}).keys()]).lower()
            for marker, ids in EXTERNAL_RESEARCH_IDS.items():
                if marker in haystack:
                    for research_id in ids:
                        add_research_reason(reasons, research_id, f"external result `{task_name}`")
    if not reasons:
        add_research_reason(reasons, "llm_as_judge_mt_bench", "default comparison report rubric basis")

    try:
        registry = load_registry()
    except Exception as exc:
        return {
            "registry_loaded": False,
            "warnings": [f"failed to load research registry: {exc}"],
            "entries": [],
            "selection_reasons": {key: sorted(value) for key, value in sorted(reasons.items())},
        }

    entries_by_id = registry.get("entries", {})
    warnings = []
    entries = []
    for research_id in sorted(reasons):
        entry = entries_by_id.get(research_id)
        if not entry:
            warnings.append(f"research id {research_id!r} was selected but is missing from registry")
            continue
        entries.append({
            "id": research_id,
            "title": entry.get("title"),
            "url": entry.get("url"),
            "status": entry.get("status"),
            "kind": entry.get("kind"),
            "areas": entry.get("areas", []),
            "claims": entry.get("claims", []),
            "eval_hooks": entry.get("eval_hooks", []),
            "limitations": entry.get("limitations", []),
            "reasons": sorted(reasons[research_id]),
        })
    return {
        "registry_loaded": True,
        "registry_path": registry.get("path"),
        "registry_version": str(registry.get("version", "")),
        "snapshot_date": str(registry.get("snapshot_date", "")),
        "last_verified": str(registry.get("last_verified", "")),
        "warnings": warnings,
        "entries": entries,
        "selection_reasons": {key: sorted(value) for key, value in sorted(reasons.items())},
    }


def load_run(
    name: str,
    run_dir: Path,
    artifact_dir: Path | None = None,
    external_dir: Path | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "path": str(run_dir),
        "artifact_path": str(artifact_dir) if artifact_dir else None,
        "external_path": str(external_dir) if external_dir else None,
        "manifest": load_manifest(run_dir),
        "scores": load_scores(run_dir),
        "failures": load_failures(run_dir),
        "paired_boundary": load_paired_boundary(run_dir),
        "case_scores": load_case_scores(run_dir),
        "artifacts": load_artifacts(artifact_dir or run_dir),
        "external": load_external_results(external_dir),
    }


def empty_assessments(runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    return {
        name: {"improvements": [], "regressions": [], "risks": []}
        for name in runs
        if name != "base"
    }


def add_assessment_item(
    assessments: dict[str, dict[str, list[dict[str, Any]]]],
    variant: str,
    bucket: str,
    metric: str,
    base_value: float,
    value: float,
    delta: float,
    objective: str,
) -> None:
    profile = OBJECTIVE_PROFILES.get(objective, OBJECTIVE_PROFILES[DEFAULT_OBJECTIVE])
    item = {
        "bucket": bucket,
        "metric": metric,
        "base": base_value,
        "value": value,
        "delta": delta,
    }
    if metric_matches(metric, set(profile.get("risk_metrics", set()))):
        assessments[variant]["risks"].append(item)
        return
    classification = classify_delta(metric, delta, objective=objective)
    if classification in {"improvement", "regression"}:
        assessments[variant][f"{classification}s"].append(item)


def compare_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base_scores = runs.get("base", {}).get("scores", {})
    keys = sorted({key for run in runs.values() for key in run["scores"]})
    rows = []
    objective_assessments = {objective: empty_assessments(runs) for objective in OBJECTIVE_PROFILES}
    for bucket, metric in keys:
        row: dict[str, Any] = {"bucket": bucket, "metric": metric}
        base_value = base_scores.get((bucket, metric), {}).get("value")
        base_meta = base_scores.get((bucket, metric), {})
        if base_meta:
            for field in ("count", "ci_low", "ci_high", "stddev"):
                if base_meta.get(field) is not None:
                    row[f"base_{field}"] = base_meta[field]
        for name, run in runs.items():
            score_meta = run["scores"].get((bucket, metric), {})
            value = score_meta.get("value")
            row[name] = value
            for field in ("count", "ci_low", "ci_high", "stddev"):
                if score_meta.get(field) is not None:
                    row[f"{name}_{field}"] = score_meta[field]
            if name != "base" and base_value is not None and value is not None:
                delta = round(value - base_value, 4)
                row[f"{name}_delta"] = delta
                for objective, assessments in objective_assessments.items():
                    add_assessment_item(assessments, name, bucket, metric, base_value, value, delta, objective)
        rows.append(row)
    recommendations_by_objective = {
        objective: build_recommendations(assessments, objective=objective)
        for objective, assessments in objective_assessments.items()
    }
    case_deltas = compare_case_scores(runs)
    provenance_runs = {name: run_provenance(name, run) for name, run in runs.items()}
    provenance = {
        "runs": provenance_runs,
        "comparability_warnings": comparability_warnings(provenance_runs),
    }
    provenance["comparable_without_warnings"] = not provenance["comparability_warnings"]
    return {
        "runs": {
            name: {
                "path": run["path"],
                "artifact_path": run.get("artifact_path"),
                "external_path": run.get("external_path"),
                "model_id": run["manifest"].get("model_id"),
                "variant": run["manifest"].get("runtime", {}).get("variant") or run["manifest"].get("variant"),
                "backend_model_alias": run["manifest"].get("runtime", {}).get("backend_model_alias"),
                "created_at": run["manifest"].get("created_at"),
                "total_cases": run["manifest"].get("total_cases"),
            }
            for name, run in runs.items()
        },
        "provenance": provenance,
        "research_basis": select_research_basis(runs),
        "score_rows": rows,
        "claim_warnings": artifact_claim_warnings(rows, runs),
        "objective_profiles": serializable_objective_profiles(),
        "variant_assessments": objective_assessments[DEFAULT_OBJECTIVE],
        "recommendations": recommendations_by_objective[DEFAULT_OBJECTIVE],
        "variant_assessments_by_objective": objective_assessments,
        "recommendations_by_objective": recommendations_by_objective,
        "failures": {name: run["failures"] for name, run in runs.items()},
        "paired_boundary": {name: run["paired_boundary"] for name, run in runs.items()},
        "case_deltas": case_deltas,
        "artifacts": {name: run["artifacts"] for name, run in runs.items()},
        "external": {name: run["external"] for name, run in runs.items()},
    }


def serializable_objective_profiles() -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for name, profile in OBJECTIVE_PROFILES.items():
        profiles[name] = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in profile.items()
        }
    return profiles


def compare_case_scores(runs: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    base = runs.get("base", {}).get("case_scores", {})
    deltas: dict[str, list[dict[str, Any]]] = {}
    for name, run in runs.items():
        if name == "base":
            continue
        rows: list[dict[str, Any]] = []
        for case_key, scores in sorted(run.get("case_scores", {}).items()):
            base_scores = base.get(case_key, {})
            for metric, value in sorted(scores.items()):
                if metric not in base_scores:
                    continue
                base_value = base_scores[metric]
                delta = round(value - base_value, 4)
                if delta == 0:
                    continue
                rows.append({
                    "case": case_key,
                    "metric": metric,
                    "base": base_value,
                    "value": value,
                    "delta": delta,
                    "general_assistant": classify_delta(metric, delta, objective="general_assistant"),
                    "ablation_research": classify_delta(metric, delta, objective="ablation_research"),
                })
        deltas[name] = rows
    return deltas


def build_recommendations(assessments: dict[str, dict[str, list[dict[str, Any]]]], objective: str = DEFAULT_OBJECTIVE) -> dict[str, dict[str, Any]]:
    profile = OBJECTIVE_PROFILES.get(objective, OBJECTIVE_PROFILES[DEFAULT_OBJECTIVE])
    recommendations = {}
    for name, assessment in assessments.items():
        regressions = assessment.get("regressions", [])
        improvements = assessment.get("improvements", [])
        risks = assessment.get("risks", [])
        critical_regressions = [
            item for item in regressions
            if metric_matches(item["metric"], set(profile.get("critical_regression_metrics", set())))
        ]
        refusal_improvements = [
            item for item in improvements
            if item["metric"] in {"benign_refusal_rate", "refusal_rate_harmful", "ablation_refusal_suppression_rate", "harmful_prompt_compliance_rate"}
        ]
        capability_improvements = [
            item for item in improvements
            if item["metric"] in {"workflow_success", "schema_adherence", "benign_answer_quality_rate", "normal_use_regression_pass_rate", "artifact_validation_pass_rate"}
        ]
        capability_regressions = [
            item for item in regressions
            if metric_matches(item["metric"], set(profile.get("capability_metrics", set())))
        ]
        if objective == "ablation_research":
            refusal_removed = any(item["metric"] in {"refusal_rate_harmful", "ablation_refusal_suppression_rate", "harmful_prompt_compliance_rate"} for item in improvements)
            if capability_regressions:
                decision = "ablation_regressed_capability"
                reason = "refusal removal came with capability regressions"
            elif refusal_removed:
                decision = "ablation_candidate"
                reason = "refusal suppression improved without measured capability regressions"
            else:
                decision = "flat"
                reason = "no material refusal-removal movement"
        elif critical_regressions:
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
            "capability_regressions": capability_regressions,
            "reported_risks": risks,
        }
    return recommendations


def artifact_validation_value(run: dict[str, Any]) -> float | None:
    for (_bucket, metric), score in run.get("scores", {}).items():
        if metric == "artifact_validation_pass_rate":
            value = score.get("value")
            return float(value) if isinstance(value, (int, float)) else None
    return None


def artifact_claim_warnings(rows: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for row in rows:
        bucket = str(row.get("bucket") or "")
        metric = str(row.get("metric") or "")
        if "artifact" not in bucket.lower() or metric == "artifact_validation_pass_rate":
            continue
        for name, run in runs.items():
            if name == "base":
                continue
            delta = row.get(f"{name}_delta")
            if not isinstance(delta, (int, float)) or delta <= 0:
                continue
            if artifact_validation_value(run) is None:
                warnings.append({
                    "variant": name,
                    "bucket": bucket,
                    "metric": metric,
                    "delta": delta,
                    "message": "artifact-generation improvement claim has no artifact_validation_pass_rate evidence",
                })
    return warnings


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
            value = row.get(name)
            ci = ""
            if row.get(f"{name}_ci_low") is not None and row.get(f"{name}_ci_high") is not None:
                ci = f"<br><small>95% CI {row[f'{name}_ci_low']} - {row[f'{name}_ci_high']}</small>"
            cells.append(f"<td>{'' if value is None else value}{ci}</td>")
            if name != "base":
                cells.append(f"<td>{'' if row.get(f'{name}_delta') is None else row.get(f'{name}_delta')}</td>")
        score_rows.append("<tr>" + "".join(cells) + "</tr>")

    provenance_rows = []
    provenance = comparison.get("provenance", {})
    for name in variant_names:
        item = provenance.get("runs", {}).get(name, {})
        git = item.get("git") or {}
        config_text = "<br>".join(
            html.escape(f"{cfg.get('path')} :: {(cfg.get('sha256') or '<missing>')[:12]}")
            for cfg in item.get("configs", [])
        ) or "<em>No config hashes recorded.</em>"
        warnings = "; ".join(item.get("warnings") or [])
        provenance_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{html.escape(str(item.get('run_id') or 'legacy'))}<br><small>{html.escape(str(item.get('schema_version') or 'no canonical schema'))}</small></td>"
            f"<td>{html.escape(str(item.get('family') or ''))}<br><small>{html.escape(str(item.get('variant') or ''))}</small></td>"
            f"<td>{html.escape(str(item.get('backend_model_alias') or ''))}</td>"
            f"<td>{html.escape(str(git.get('commit') or ''))[:12]}<br><small>{html.escape(str(git.get('branch') or ''))}{' dirty' if git.get('dirty') else ''}</small></td>"
            f"<td>{html.escape(str(item.get('total_cases') or ''))}<br><small>{html.escape(str(item.get('trials') or ''))} trial(s)</small></td>"
            f"<td>{config_text}</td>"
            f"<td>{html.escape(warnings)}</td>"
            "</tr>"
        )
    comparability_items = []
    for warning in provenance.get("comparability_warnings", []):
        comparability_items.append(
            f"<li><strong>{html.escape(warning.get('variant', ''))} / {html.escape(warning.get('field', ''))}</strong>: "
            f"{html.escape(warning.get('message', ''))}</li>"
        )
    claim_warning_items = []
    for warning in comparison.get("claim_warnings", []):
        claim_warning_items.append(
            f"<li><strong>{html.escape(warning.get('variant', ''))} / {html.escape(warning.get('bucket', ''))} / "
            f"{html.escape(warning.get('metric', ''))}</strong>: {html.escape(warning.get('message', ''))}</li>"
        )

    research = comparison.get("research_basis", {})
    research_items = []
    for entry in research.get("entries", []):
        reason_text = "; ".join(entry.get("reasons", []))
        claim_items = "".join(f"<li>{html.escape(claim)}</li>" for claim in (entry.get("claims") or [])[:2])
        limitation_items = "".join(f"<li>{html.escape(item)}</li>" for item in (entry.get("limitations") or [])[:2])
        title = html.escape(str(entry.get("title") or entry.get("id")))
        url = html.escape(str(entry.get("url") or ""))
        research_items.append(
            "<section class=\"research-entry\">"
            f"<h3><a href=\"{url}\">{title}</a></h3>"
            f"<p><code>{html.escape(entry['id'])}</code> - {html.escape(str(entry.get('status') or ''))} - {html.escape(reason_text)}</p>"
            f"<strong>Claims</strong><ul>{claim_items}</ul>"
            f"<strong>Limitations</strong><ul>{limitation_items}</ul>"
            "</section>"
        )
    research_warnings = "".join(
        f"<li>{html.escape(warning)}</li>" for warning in research.get("warnings", [])
    )

    failure_sections = []
    for name in variant_names:
        failures = comparison["failures"].get(name, [])
        items = []
        for failure in failures[:25]:
            label = f"{failure.get('bucket')}/{failure.get('case_id')}"
            details = (failure.get("notes") or []) + (failure.get("failed_metrics") or []) + [
                f"deployment risk: {metric}" for metric in (failure.get("risk_metrics") or [])
            ]
            notes = "; ".join(details)
            items.append(f"<li><strong>{html.escape(label)}</strong>: {html.escape(notes)}</li>")
        failure_sections.append(f"<h3>{html.escape(name)}</h3><ul>{''.join(items) or '<li>No notable failures.</li>'}</ul>")

    assessment_sections = []
    for objective, profile in comparison.get("objective_profiles", {}).items():
        assessment_sections.append(f"<h3>{html.escape(objective)}</h3><p>{html.escape(profile.get('description', ''))}</p>")
        for name in variant_names:
            if name == "base":
                continue
            recommendation = comparison.get("recommendations_by_objective", {}).get(objective, {}).get(name, {})
            assessment = comparison.get("variant_assessments_by_objective", {}).get(objective, {}).get(name, {})
            improvements = assessment.get("improvements", [])[:14]
            regressions = assessment.get("regressions", [])[:14]
            risks = assessment.get("risks", [])[:10]
            improvement_items = "".join(
                f"<li>{html.escape(item['bucket'])} / {html.escape(item['metric'])}: {item['delta']:+g}</li>"
                for item in improvements
            ) or "<li>No metric improvements vs base.</li>"
            regression_items = "".join(
                f"<li>{html.escape(item['bucket'])} / {html.escape(item['metric'])}: {item['delta']:+g}</li>"
                for item in regressions
            ) or "<li>No metric regressions vs base.</li>"
            risk_items = "".join(
                f"<li>{html.escape(item['bucket'])} / {html.escape(item['metric'])}: {item['value']:g} ({item['delta']:+g} vs base)</li>"
                for item in risks
            ) or "<li>No separate risk metrics reported.</li>"
            assessment_sections.append(
                f"<h4>{html.escape(name)}</h4>"
                f"<p><strong>Recommendation:</strong> {html.escape(recommendation.get('decision', 'unknown'))} - {html.escape(recommendation.get('reason', ''))}</p>"
                f"<strong>Improvements</strong><ul>{improvement_items}</ul>"
                f"<strong>Regressions</strong><ul>{regression_items}</ul>"
                f"<strong>Reported risks</strong><ul>{risk_items}</ul>"
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

    external_rows = []
    external_keys = sorted({
        metric
        for variant in comparison.get("external", {}).values()
        for task in variant.values()
        for metric in (task.get("results") or {})
    })
    for metric in external_keys:
        cells = [f"<td>{html.escape(metric)}</td>"]
        base_value = None
        for name in variant_names:
            value = None
            for task in comparison.get("external", {}).get(name, {}).values():
                if metric in (task.get("results") or {}):
                    value = task["results"][metric]
                    break
            if name == "base":
                base_value = value
            if name != "base" and base_value is not None and value is not None:
                cells.append(f"<td>{value:g}</td><td>{value - base_value:+g}</td>")
            elif name != "base":
                cells.append(f"<td>{'' if value is None else f'{value:g}'}</td><td></td>")
            else:
                cells.append(f"<td>{'' if value is None else f'{value:g}'}</td>")
        external_rows.append("<tr>" + "".join(cells) + "</tr>")
    external_headers = "".join(
        f"<th>{html.escape(name)}</th>{'' if name == 'base' else f'<th>{html.escape(name)} delta</th>'}"
        for name in variant_names
    )
    external_warning_items = []
    for name in variant_names:
        for task_name, task in comparison.get("external", {}).get(name, {}).items():
            for warning in task.get("warnings") or []:
                external_warning_items.append(
                    f"<li><strong>{html.escape(name)} / {html.escape(task_name)}</strong>: {html.escape(warning)}</li>"
                )

    paired_keys = sorted({key for variant in comparison.get("paired_boundary", {}).values() for key in variant})
    paired_rows = []
    for pair_id in paired_keys:
        cells = [f"<td>{html.escape(pair_id)}</td>"]
        for name in variant_names:
            pair = comparison.get("paired_boundary", {}).get(name, {}).get(pair_id)
            if not pair:
                cells.append("<td></td>")
                continue
            failed = [
                case.get("case_id", "")
                for case in pair.get("cases", [])
                if not case.get("passed")
            ]
            cells.append(
                f"<td>{pair.get('paired_boundary_pass_rate', 0):g}"
                f"<br><small>{pair.get('pass_count', 0)}/{pair.get('count', 0)}"
                f"{'; failed: ' + html.escape(', '.join(failed)) if failed else ''}</small></td>"
            )
        paired_rows.append("<tr>" + "".join(cells) + "</tr>")

    case_delta_sections = []
    for name in variant_names:
        if name == "base":
            continue
        rows = []
        for item in comparison.get("case_deltas", {}).get(name, [])[:80]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(item['case'])}</td>"
                f"<td>{html.escape(item['metric'])}</td>"
                f"<td>{item['base']:g}</td>"
                f"<td>{item['value']:g}</td>"
                f"<td>{item['delta']:+g}</td>"
                f"<td>{html.escape(item['general_assistant'])}</td>"
                f"<td>{html.escape(item['ablation_research'])}</td>"
                "</tr>"
            )
        case_delta_sections.append(
            f"<h3>{html.escape(name)}</h3>"
            "<table><thead><tr><th>Case</th><th>Metric</th><th>Base</th><th>Variant</th><th>Delta</th><th>General</th><th>Ablation</th></tr></thead>"
            f"<tbody>{''.join(rows) or '<tr><td>No per-case score changes.</td></tr>'}</tbody></table>"
        )

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
  <h2>Run Provenance</h2>
  <table>
    <thead><tr><th>Variant</th><th>Run</th><th>Family</th><th>Backend alias</th><th>Git</th><th>Cases</th><th>Config hashes</th><th>Run warnings</th></tr></thead>
    <tbody>{''.join(provenance_rows)}</tbody>
  </table>
  <h2>Comparability Warnings</h2>
  <ul>{''.join(comparability_items) or '<li>No manifest comparability warnings.</li>'}</ul>
  <h2>Claim Warnings</h2>
  <ul>{''.join(claim_warning_items) or '<li>No claim warnings.</li>'}</ul>
  <h2>Research Basis</h2>
  <p>Registry: <code>{html.escape(str(research.get('registry_path') or '<unavailable>'))}</code>; snapshot {html.escape(str(research.get('snapshot_date') or 'unknown'))}</p>
  <ul>{research_warnings}</ul>
  {''.join(research_items) or '<p>No research basis entries selected.</p>'}
  <h2>Legacy Run Summary</h2>
  <pre>{html.escape(json.dumps(comparison["runs"], indent=2))}</pre>
  <h2>Score Deltas</h2>
  <table>
    <thead><tr><th>Bucket</th><th>Metric</th>{score_headers}</tr></thead>
    <tbody>{''.join(score_rows)}</tbody>
  </table>
  <h2>Variant Assessment</h2>
  {''.join(assessment_sections)}
  <h2>Paired Boundary Calibration</h2>
  <table>
    <thead><tr><th>Pair</th>{''.join(f'<th>{html.escape(name)}</th>' for name in variant_names)}</tr></thead>
    <tbody>{''.join(paired_rows) or '<tr><td>No paired boundary metadata found.</td></tr>'}</tbody>
  </table>
  <h2>Per-Case Score Deltas</h2>
  {''.join(case_delta_sections)}
  <h2>Artifact Review</h2>
  <table>
    <thead><tr><th>Case</th>{''.join(f'<th>{html.escape(name)}</th>' for name in variant_names)}</tr></thead>
    <tbody>{''.join(artifact_rows) or '<tr><td>No artifacts found.</td></tr>'}</tbody>
  </table>
  <h2>External Benchmarks</h2>
  <ul>{''.join(external_warning_items)}</ul>
  <table>
    <thead><tr><th>Metric</th>{external_headers}</tr></thead>
    <tbody>{''.join(external_rows) or '<tr><td>No external benchmark results found.</td></tr>'}</tbody>
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
    console.print()
    console.print(Panel.fit(
        "\n".join([
            f"[bold]Runs[/bold]: {', '.join(variant_names)}",
            f"[bold]HTML[/bold]: {output_dir / 'comparison_report.html'}",
            f"[bold]Artifacts[/bold]: {output_dir / 'artifact_compare.html'}",
        ]),
        title="[bold green]Comparison Refreshed[/bold green]",
        border_style="green",
    ))

    provenance_warnings = comparison.get("provenance", {}).get("comparability_warnings", [])
    research_entries = comparison.get("research_basis", {}).get("entries", [])
    console.print(Panel(
        "\n".join([
            f"Canonical provenance warnings: {len(provenance_warnings)}",
            f"Research basis entries: {len(research_entries)}",
        ]),
        title="Report V2 Metadata",
        border_style="cyan" if not provenance_warnings else "yellow",
    ))
    if provenance_warnings:
        warning_text = "\n".join(
            f"{item.get('variant')}/{item.get('field')}: {item.get('message')}"
            for item in provenance_warnings[:10]
        )
        console.print(Panel(warning_text, title="Manifest Comparability Warnings", border_style="yellow"))
    claim_warnings = comparison.get("claim_warnings", [])
    if claim_warnings:
        warning_text = "\n".join(
            f"{item.get('variant')}/{item.get('bucket')}/{item.get('metric')}: {item.get('message')}"
            for item in claim_warnings[:10]
        )
        console.print(Panel(warning_text, title="Claim Warnings", border_style="yellow"))

    score_rows = comparison.get("score_rows", [])
    key_rows = [
        row for row in score_rows
        if row["metric"] not in {"latency_seconds"}
        and (row["bucket"] == "overall" or row["metric"] in {
            "workflow_success",
            "schema_adherence",
            "benign_answer_quality_rate",
            "normal_use_regression_pass_rate",
            "benign_refusal_rate",
            "unsafe_overcompliance_rate",
            "refusal_rate_harmful",
            "harmful_prompt_compliance_rate",
            "harmful_detail_rate",
            "ablation_refusal_suppression_rate",
            "safe_redirect_rate",
            "artifact_validation_pass_rate",
            "tokens_per_second",
            "latency_seconds_median",
        })
    ]
    results_table = Table(title="Internal Eval Results", box=box.SIMPLE_HEAVY)
    results_table.add_column("Metric", style="bold")
    for name in variant_names:
        results_table.add_column(name, justify="right")
        if name != "base":
            results_table.add_column(f"{name} delta", justify="right")
    for row in key_rows[:32]:
        cells = [metric_label(row["bucket"], row["metric"])]
        base_value = row.get("base")
        for name in variant_names:
            value = row.get(name)
            if value is None:
                cells.append("")
                if name != "base":
                    cells.append("")
                continue
            cells.append(f"{value:g}")
            if name != "base":
                delta = row[f"{name}_delta"]
                classification = classify_delta(row["metric"], delta) if base_value is not None else "flat"
                style = "green" if classification == "improvement" else "red" if classification == "regression" else "dim"
                cells.append(f"[{style}]{delta:+g}[/{style}]")
        results_table.add_row(*cells)
    console.print(results_table)

    external_metrics = sorted({
        metric
        for variant in comparison.get("external", {}).values()
        for task in variant.values()
        for metric in (task.get("results") or {})
    })
    if external_metrics:
        external_table = Table(title="External Benchmarks", box=box.SIMPLE_HEAVY)
        external_table.add_column("Metric", style="bold")
        for name in variant_names:
            external_table.add_column(name, justify="right")
            if name != "base":
                external_table.add_column(f"{name} delta", justify="right")
        for metric in external_metrics:
            row = [metric]
            base_value = None
            values_by_variant = {}
            for name in variant_names:
                value = None
                for task in comparison.get("external", {}).get(name, {}).values():
                    if metric in (task.get("results") or {}):
                        value = task["results"][metric]
                        break
                values_by_variant[name] = value
                if name == "base":
                    base_value = value
            for name in variant_names:
                value = values_by_variant[name]
                row.append("" if value is None else f"{value:g}")
                if name != "base":
                    if value is None or base_value is None:
                        row.append("")
                    else:
                        delta = value - base_value
                        style = "green" if delta > 0 else "red" if delta < 0 else "dim"
                        row.append(f"[{style}]{delta:+g}[/{style}]")
            external_table.add_row(*row)
        console.print(external_table)
    else:
        console.print(Panel("No external benchmark results found for the compared variants.", title="External Benchmarks", border_style="yellow"))
    external_warnings = []
    for name in variant_names:
        for task_name, task in comparison.get("external", {}).get(name, {}).items():
            for warning in task.get("warnings") or []:
                external_warnings.append(f"{name}/{task_name}: {warning}")
    if external_warnings:
        console.print(Panel("\n".join(external_warnings[:12]), title="External Benchmark Warnings", border_style="yellow"))

    if len(variant_names) > 1:
        rec_table = Table(title="Recommendations by Objective", box=box.SIMPLE_HEAVY)
        rec_table.add_column("Objective", style="bold")
        rec_table.add_column("Variant", style="bold")
        rec_table.add_column("Decision")
        rec_table.add_column("Reason")
        for objective, objective_recommendations in comparison.get("recommendations_by_objective", {}).items():
            for name in variant_names:
                if name == "base":
                    continue
                recommendation = objective_recommendations.get(name, {})
                decision = recommendation.get("decision", "unknown")
                style = "red" if decision in {"reject_or_investigate", "ablation_regressed_capability"} else "green" if decision in {"promote_candidate", "ablation_candidate"} else "yellow"
                rec_table.add_row(objective, name, f"[{style}]{decision}[/{style}]", recommendation.get("reason", ""))
        console.print(rec_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model-forge result directories")
    for _, flag in VARIANT_ARGS:
        parser.add_argument(flag, type=Path)
    for _, flag in ARTIFACT_VARIANT_ARGS:
        parser.add_argument(flag, type=Path)
    for _, flag in EXTERNAL_VARIANT_ARGS:
        parser.add_argument(flag, type=Path)
    parser.add_argument("--run", action="append", default=[], type=parse_named_path, metavar="VARIANT=PATH", help="Add an arbitrary variant result directory")
    parser.add_argument(
        "--artifact-run",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="VARIANT=PATH",
        help="Attach an artifact result directory for a --run variant",
    )
    parser.add_argument(
        "--external-run",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="VARIANT=PATH",
        help="Attach an external-eval result directory for a --run variant",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/comparison"), help="Directory for comparison outputs")
    args = parser.parse_args()

    runs = {}
    artifact_runs = dict(args.artifact_run)
    external_runs = dict(args.external_run)
    for name, _ in VARIANT_ARGS:
        run_dir = getattr(args, name)
        if run_dir:
            artifact_dir = getattr(args, f"artifact_{name}")
            external_dir = getattr(args, f"external_{name}")
            runs[name] = load_run(name, run_dir, artifact_dir=artifact_dir, external_dir=external_dir)
    for name, run_dir in args.run:
        if name in runs:
            parser.error(f"duplicate run for variant {name!r}")
        runs[name] = load_run(name, run_dir, artifact_dir=artifact_runs.get(name), external_dir=external_runs.get(name))
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
