from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.benchmarks.serve import DEFAULT_CONFIG as DEFAULT_SERVE_CONFIG
from model_forge.benchmarks.serve import load_config as load_serve_config
from model_forge.evals.run_eval import (
    EvalCase,
    build_manifest,
    collect_cases,
    load_config as load_eval_config,
    run_cases_with_progress,
    summarize_scores,
    write_outputs,
)
from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


DEFAULT_CONFIG = REPO_DIR / "configs" / "serving" / "serve_eval_quality_behavior.yaml"
SCHEMA_VERSION = "model_forge.serving_eval.v1"
COMPARISON_SCHEMA_VERSION = "model_forge.serving_eval_comparison.v1"

console = Console(stderr=True)


@dataclass(frozen=True)
class SampleSpec:
    prompt_set: str
    case_ids: tuple[str, ...]
    count: int | None


@dataclass(frozen=True)
class ServingEvalConfig:
    name: str
    description: str
    serving_config: Path
    eval_config: Path | None
    output_root: Path
    samples: list[SampleSpec]
    trials: int
    timeout_seconds: int | None
    raw_config: dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_DIR / path


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(path)}")
    return data


def family_eval_config(family: str | None) -> Path | None:
    if not family:
        return None
    path = REPO_DIR / "configs" / "model_families" / f"{family}.yaml"
    if not path.exists():
        return None
    data = load_yaml(path)
    eval_data = data.get("eval") or {}
    raw_config = eval_data.get("config")
    return resolve_repo_path(str(raw_config)) if raw_config else None


def parse_samples(raw_samples: Any) -> list[SampleSpec]:
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("sample prompt_sets must be a non-empty list")
    samples: list[SampleSpec] = []
    for raw in raw_samples:
        if not isinstance(raw, Mapping):
            raise ValueError("sample prompt_sets entries must be mappings")
        prompt_set = str(raw.get("prompt_set") or raw.get("bucket") or "").strip()
        if not prompt_set:
            raise ValueError("each sample prompt_sets entry needs prompt_set")
        case_ids = tuple(str(item) for item in (raw.get("case_ids") or []))
        count = int(raw["count"]) if raw.get("count") is not None else None
        if count is not None and count < 1:
            raise ValueError("sample count must be >= 1")
        if not case_ids and count is None:
            raise ValueError(f"sample entry for {prompt_set!r} needs case_ids or count")
        samples.append(SampleSpec(prompt_set=prompt_set, case_ids=case_ids, count=count))
    return samples


def load_serving_eval_config(path: Path, *, serving_config: Path | None = None, eval_config: Path | None = None) -> ServingEvalConfig:
    raw = load_yaml(path)
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{display_path(path)} must use schema_version {SCHEMA_VERSION}")
    serving_data = raw.get("serving") or {}
    eval_data = raw.get("eval") or {}
    sample_data = raw.get("sample") or {}
    raw_serving_config = serving_config or serving_data.get("benchmark_config") or DEFAULT_SERVE_CONFIG
    raw_eval_config = eval_config or eval_data.get("config")
    return ServingEvalConfig(
        name=str(raw.get("name") or "serving_quality_behavior_sample"),
        description=str(raw.get("description") or ""),
        serving_config=resolve_repo_path(raw_serving_config),
        eval_config=resolve_repo_path(raw_eval_config) if raw_eval_config else None,
        output_root=resolve_repo_path(eval_data.get("output_root") or "reports/generated/serving_evals"),
        samples=parse_samples(sample_data.get("prompt_sets") or sample_data.get("samples")),
        trials=int(eval_data.get("trials", 1)),
        timeout_seconds=int(eval_data["timeout_seconds"]) if eval_data.get("timeout_seconds") is not None else None,
        raw_config=raw,
    )


def select_sample_cases(samples: list[SampleSpec], *, prompt_root: Path, max_cases: int | None = None) -> list[EvalCase]:
    prompt_sets = sorted({sample.prompt_set for sample in samples})
    all_cases = collect_cases(prompt_root, prompt_sets)
    by_bucket: dict[str, list[EvalCase]] = {}
    for case in all_cases:
        by_bucket.setdefault(case.bucket, []).append(case)

    selected: list[EvalCase] = []
    seen: set[tuple[str, str]] = set()
    for sample in samples:
        bucket_cases = by_bucket.get(sample.prompt_set) or []
        if not bucket_cases:
            raise ValueError(f"prompt set {sample.prompt_set!r} has no cases")
        case_by_id = {case.case_id: case for case in bucket_cases}
        if sample.case_ids:
            chosen = []
            for case_id in sample.case_ids:
                if case_id not in case_by_id:
                    raise ValueError(f"case {sample.prompt_set}/{case_id} not found")
                chosen.append(case_by_id[case_id])
        else:
            chosen = bucket_cases[: sample.count]
        if sample.count is not None and sample.case_ids:
            chosen = chosen[: sample.count]
        for case in chosen:
            key = (case.bucket, case.case_id)
            if key in seen:
                continue
            seen.add(key)
            selected.append(case)
            if max_cases is not None and len(selected) >= max_cases:
                return selected
    return selected


def run_id_for(name: str, family: str | None, variant: str | None, now: datetime | None = None) -> str:
    now = now or utc_now()
    parts = [name, family or "generic", variant or "model", now.strftime("%Y%m%dT%H%M%SZ")]
    return sanitize_run_id("_".join(parts))


def serving_summary_excerpt(path: Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected serving summary object in {display_path(path)}")
    return {
        "path": display_path(path),
        "name": data.get("name"),
        "model": data.get("model"),
        "family": data.get("family"),
        "variant": data.get("variant"),
        "request_count": data.get("request_count"),
        "successful_requests": data.get("successful_requests"),
        "success_rate": data.get("success_rate"),
        "metrics": data.get("metrics", {}),
        "memory": data.get("memory", {}),
        "output_dir": data.get("output_dir"),
    }


def plan_from(
    config: ServingEvalConfig,
    *,
    config_path: Path,
    family: str | None,
    variant: str | None,
    model: str | None,
    base_url: str | None,
    eval_config: Path | None,
    output_dir: Path | None,
    run_id: str | None,
    trials: int | None,
    max_cases: int | None,
    serving_summary: Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    serve_config = load_serve_config(
        config.serving_config,
        family=family,
        variant=variant,
        model=model,
        base_url=base_url,
        env=os.environ,
    )
    resolved_eval_config = eval_config or config.eval_config or family_eval_config(serve_config.family)
    if not resolved_eval_config:
        raise ValueError("serving eval needs --eval-config, eval.config in its config, or a family eval config")
    selected = select_sample_cases(config.samples, prompt_root=REPO_DIR / "evals" / "prompts", max_cases=max_cases)
    actual_trials = trials or config.trials
    actual_run_id = run_id or run_id_for(config.name, serve_config.family, serve_config.variant)
    root = resolve_repo_path(output_dir or config.output_root) / actual_run_id
    command = [
        "./forge",
        "bench",
        "serve-eval",
        "run",
        "--config",
        display_path(config_path),
        "--serving-config",
        display_path(config.serving_config),
        "--eval-config",
        display_path(resolved_eval_config),
        "--output-dir",
        display_path(resolve_repo_path(output_dir or config.output_root)),
        "--run-id",
        actual_run_id,
    ]
    if serve_config.family:
        command.extend(["--family", serve_config.family])
    if serve_config.variant:
        command.extend(["--variant", serve_config.variant])
    if model:
        command.extend(["--model", model])
    if base_url:
        command.extend(["--base-url", base_url])
    if actual_trials != 1:
        command.extend(["--trials", str(actual_trials)])
    if max_cases is not None:
        command.extend(["--max-cases", str(max_cases)])
    if serving_summary:
        command.extend(["--serving-summary", display_path(serving_summary)])
    if dry_run:
        command.append("--dry-run")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "dry_run_only": dry_run,
        "name": config.name,
        "description": config.description,
        "serving": {
            "benchmark_config": display_path(config.serving_config),
            "workload_sources": [display_path(path) for path in serve_config.workload_sources],
            "summary": display_path(serving_summary) if serving_summary else None,
        },
        "eval": {
            "config": display_path(resolved_eval_config),
            "output_dir": display_path(root),
            "trials": actual_trials,
            "timeout_seconds": config.timeout_seconds,
            "case_count": len(selected),
            "total_cases": len(selected) * actual_trials,
            "samples": [
                {
                    "bucket": case.bucket,
                    "category": case.category,
                    "case_id": case.case_id,
                    "expects_json": case.expects_json,
                }
                for case in selected
            ],
        },
        "target": {
            "family": serve_config.family,
            "variant": serve_config.variant,
            "model": serve_config.model,
            "base_url": serve_config.base_url,
            "streaming_benchmark_endpoint": serve_config.stream,
        },
        "execution_contract": {
            "starts_server": False,
            "requires_existing_openai_compatible_endpoint": True,
            "uses_same_endpoint_and_served_model_as_serving_benchmark": True,
            "notes": [
                "Start exactly one server configuration before running this sampled eval.",
                "Use the same family, variant, model alias, base URL, server env, and hardware profile as the serving benchmark under comparison.",
                "Use repeated trials for publishable claims; a dry run only verifies plumbing and artifact shape.",
            ],
        },
        "command": command,
    }


def build_eval_config_for_plan(plan: Mapping[str, Any]) -> Any:
    eval_config = load_eval_config(resolve_repo_path(str((plan.get("eval") or {})["config"])))
    target = plan.get("target") or {}
    backend = dict(eval_config.backend)
    backend["base_url"] = target["base_url"]
    backend["model_alias"] = target["model"]
    timeout_seconds = (plan.get("eval") or {}).get("timeout_seconds")
    if timeout_seconds is not None:
        backend["timeout_seconds"] = int(timeout_seconds)
    return replace(
        eval_config,
        experiment_name=str(plan.get("name") or eval_config.experiment_name),
        family=str(target.get("family") or eval_config.family),
        variant=str(target.get("variant") or eval_config.variant),
        model_id=str(target.get("model") or eval_config.model_id),
        output_dir=str((plan.get("eval") or {})["output_dir"]),
        backend=backend,
    )


def write_serving_eval_card(path: Path, plan: Mapping[str, Any], score_rows: list[dict[str, Any]], serving_summary: Mapping[str, Any] | None) -> None:
    target = plan.get("target") or {}
    eval_data = plan.get("eval") or {}
    serving_data = plan.get("serving") or {}
    sample_rows = []
    for sample in eval_data.get("samples") or []:
        sample_rows.append(
            "| "
            + " | ".join([
                str(sample.get("bucket")),
                str(sample.get("case_id")),
                str(sample.get("category")),
                str(sample.get("expects_json")),
            ])
            + " |"
        )
    if not sample_rows:
        sample_rows.append("| n/a | n/a | n/a | n/a |")

    score_table_rows = []
    for row in score_rows:
        metric = row.get("metric")
        if metric in {"latency_seconds"}:
            continue
        score_table_rows.append(
            "| "
            + " | ".join([
                str(row.get("bucket", "")),
                str(metric or ""),
                str(row.get("value", "")),
                str(row.get("count", "")),
                str(row.get("ci_low", "")),
                str(row.get("ci_high", "")),
            ])
            + " |"
        )
    if not score_table_rows:
        score_table_rows.append("| n/a | n/a | n/a | n/a | n/a | n/a |")

    serving_success = "n/a"
    serving_output = "n/a"
    if serving_summary:
        serving_success = f"{serving_summary.get('successful_requests')}/{serving_summary.get('request_count')} @ {serving_summary.get('success_rate')}"
        serving_output = str(serving_summary.get("output_dir") or serving_summary.get("path") or "n/a")

    lines = [
        f"# Serving Eval Card: {plan.get('name')}",
        "",
        "## Identity",
        "",
        f"- Family: `{target.get('family')}`",
        f"- Variant: `{target.get('variant')}`",
        f"- Model: `{target.get('model')}`",
        f"- Base URL: `{target.get('base_url')}`",
        f"- Eval config: `{eval_data.get('config')}`",
        f"- Serving benchmark config: `{serving_data.get('benchmark_config')}`",
        f"- Serving summary: `{serving_data.get('summary') or 'n/a'}`",
        f"- Output directory: `{eval_data.get('output_dir')}`",
        f"- Trials: `{eval_data.get('trials')}`",
        f"- Cases: `{eval_data.get('case_count')}`",
        "",
        "## Serving Context",
        "",
        f"- Serving benchmark success: `{serving_success}`",
        f"- Serving benchmark output: `{serving_output}`",
        "- This eval does not start a server; it samples quality and behavior against an already-running endpoint.",
        "- Compare only against runs with the same server env, hardware profile, base URL, model alias, sampling, and workload intent.",
        "",
        "## Sampled Cases",
        "",
        "| Bucket | Case | Category | JSON |",
        "|---|---|---|---:|",
        *sample_rows,
        "",
        "## Scores",
        "",
        "| Bucket | Metric | Value | n | CI low | CI high |",
        "|---|---|---:|---:|---:|---:|",
        *score_table_rows,
        "",
        "## Promotion Use",
        "",
        "- Use this as a sampled gate beside TTFT/ITL/tok-sec/memory evidence, not as a full benchmark replacement.",
        "- For ablated variants, interpret low refusal on refusal-target prompts as objective progress only if normal-use, structured-output, and capability samples stay comparable.",
        "- Run the full internal/artifact/external suite before publishing model-quality claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_serving_eval(plan: dict[str, Any], *, config_path: Path, dry_run: bool) -> tuple[Path, dict[str, Any]]:
    eval_config = build_eval_config_for_plan(plan)
    selected_cases = select_sample_cases(
        [
            SampleSpec(
                prompt_set=str(sample["bucket"]),
                case_ids=(str(sample["case_id"]),),
                count=None,
            )
            for sample in (plan.get("eval") or {}).get("samples", [])
        ],
        prompt_root=REPO_DIR / "evals" / "prompts",
    )
    manifest = build_manifest(
        eval_config,
        selected_cases,
        dry_run=dry_run,
        trials=int((plan.get("eval") or {}).get("trials") or 1),
        config_path=resolve_repo_path(str((plan.get("eval") or {})["config"])),
        command=sys.argv,
    )
    serving_summary_path = (plan.get("serving") or {}).get("summary")
    summary_excerpt = serving_summary_excerpt(resolve_repo_path(str(serving_summary_path))) if serving_summary_path else None
    manifest["serving_context"] = {
        "schema_version": SCHEMA_VERSION,
        "serving": plan.get("serving"),
        "target": plan.get("target"),
        "serving_summary": summary_excerpt,
        "sampled_eval_config": display_path(config_path),
    }
    manifest["canonical"]["metadata"]["serving_context"] = manifest["serving_context"]
    manifest["canonical"]["outputs"]["artifacts"]["serving_eval_card_md"] = "serving_eval_card.md"
    manifest["canonical"]["configs"].append({"path": display_path(config_path), "exists": config_path.exists()})
    results = run_cases_with_progress(
        selected_cases,
        eval_config,
        dry_run=dry_run,
        trials=int((plan.get("eval") or {}).get("trials") or 1),
    )
    output_root = resolve_repo_path(str((plan.get("eval") or {})["output_dir"]))
    write_outputs(output_root, manifest, results)
    score_rows = summarize_scores(results)
    (output_root / "serving_eval_context.json").write_text(
        json.dumps(redact_value({"plan": plan, "serving_summary": summary_excerpt}), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_serving_eval_card(output_root / "serving_eval_card.md", plan, score_rows, summary_excerpt)
    return output_root, manifest


def response_rows(eval_dir: Path) -> list[dict[str, Any]]:
    path = resolve_repo_path(eval_dir) / "responses.jsonl"
    if not path.exists():
        raise ValueError(f"serving eval responses not found: {display_path(path)}")
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"expected JSON object in {display_path(path)} line {line_number}")
        rows.append(row)
    return rows


def response_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("bucket") or ""),
        str(row.get("case_id") or ""),
        int(row.get("trial_index") or row.get("trial") or 1),
    )


def metric_regressions(source: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_scores = source.get("scores") or {}
    candidate_scores = candidate.get("scores") or {}
    if not isinstance(source_scores, Mapping) or not isinstance(candidate_scores, Mapping):
        return []
    regressions = []
    for metric, source_value in sorted(source_scores.items()):
        candidate_value = candidate_scores.get(metric)
        if not isinstance(source_value, (int, float)) or not isinstance(candidate_value, (int, float)):
            continue
        delta = float(candidate_value) - float(source_value)
        if float(source_value) >= 1.0 and float(candidate_value) < 1.0:
            bucket, case_id, trial = response_key(source)
            regressions.append(
                {
                    "bucket": bucket,
                    "case_id": case_id,
                    "trial_index": trial,
                    "metric": metric,
                    "source": float(source_value),
                    "candidate": float(candidate_value),
                    "delta": round(delta, 6),
                    "source_notes": source.get("notes") or [],
                    "candidate_notes": candidate.get("notes") or [],
                    "source_latency_seconds": source.get("latency_seconds"),
                    "candidate_latency_seconds": candidate.get("latency_seconds"),
                    "source_response_text": source.get("response_text"),
                    "candidate_response_text": candidate.get("response_text"),
                }
            )
    return regressions


def build_comparison_report(
    *,
    source_eval: Path,
    candidate_eval: Path,
    output_dir: Path,
    run_id: str,
    max_response_chars: int = 1600,
) -> dict[str, Any]:
    source_dir = resolve_repo_path(source_eval)
    candidate_dir = resolve_repo_path(candidate_eval)
    source_by_key = {response_key(row): row for row in response_rows(source_dir)}
    candidate_by_key = {response_key(row): row for row in response_rows(candidate_dir)}
    source_keys = set(source_by_key)
    candidate_keys = set(candidate_by_key)
    regressions = []
    for key in sorted(source_keys & candidate_keys):
        regressions.extend(metric_regressions(source_by_key[key], candidate_by_key[key]))

    for item in regressions:
        for key in ("source_response_text", "candidate_response_text"):
            value = item.get(key)
            if isinstance(value, str) and len(value) > max_response_chars:
                item[key] = value[:max_response_chars].rstrip() + "\n...[truncated]"

    return redact_value(
        {
            "schema_version": COMPARISON_SCHEMA_VERSION,
            "created_at": utc_now().isoformat(),
            "run_id": sanitize_run_id(run_id),
            "source_eval": display_path(source_dir),
            "candidate_eval": display_path(candidate_dir),
            "compared_cases": len(source_keys & candidate_keys),
            "source_only_cases": ["/".join(map(str, key)) for key in sorted(source_keys - candidate_keys)],
            "candidate_only_cases": ["/".join(map(str, key)) for key in sorted(candidate_keys - source_keys)],
            "source_pass_candidate_fail_count": len(regressions),
            "regressions": regressions,
            "output_dir": display_path(output_dir),
            "notes": [
                "This report compares existing serving-eval artifacts; it does not start servers or send requests.",
                "A regression is recorded when a metric scored 1.0 on the source response and below 1.0 on the candidate response for the same bucket/case/trial.",
                "Use this report to debug quantization, fine-tuning, or behavior-edit regressions before promotion.",
            ],
        }
    )


def write_comparison_report(report: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(report["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "serving_eval_comparison.json").write_text(
        json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# Serving Eval Comparison: {report.get('run_id')}",
        "",
        f"- Source: `{report.get('source_eval')}`",
        f"- Candidate: `{report.get('candidate_eval')}`",
        f"- Compared cases: `{report.get('compared_cases')}`",
        f"- Source-pass/candidate-fail metrics: `{report.get('source_pass_candidate_fail_count')}`",
        "",
        "## Regressions",
        "",
    ]
    regressions = report.get("regressions") or []
    if not regressions:
        lines.append("No source-pass/candidate-fail metric regressions found.")
    for index, item in enumerate(regressions, start=1):
        lines.extend(
            [
                f"### {index}. {item.get('bucket')} / {item.get('case_id')} / trial {item.get('trial_index')} / {item.get('metric')}",
                "",
                f"- Source score: `{item.get('source')}`",
                f"- Candidate score: `{item.get('candidate')}`",
                f"- Delta: `{item.get('delta')}`",
                f"- Source notes: `{item.get('source_notes')}`",
                f"- Candidate notes: `{item.get('candidate_notes')}`",
                "",
                "Source response:",
                "",
                "```text",
                str(item.get("source_response_text") or ""),
                "```",
                "",
                "Candidate response:",
                "",
                "```text",
                str(item.get("candidate_response_text") or ""),
                "```",
                "",
            ]
        )
    lines.extend(["## Notes", ""])
    lines.extend(f"- {note}" for note in report.get("notes") or [])
    (output_dir / "serving_eval_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir


def render_plan(plan: Mapping[str, Any]) -> None:
    target = plan.get("target") or {}
    eval_data = plan.get("eval") or {}
    table = Table(title="Serving Eval Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in [
        ("family", target.get("family")),
        ("variant", target.get("variant")),
        ("model", target.get("model")),
        ("base_url", target.get("base_url")),
        ("eval_config", eval_data.get("config")),
        ("cases", eval_data.get("case_count")),
        ("trials", eval_data.get("trials")),
        ("output_dir", eval_data.get("output_dir")),
    ]:
        table.add_row(key, str(value))
    console.print(table)
    for note in (plan.get("execution_contract") or {}).get("notes", []):
        console.print(f"- {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample quality and behavior evals under a serving benchmark config")
    subparsers = parser.add_subparsers(dest="action", required=True)

    def add_common_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Serving eval sample config")
        subparser.add_argument("--serving-config", type=Path, help="Serving benchmark config to inherit endpoint/model resolution from")
        subparser.add_argument("--eval-config", type=Path, help="Eval config; defaults to the family eval config")
        subparser.add_argument("--serving-summary", type=Path, help="Optional summary.json from a serving benchmark run to link")
        subparser.add_argument("--family", help="Optional model family used to resolve served model")
        subparser.add_argument("--variant", default="base", help="Variant within --family")
        subparser.add_argument("--model", help="Served model alias; overrides config/env/family lookup")
        subparser.add_argument("--base-url", help="OpenAI-compatible base URL; overrides config/env")
        subparser.add_argument("--output-dir", type=Path, help="Output root for sampled eval artifacts")
        subparser.add_argument("--run-id", help="Stable run id/output subdirectory")
        subparser.add_argument("--trials", type=int, help="Override configured trial count")
        subparser.add_argument("--max-cases", type=int, help="Use only the first N sampled cases")
        subparser.add_argument("--json", action="store_true", help="Print JSON output")

    plan_parser = subparsers.add_parser("plan", help="Resolve the sampled eval plan without running requests")
    add_common_args(plan_parser)

    run_parser = subparsers.add_parser("run", help="Run the sampled eval against an existing endpoint")
    add_common_args(run_parser)
    run_parser.add_argument("--dry-run", action="store_true", help="Write placeholder eval artifacts without sending requests")

    compare_parser = subparsers.add_parser("compare", help="Compare two existing serving-eval output directories")
    compare_parser.add_argument("--source-eval", type=Path, required=True, help="Source serving-eval output directory")
    compare_parser.add_argument("--candidate-eval", type=Path, required=True, help="Candidate serving-eval output directory")
    compare_parser.add_argument("--output-dir", type=Path, default=REPO_DIR / "reports" / "generated" / "serving_eval_comparisons")
    compare_parser.add_argument("--run-id", required=True)
    compare_parser.add_argument("--max-response-chars", type=int, default=1600)
    compare_parser.add_argument("--write-report", action="store_true", help="Write serving_eval_comparison.json and .md")
    compare_parser.add_argument("--json", action="store_true", help="Print JSON output")

    args = parser.parse_args()

    if args.action == "compare":
        output_dir = resolve_repo_path(args.output_dir) / sanitize_run_id(args.run_id)
        report = build_comparison_report(
            source_eval=args.source_eval,
            candidate_eval=args.candidate_eval,
            output_dir=output_dir,
            run_id=args.run_id,
            max_response_chars=args.max_response_chars,
        )
        if args.write_report:
            write_comparison_report(report)
        if args.json:
            print(json.dumps(redact_value(report), indent=2, sort_keys=True) + "\n")
        else:
            console.print(
                f"Compared {report['compared_cases']} cases; "
                f"{report['source_pass_candidate_fail_count']} source-pass/candidate-fail metrics"
            )
        return

    config_path = resolve_repo_path(args.config)
    config = load_serving_eval_config(
        config_path,
        serving_config=args.serving_config,
        eval_config=args.eval_config,
    )
    plan = plan_from(
        config,
        config_path=config_path,
        family=args.family,
        variant=args.variant if args.family else None,
        model=args.model,
        base_url=args.base_url,
        eval_config=resolve_repo_path(args.eval_config) if args.eval_config else None,
        output_dir=args.output_dir,
        run_id=args.run_id,
        trials=args.trials,
        max_cases=args.max_cases,
        serving_summary=resolve_repo_path(args.serving_summary) if args.serving_summary else None,
        dry_run=True if args.action == "plan" else args.dry_run,
    )

    if args.action == "plan":
        if args.json:
            print(json.dumps(redact_value(plan), indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)
        return

    output_root, manifest = run_serving_eval(plan, config_path=config_path, dry_run=args.dry_run)
    result = {
        "schema_version": SCHEMA_VERSION,
        "output_dir": display_path(output_root),
        "manifest": "manifest.json",
        "serving_eval_card": "serving_eval_card.md",
        "dry_run": args.dry_run,
        "total_cases": manifest.get("total_cases"),
    }
    if args.json:
        print(json.dumps(redact_value(result), indent=2, sort_keys=True) + "\n")
    else:
        console.print(f"Output: {display_path(output_root)}")


if __name__ == "__main__":
    main()
