#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_forge.evals.run_eval import (
    SCORING_VERSION,
    EvalCase,
    EvalConfig,
    EvalResult,
    collect_cases,
    load_config,
    score_case,
    write_outputs,
)


OUTPUT_ARTIFACTS = {
    "manifest_json": "manifest.json",
    "responses_jsonl": "responses.jsonl",
    "scores_csv": "scores.csv",
    "examples_md": "examples.md",
    "eval_provenance_card_json": "eval_provenance_card.json",
    "eval_provenance_card_md": "eval_provenance_card.md",
}


def find_repo_root(anchor: Path) -> Path:
    for candidate in (anchor, *anchor.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "evals" / "prompts").exists():
            return candidate
    raise SystemExit(f"could not find repo root from {anchor}")


def config_path_from_manifest(manifest: dict[str, Any], run_dir: Path) -> Path | None:
    repo_root = find_repo_root(run_dir)
    canonical_configs = ((manifest.get("canonical") or {}).get("configs") or [])
    for item in canonical_configs:
        path = item.get("path")
        if path:
            candidate = Path(path)
            if candidate.exists():
                return candidate.resolve()
            repo_candidate = repo_root / candidate
            if repo_candidate.exists():
                return repo_candidate.resolve()
    return None


def build_case_index(config_path: Path) -> dict[tuple[str, str], EvalCase]:
    cfg = load_config(config_path)
    repo_root = find_repo_root(config_path)
    cases = collect_cases(repo_root / "evals" / "prompts", cfg.prompt_sets)
    return {(case.bucket, case.case_id): case for case in cases}


def fallback_case(row: dict[str, Any]) -> EvalCase:
    return EvalCase(
        bucket=str(row["bucket"]),
        category=str(row.get("category") or "generic"),
        case_id=str(row["case_id"]),
        prompt=str(row.get("prompt") or ""),
        expects_json=row.get("parsed_json") is not None,
        checks=row.get("checks") or {},
    )


def refresh_manifest_for_rescore(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    output_dir: Path,
    resolved_config: Path | None,
) -> dict[str, Any]:
    rescored_manifest = dict(manifest)
    rescored_manifest["rescored_from"] = str(run_dir)
    rescored_manifest["rescored_at"] = datetime.now(timezone.utc).isoformat()
    rescored_manifest["scoring_version"] = SCORING_VERSION
    if resolved_config:
        rescored_manifest["rescore_config"] = str(resolved_config)

    canonical = rescored_manifest.get("canonical")
    if isinstance(canonical, dict):
        metadata = canonical.get("metadata")
        if isinstance(metadata, dict):
            metadata["scoring_version"] = SCORING_VERSION
        outputs = canonical.get("outputs")
        if isinstance(outputs, dict):
            outputs["output_dir"] = str(output_dir)
            outputs.setdefault("artifacts", OUTPUT_ARTIFACTS)
        canonical["updated_at"] = rescored_manifest["rescored_at"]
        canonical.setdefault("metadata", {})["rescore_from"] = str(run_dir)
        if resolved_config:
            canonical["metadata"]["rescore_config"] = str(resolved_config)
    return rescored_manifest


def rescore_run(run_dir: Path, output_dir: Path, config_path: Path | None = None) -> None:
    manifest_path = run_dir / "manifest.json"
    responses_path = run_dir / "responses.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    if not responses_path.exists():
        raise SystemExit(f"missing responses: {responses_path}")

    manifest = json.loads(manifest_path.read_text())
    resolved_config = config_path.resolve() if config_path else config_path_from_manifest(manifest, run_dir)
    case_index = build_case_index(resolved_config) if resolved_config else {}

    results: list[EvalResult] = []
    for line in responses_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        case = case_index.get((row["bucket"], row["case_id"])) or fallback_case(row)
        scores, notes = score_case(case, row.get("response_text") or "", row.get("parsed_json"))
        results.append(
            EvalResult(
                case=case,
                response_text=row.get("response_text") or "",
                latency_seconds=float(row.get("latency_seconds") or 0.0),
                usage=row.get("usage") or {},
                scores=scores,
                notes=notes,
                trial_index=int(row.get("trial_index") or 1),
                parsed_json=row.get("parsed_json"),
            )
        )

    rescored_manifest = refresh_manifest_for_rescore(
        manifest,
        run_dir=run_dir,
        output_dir=output_dir,
        resolved_config=resolved_config,
    )
    write_outputs(output_dir, rescored_manifest, results)
    print(f"rescored {len(results)} responses")
    print(f"output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score an existing model-forge internal eval run without re-querying the model")
    parser.add_argument("run_dir", type=Path, help="Existing eval output directory containing responses.jsonl")
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML to use for current prompt/check definitions")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory; defaults to <run_dir>_rescored")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir.with_name(f"{run_dir.name}_rescored")
    rescore_run(run_dir, output_dir, args.config)


if __name__ == "__main__":
    main()
