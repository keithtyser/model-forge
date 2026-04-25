from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalConfig:
    experiment_name: str
    model_id: str
    variant: str
    prompt_sets: list[str]
    output_dir: str
    backend: dict[str, Any]


def load_config(path: Path) -> EvalConfig:
    raw = yaml.safe_load(path.read_text())
    return EvalConfig(
        experiment_name=raw["experiment_name"],
        model_id=raw["model"]["id"],
        variant=raw["model"].get("variant", "base"),
        prompt_sets=raw["eval"]["prompt_sets"],
        output_dir=raw["eval"]["output_dir"],
        backend=raw.get("backend", {}),
    )


def collect_prompts(root: Path, prompt_sets: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name in prompt_sets:
        p = root / f"{name}.txt"
        prompts = [line.strip() for line in p.read_text().splitlines() if line.strip() and not line.startswith("#")]
        out[name] = prompts
    return out


def build_manifest(cfg: EvalConfig, prompt_map: dict[str, list[str]], dry_run: bool) -> dict[str, Any]:
    return {
        "experiment_name": cfg.experiment_name,
        "model_id": cfg.model_id,
        "variant": cfg.variant,
        "backend": cfg.backend,
        "prompt_counts": {k: len(v) for k, v in prompt_map.items()},
        "dry_run": dry_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def write_outputs(root: Path, manifest: dict[str, Any], prompt_map: dict[str, list[str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (root / "scores.csv").write_text("variant,metric,value\n")
    sample_lines = []
    for bucket, prompts in prompt_map.items():
        for i, prompt in enumerate(prompts[:2], start=1):
            sample_lines.append(f"## {bucket} :: sample_{i}\n{prompt}\n")
    (root / "examples.md").write_text("\n".join(sample_lines).strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model-forge evaluation")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and emit placeholder outputs")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    repo_root = config_path.parents[2]
    cfg = load_config(config_path)
    prompt_root = repo_root / "evals" / "prompts"
    prompt_map = collect_prompts(prompt_root, cfg.prompt_sets)
    manifest = build_manifest(cfg, prompt_map, dry_run=args.dry_run)
    output_root = repo_root / cfg.output_dir
    write_outputs(output_root, manifest, prompt_map)

    print(json.dumps({
        "ok": True,
        "config": str(config_path),
        "output_dir": str(output_root),
        "prompt_counts": manifest["prompt_counts"],
        "dry_run": args.dry_run,
    }, indent=2))


if __name__ == "__main__":
    main()
