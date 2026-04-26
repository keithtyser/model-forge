from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_variant(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must be NAME=MODEL_ID_OR_PATH")
    name, model = value.split("=", 1)
    name = name.strip()
    model = model.strip()
    if not name or not model:
        raise argparse.ArgumentTypeError("variant must be NAME=MODEL_ID_OR_PATH")
    return name, model


def run_eval_variant(args: argparse.Namespace, variant: str, model: str) -> Path:
    suffix = f"{args.output_prefix}_{variant}"
    cmd = [
        sys.executable,
        "-m",
        "model_forge.evals.run_eval",
        "--config",
        str(args.config),
        "--output-suffix",
        suffix,
    ]
    if args.max_cases is not None:
        cmd.extend(["--max-cases", str(args.max_cases)])
    if args.trials is not None:
        cmd.extend(["--trials", str(args.trials)])
    if args.dry_run:
        cmd.append("--dry-run")
    for bucket in args.bucket or []:
        cmd.extend(["--bucket", bucket])

    env = os.environ.copy()
    env["MODEL_FORGE_VARIANT"] = variant
    env["MODEL_FORGE_MODEL"] = model
    subprocess.run(cmd, env=env, check=True)

    config_path = Path(args.config).resolve()
    output_root = Path("results")
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text())
        output_root = Path(raw["eval"]["output_dir"])
    except Exception:
        pass
    return output_root / suffix


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model-forge base/ft/abli/combined evaluation matrix")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", action="append", type=parse_variant, required=True, help="Variant mapping like base=Qwen/Qwen3.5-9B")
    parser.add_argument("--output-prefix", required=True, help="Prefix for per-variant output suffixes")
    parser.add_argument("--bucket", action="append", default=None)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-compare", action="store_true", help="Skip comparison report generation")
    parser.add_argument("--compare-output-dir", type=Path)
    args = parser.parse_args()

    variants = dict(args.variant)
    if "base" not in variants:
        parser.error("matrix requires a base=... variant")

    run_dirs = {}
    for variant, model in variants.items():
        run_dirs[variant] = run_eval_variant(args, variant, model)

    compare_output = None
    if not args.no_compare and len(run_dirs) > 1:
        compare_output = args.compare_output_dir or Path("reports/generated") / f"{args.output_prefix}_comparison"
        cmd = [
            sys.executable,
            "-m",
            "model_forge.evals.compare_runs",
            "--base",
            str(run_dirs["base"]),
            "--output-dir",
            str(compare_output),
        ]
        flag_by_variant = {
            "ft": "--ft",
            "abli": "--abli",
            "ft_then_abli": "--ft-then-abli",
            "abli_then_ft": "--abli-then-ft",
        }
        for variant, flag in flag_by_variant.items():
            if variant in run_dirs:
                cmd.extend([flag, str(run_dirs[variant])])
        subprocess.run(cmd, check=True)

    print(json.dumps({
        "ok": True,
        "runs": {name: str(path) for name, path in run_dirs.items()},
        "comparison": str(compare_output) if compare_output else None,
    }, indent=2))


if __name__ == "__main__":
    main()
