#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml


REPO_DIR = Path(__file__).resolve().parents[1]


def load_family(name: str) -> dict[str, Any]:
    path = REPO_DIR / "configs" / "model_families" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"unknown family {name!r}; expected {path}")
    return yaml.safe_load(path.read_text())


def python_executable() -> str:
    configured = os.environ.get("PYTHON")
    if configured:
        return configured
    venv_python = REPO_DIR / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def models_dir(family: dict[str, Any]) -> Path:
    env_name = family.get("models_dir_env", "MODEL_FORGE_MODELS_DIR")
    raw = os.environ.get(env_name) or family.get("default_models_dir", "~/models")
    return Path(raw).expanduser()


def variant_config(family: dict[str, Any], variant: str) -> dict[str, Any]:
    variants = family.get("variants", {})
    if variant not in variants:
        valid = ", ".join(sorted(variants))
        raise SystemExit(f"unknown variant {variant!r}; valid variants: {valid}")
    return variants[variant]


def variant_local_path(family: dict[str, Any], variant: str) -> Path:
    cfg = variant_config(family, variant)
    local_dir = Path(cfg["local_dir"]).expanduser()
    if local_dir.is_absolute():
        return local_dir
    return models_dir(family) / local_dir


def served_model_name(family: dict[str, Any], variant: str) -> str:
    cfg = variant_config(family, variant)
    return cfg.get("served_model_name") or cfg["repo_id"]


def format_template(template: str, family_name: str, variant: str) -> str:
    return template.format(family=family_name, variant=variant)


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=REPO_DIR, env=env, check=True)


def step(message: str) -> None:
    print()
    print(f"[model-forge] {message}", flush=True)


def action_serve(family: dict[str, Any], family_name: str, variant: str) -> None:
    path = variant_local_path(family, variant)
    if not path.is_dir():
        raise SystemExit(f"model path does not exist: {path}")
    serve = family["serve"]
    env = os.environ.copy()
    env["MODEL_FORGE_MODEL"] = str(path)
    env["MODEL_FORGE_SERVED_MODEL_NAME"] = served_model_name(family, variant)
    env.setdefault("MODEL_FORGE_MODELS_DIR", str(models_dir(family)))
    if serve.get("default_gpu_memory_utilization"):
        env.setdefault("GPU_MEMORY_UTILIZATION", str(serve["default_gpu_memory_utilization"]))
    if serve.get("default_max_model_len"):
        env.setdefault("MAX_MODEL_LEN", str(serve["default_max_model_len"]))
    run([str(REPO_DIR / serve["script"])], env=env)


def action_eval(family: dict[str, Any], family_name: str, variant: str, kind: str) -> None:
    eval_cfg = family["eval"]
    config_key = "artifact_config" if kind == "artifact" else "config"
    suffix_key = {
        "smoke": "smoke_suffix",
        "full": "full_suffix",
        "artifact": "artifact_suffix",
    }[kind]
    suffix = format_template(eval_cfg[suffix_key], family_name, variant)
    env = os.environ.copy()
    env["MODEL_FORGE_MODEL"] = served_model_name(family, variant)
    env["MODEL_FORGE_VARIANT"] = variant
    if kind == "smoke":
        env.setdefault("MODEL_FORGE_MAX_CASES", "4")
    env.setdefault("MODEL_FORGE_BASE_URL", "http://127.0.0.1:8000/v1")
    env.setdefault("MODEL_FORGE_HARDWARE_LABEL", "DGX Spark")
    env.setdefault("MODEL_FORGE_QUANT", "bf16")
    env.setdefault("MODEL_FORGE_CONTEXT_LENGTH", os.environ.get("MAX_MODEL_LEN", "32768"))
    timeout = "480" if kind == "artifact" else "180"
    max_tokens = "4096" if kind == "artifact" else "1200"
    env.setdefault("MODEL_FORGE_TIMEOUT_SECONDS", timeout)
    env.setdefault("MODEL_FORGE_MAX_TOKENS", max_tokens)
    run([
        str(REPO_DIR / "scripts" / "run_dgx_spark_eval.sh"),
        eval_cfg[config_key],
        suffix,
    ], env=env)


def action_compare(family: dict[str, Any], family_name: str) -> None:
    eval_cfg = family["eval"]
    output_root = REPO_DIR / eval_cfg["output_root"]
    cmd = [
        python_executable(),
        "-m",
        "model_forge.evals.compare_runs",
        "--base",
        str(output_root / format_template(eval_cfg["full_suffix"], family_name, "base")),
        "--output-dir",
        family["comparison"]["output_dir"],
    ]
    flag_by_variant = {
        "ft": "--ft",
        "abli": "--abli",
        "ft_then_abli": "--ft-then-abli",
        "abli_then_ft": "--abli-then-ft",
    }
    for variant, flag in flag_by_variant.items():
        if variant in family.get("variants", {}):
            path = output_root / format_template(eval_cfg["full_suffix"], family_name, variant)
            if path.exists():
                cmd.extend([flag, str(path)])
    run(cmd)


def slugify(value: str) -> str:
    value = value.replace(",", "_").replace("/", "_").replace(":", "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "", value)


def assert_served_model(base_url: str, expected: str) -> None:
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise SystemExit(f"[model-forge] ERROR: could not reach {url}: {exc}") from exc
    ids = [item.get("id", "") for item in data.get("data", [])]
    if expected not in ids:
        advertised = ", ".join(ids) or "<none>"
        raise SystemExit(
            f"[model-forge] ERROR: requested external model {expected!r} is not advertised by server\n"
            f"[model-forge] advertised models: {advertised}"
        )


def action_external(family: dict[str, Any], family_name: str, variant: str, tasks: str, dry_run: bool) -> None:
    external = family["external"]
    base_url = os.environ.get("MODEL_FORGE_BASE_URL", "http://127.0.0.1:8000/v1")
    model = served_model_name(family, variant)
    output_dir = str(REPO_DIR / external["output_root"] / variant / f"lm-eval_{slugify(tasks)}")
    concurrency = os.environ.get("MODEL_FORGE_EXTERNAL_CONCURRENCY", "1")

    if not dry_run:
        assert_served_model(base_url, model)

    cmd = [
        python_executable(),
        "-m",
        "model_forge.evals.external",
        "lm-eval",
        "--output-dir",
        output_dir,
    ]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend([
        "--",
        "--model",
        "local-chat-completions",
        "--model_args",
        f"model={model},base_url={base_url.rstrip('/')}/chat/completions,num_concurrent={concurrency},max_retries=3,tokenized_requests=False",
        "--tasks",
        tasks,
        "--apply_chat_template",
        "--fewshot_as_multiturn",
        "--output_path",
        f"{output_dir}/lm_eval",
        "--log_samples",
        "--confirm_run_unsafe_code",
    ])
    if os.environ.get("MODEL_FORGE_EXTERNAL_LIMIT"):
        cmd.extend(["--limit", os.environ["MODEL_FORGE_EXTERNAL_LIMIT"]])

    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "local")
    print(f"[model-forge] external lm-eval family:  {family_name}", flush=True)
    print(f"[model-forge] external lm-eval variant: {variant}", flush=True)
    print(f"[model-forge] external lm-eval model:   {model}", flush=True)
    print(f"[model-forge] external lm-eval tasks:   {tasks}", flush=True)
    print(f"[model-forge] external lm-eval output:  {output_dir}", flush=True)
    run(cmd, env=env)


def action_suite(family: dict[str, Any], family_name: str, variant: str, tasks: str) -> None:
    variant_config(family, variant)
    step(f"eval suite: internal checks ({variant})")
    action_eval(family, family_name, variant, "full")
    step(f"eval suite: artifact generation ({variant})")
    action_eval(family, family_name, variant, "artifact")
    step(f"eval suite: external benchmarks ({variant})")
    action_external(family, family_name, variant, tasks, dry_run=False)
    step("eval suite: comparison report")
    action_compare(family, family_name)


def action_download(family: dict[str, Any], variant: str) -> None:
    variants: list[str]
    if variant == "all":
        variants = list(family.get("variants", {}).keys())
    else:
        variant_config(family, variant)
        variants = [variant]

    env = os.environ.copy()
    model_dir = models_dir(family)
    hf_home = Path(os.environ.get("HF_HOME", str(model_dir / ".hf-cache"))).expanduser()
    env["HF_HOME"] = str(hf_home)
    env["HF_HUB_CACHE"] = os.environ.get("HF_HUB_CACHE", str(hf_home / "hub"))
    env["HF_XET_CACHE"] = os.environ.get("HF_XET_CACHE", str(hf_home / "xet"))
    env["HF_XET_HIGH_PERFORMANCE"] = os.environ.get("HF_XET_HIGH_PERFORMANCE", "1")
    env["HF_XET_NUM_CONCURRENT_RANGE_GETS"] = os.environ.get("HF_XET_NUM_CONCURRENT_RANGE_GETS", "64")
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    model_dir.mkdir(parents=True, exist_ok=True)
    hf_home.mkdir(parents=True, exist_ok=True)

    py = python_executable()
    if os.environ.get("MODEL_FORGE_SKIP_HF_INSTALL") != "1":
        if shutil_which("uv"):
            run(["uv", "pip", "install", "-U", "huggingface_hub[hf_xet]"], env=env)
        else:
            run([py, "-m", "pip", "install", "-U", "huggingface_hub[hf_xet]"], env=env)

    token = env.get("HF_TOKEN")
    if not token:
        import getpass
        token = getpass.getpass("HF token: ")
        env["HF_TOKEN"] = token

    hf = str(REPO_DIR / ".venv" / "bin" / "hf")
    if not Path(hf).exists():
        hf = shutil_which("hf") or "hf"

    print(f"[model-forge] cache: {hf_home}")
    print(f"[model-forge] models: {model_dir}")
    print(f"[model-forge] workers: {os.environ.get('HF_MAX_WORKERS', '32')}")
    print(f"[model-forge] xet range gets: {env['HF_XET_NUM_CONCURRENT_RANGE_GETS']}")
    print(f"[model-forge] hf: {hf}")
    run([hf, "auth", "login", "--token", token, "--force"], env=env)
    run([hf, "auth", "whoami"], env=env)

    workers = os.environ.get("HF_MAX_WORKERS", "32")
    for item in variants:
        cfg = variant_config(family, item)
        target = variant_local_path(family, item)
        print()
        print(f"[model-forge] downloading {cfg['repo_id']} -> {target}")
        run([
            hf,
            "download",
            cfg["repo_id"],
            "--local-dir",
            str(target),
            "--max-workers",
            workers,
            "--token",
            token,
        ], env=env)


def install_external() -> None:
    if shutil_which("uv"):
        run(["uv", "pip", "install", "-e", ".[external]"])
    else:
        run([python_executable(), "-m", "pip", "install", "-e", ".[external]"])


def shutil_which(command: str) -> str | None:
    import shutil
    return shutil.which(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic DGX Spark workflow runner for model-forge families")
    parser.add_argument("family", help="Model family id, e.g. gemma4_26b_a4b")
    parser.add_argument("action", choices=[
        "download",
        "serve",
        "smoke",
        "suite",
        "full",
        "artifact",
        "compare",
        "external",
        "external-dry-run",
        "external-install",
    ])
    parser.add_argument("variant", nargs="?", default="base", help="Variant such as base, ft, or abli")
    parser.add_argument("tasks", nargs="?", default=None, help="External lm-eval tasks, e.g. ifeval")
    args = parser.parse_args()

    family = load_family(args.family)
    family_name = family["name"]
    if args.action == "serve":
        action_serve(family, family_name, args.variant)
    elif args.action == "suite":
        tasks = args.tasks or os.environ.get("MODEL_FORGE_EXTERNAL_TASKS") or family["external"].get("default_tasks", "ifeval")
        action_suite(family, family_name, args.variant, tasks)
    elif args.action in {"smoke", "full", "artifact"}:
        action_eval(family, family_name, args.variant, args.action)
    elif args.action == "compare":
        action_compare(family, family_name)
    elif args.action == "external":
        tasks = args.tasks or os.environ.get("MODEL_FORGE_EXTERNAL_TASKS") or family["external"].get("default_tasks", "ifeval")
        action_external(family, family_name, args.variant, tasks, dry_run=False)
    elif args.action == "external-dry-run":
        tasks = args.tasks or os.environ.get("MODEL_FORGE_EXTERNAL_TASKS") or family["external"].get("default_tasks", "ifeval")
        action_external(family, family_name, args.variant, tasks, dry_run=True)
    elif args.action == "external-install":
        install_external()
    elif args.action == "download":
        action_download(family, args.variant)


if __name__ == "__main__":
    main()
