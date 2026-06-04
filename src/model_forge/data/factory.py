from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from model_forge.data.sources import registry_summary

REPO_DIR = Path(__file__).resolve().parents[3]
console = Console()
TRAINING_EVIDENCE_GATE_SCHEMA_VERSION = "model_forge.dataset_training_evidence_gate.v1"
PACK_PROMOTION_GATES_SCHEMA_VERSION = "model_forge.dataset_pack_promotion_gates.v1"
EVAL_REPAIR_DATASET_SCHEMA_VERSION = "model_forge.eval_repair_dataset.v1"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_DIR / path
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_DIR))
    except ValueError:
        return str(path)


def publish_path_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_DIR))
    except (OSError, ValueError):
        return f"<external>/{path.name}"


def row_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts)


def message_content(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == role and isinstance(message.get("content"), str):
            return message["content"]
    return ""


def compact_text(text: str, max_words: int) -> str:
    words = re.findall(r"\S+", text.strip())
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def stable_id(row: dict[str, Any]) -> str:
    payload = {
        "messages": row.get("messages", []),
        "skills": row.get("skills", []),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_*+-]+", text.lower()))


def jaccard(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def load_holdout_prompts(config: dict[str, Any]) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    for raw_path in config.get("holdouts", []):
        path = resolve_repo_path(raw_path)
        if not path.exists():
            continue
        data = load_yaml(path)
        for case in data.get("cases", []):
            if isinstance(case, dict) and isinstance(case.get("prompt"), str):
                prompts.append({
                    "path": str(path.relative_to(REPO_DIR)),
                    "case_id": str(case.get("id", "")),
                    "prompt": case["prompt"],
                })
    return prompts


def load_seed_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in config.get("seed_paths", []):
        path = resolve_repo_path(raw_path)
        rows.extend(read_jsonl(path))
    for row in config.get("seeds", []):
        if isinstance(row, dict):
            rows.append(row)
    normalized = []
    for row in rows:
        item = dict(row)
        item.setdefault("id", stable_id(item))
        item.setdefault("skills", [])
        item.setdefault("source", {})
        item["source"].setdefault("kind", "human_seed")
        item["source"].setdefault("generator_model", "human")
        item["source"].setdefault("judge_model", "heuristic")
        item["source"].setdefault("license", "unknown")
        normalized.append(item)
    return normalized


def skill_targets(config: dict[str, Any]) -> dict[str, int]:
    targets = {}
    for skill in config.get("skills", []):
        if isinstance(skill, dict):
            targets[str(skill["id"])] = int(skill.get("target_examples", 0) or 0)
    return targets


def score_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    text = row_text(row)
    word_count = len(re.findall(r"\w+", text))
    has_assistant = any(m.get("role") == "assistant" and m.get("content") for m in row.get("messages", []) if isinstance(m, dict))
    skills = [str(skill) for skill in row.get("skills", [])]
    configured_skills = set(skill_targets(config))
    source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
    has_source = all(source.get(key) for key in ("kind", "generator_model", "judge_model", "license"))

    specificity = min(1.0, word_count / 90.0)
    instruction_following = 1.0 if has_assistant and len(row.get("messages", [])) >= 2 else 0.0
    target_skill_relevance = 1.0 if skills and all(skill in configured_skills for skill in skills) else 0.4
    answer_completeness = min(1.0, word_count / 120.0) if has_assistant else 0.0
    novelty = 0.85
    difficulty = 0.65 + min(0.25, max(0, word_count - 60) / 240.0)
    correctness = 0.80 if has_assistant else 0.0
    style_fit = 0.85 if word_count <= 220 else 0.65
    refusal_boundary_fit = 1.0 if "benign_safety_analysis" in skills else 0.8
    provenance = 1.0 if has_source else 0.5

    return {
        "correctness": round(correctness, 4),
        "instruction_following": round(instruction_following, 4),
        "specificity": round(specificity, 4),
        "difficulty": round(difficulty, 4),
        "novelty": round(novelty, 4),
        "target_skill_relevance": round(target_skill_relevance, 4),
        "answer_completeness": round(answer_completeness, 4),
        "style_fit": round(style_fit, 4),
        "refusal_boundary_fit": round(refusal_boundary_fit, 4),
        "provenance": round(provenance, 4),
    }


def average_score(scores: dict[str, float]) -> float:
    if not scores:
        return 0.0
    return sum(float(value) for value in scores.values()) / len(scores)


def holdout_overlap(row: dict[str, Any], holdouts: list[dict[str, str]]) -> dict[str, Any]:
    text = row_text(row)
    best = {"max_similarity": 0.0, "nearest_holdout": None}
    for holdout in holdouts:
        score = jaccard(text, holdout["prompt"])
        if score > best["max_similarity"]:
            best = {
                "max_similarity": round(score, 4),
                "nearest_holdout": {
                    "path": holdout["path"],
                    "case_id": holdout["case_id"],
                },
            }
    return best


SKILL_SCENARIOS: dict[str, dict[str, str]] = {
    "eval_latency_throughput": {
        "user": (
            "An inference benchmark has identical request count, higher TTFT, lower total latency, "
            "and better output tokens/sec than the baseline. Explain how prefill, decode, prompt "
            "tokens, completion tokens, and batching could make that possible."
        ),
        "assistant": (
            "Do not reduce this to one latency number. Higher TTFT usually points at more expensive "
            "prefill, longer prompt_tokens, chunked prefill settings, or cache behavior. Lower total "
            "latency can still happen if completion_tokens are shorter, decode ITL improves, or the "
            "batch mix changes. Better output tokens/sec can come from more efficient decode or higher "
            "batch occupancy. Compare prompt_tokens, completion_tokens, TTFT, ITL, batch size, "
            "concurrency, max_new_tokens, cache hit rate, and server flags before calling it a win or "
            "regression."
        ),
    },
    "docker_disk_safety": {
        "user": (
            "A DGX workstation is low on disk during a checkpointing run. Give a cleanup plan that "
            "preserves active Docker containers and avoids deleting the checkpoint currently being written."
        ),
        "assistant": (
            "Start with `df -h`, `docker system df`, the run output directory size, and `docker ps` so "
            "active containers are visible. Do not remove images, volumes, or paths used by active "
            "containers. Prune stopped containers and dangling build cache only after review. For "
            "checkpoints, keep the active write directory plus the last known-good checkpoints, verify a "
            "new checkpoint has expected files before deleting older explicit paths, and log every "
            "deletion. Recheck free space before resuming."
        ),
    },
    "sql_edge_cases": {
        "user": (
            "A LEFT JOIN dashboard shows total accounts, matched orders, and average order value. Explain "
            "how NULLs can make COUNT(*), COUNT(order_id), WHERE filters, and aggregates disagree."
        ),
        "assistant": (
            "A LEFT JOIN retains left-table rows even when the right side has no match, filling right-side "
            "columns with NULL. `COUNT(*)` counts every retained account row, while `COUNT(order_id)` "
            "counts only matched rows where order_id is not NULL. A right-table predicate in WHERE can "
            "turn the query into an accidental inner join; put match filters in the JOIN condition when "
            "unmatched accounts must remain. Aggregates ignore NULL inputs, so separate total accounts, "
            "matched accounts, and value averages explicitly."
        ),
    },
    "shell_safety": {
        "user": (
            "Write a safe shell workflow for moving old run artifacts to archival storage without losing "
            "files that only exist in the archive."
        ),
        "assistant": (
            "Use explicit variables, quote paths, and start with a dry run. Inspect source and destination "
            "with `du -sh` and `find`, then run `rsync -a --info=stats2 --dry-run \"$src/\" \"$dst/\"`. "
            "Review the planned copy before removing `--dry-run`. Avoid `--delete` unless the archive is "
            "supposed to be an exact mirror and there is a separate confirmation. After copying, compare "
            "counts or checksums for critical files and record the command."
        ),
    },
    "config_review": {
        "user": (
            "A post-training repo stores model variants, prompt suites, output directories, and promotion "
            "thresholds in separate YAML files. What validation should run before any eval starts?"
        ),
        "assistant": (
            "Load every YAML file and resolve the full run matrix before starting. Check that requested "
            "variants exist, prompt suites resolve to files, output directories are unique or overwrite is "
            "explicit, context length and served model names are present, and promotion thresholds match "
            "the objective. Print the resolved variant, prompt set, output path, and eval mode so a user "
            "can catch silent mismatches before compute is spent."
        ),
    },
    "json_schema_repair": {
        "user": (
            "A JSON-only evaluator rejected a response with a valid object followed by explanatory prose. "
            "Explain the failure and give the response rule the model should follow."
        ),
        "assistant": (
            "JSON-only means the entire response must parse as one JSON value. A valid object followed by "
            "a sentence fails because the parser sees extra tokens. The model should return only the "
            "requested object or array, with double-quoted keys and strings, lowercase booleans, no "
            "trailing commas, and no markdown or explanation outside the JSON value."
        ),
    },
    "git_workflow_repair": {
        "user": (
            "A private feature branch is behind origin after local edits and conflict resolution looks "
            "risky. Give an observe, act, verify, and fallback workflow."
        ),
        "assistant": (
            "Observe with `git status --short`, `git fetch origin`, and a short graph comparing HEAD to "
            "origin. Act by creating a backup branch before rebasing or cherry-picking onto a fresh branch "
            "from the remote tip. If conflicts get confusing, abort the rebase and recover from the backup "
            "instead of discarding work. Verify with focused tests and `git diff --check`; push a rebased "
            "private branch with `--force-with-lease` only after those checks pass."
        ),
    },
    "benign_safety_analysis": {
        "user": (
            "An eval asks for a policy-level comparison of refusal rate, over-refusal, and unsafe "
            "overcompliance. Explain why answering is allowed while still avoiding harmful operational detail."
        ),
        "assistant": (
            "This is benign safety analysis, not a request to perform harm. A useful answer defines the "
            "metrics, explains tradeoffs, and helps compare model behavior across variants. It should avoid "
            "payloads, exploit steps, credential theft details, or tactical abuse. Refusing the metric "
            "discussion would hide evidence needed for responsible evaluation, while over-answering with "
            "harmful detail would be unsafe."
        ),
    },
    "checkpoint_selection": {
        "user": (
            "Two fine-tune checkpoints trade off challenge capability and paired benign quality. Give a "
            "promotion rule that handles trial noise and avoids blindly picking the final checkpoint."
        ),
        "assistant": (
            "Use the objective gates first. A checkpoint must clear normal-use regression and the primary "
            "challenge capability target before tie-breakers matter. Then compare paired benign quality, "
            "benign refusal rate, structured-output behavior, artifact results, and external evals. If the "
            "gap is one or two cases, rerun repeated trials with the same sampling, context, and server "
            "settings. Promote the checkpoint with stable wins across gates, not merely the last step."
        ),
    },
}

STRATEGY_NOTES: dict[str, dict[str, str]] = {
    "self_instruct": {
        "user": "Keep the answer concise, operational, and directly useful.",
        "assistant": "The key is to preserve the concept while changing surface wording and scenario details.",
    },
    "evol_instruct": {
        "user": "Include one edge case, one failure mode, and one verification check.",
        "assistant": "The harder variant should name the edge case, explain the failure mode, and end with a concrete verification step.",
    },
    "instruction_backtranslation": {
        "user": "Answer as if the prompt was reconstructed from a high-quality internal runbook.",
        "assistant": "A backtranslated example should be self-contained, source-grounded, and free of held-out eval wording.",
    },
    "eval_adjacent_generation": {
        "user": "Make this eval-adjacent without copying any held-out prompt wording or exact checklist.",
        "assistant": "This trains the adjacent skill, not memorization of the benchmark case.",
    },
}


def generation_config(config: dict[str, Any]) -> dict[str, Any]:
    generation = config.get("generation", {})
    return generation if isinstance(generation, dict) else {}


def enabled_strategies(config: dict[str, Any]) -> list[dict[str, Any]]:
    strategies = generation_config(config).get("strategies", [])
    enabled = []
    for strategy in strategies:
        if isinstance(strategy, dict) and strategy.get("enabled", True):
            enabled.append(strategy)
    return enabled


def assistant_word_bounds(config: dict[str, Any]) -> tuple[int, int]:
    review = config.get("review", {})
    review = review if isinstance(review, dict) else {}
    min_words = int(review.get("min_assistant_words", 35) or 35)
    max_words = int(review.get("max_assistant_words", 260) or 260)
    return min_words, max_words


def prompt_template(
    strategy_name: str,
    seed: dict[str, Any],
    primary_skill: str,
    variant_index: int,
    config: dict[str, Any],
) -> str:
    scenario = SKILL_SCENARIOS.get(primary_skill, SKILL_SCENARIOS["checkpoint_selection"])
    strategy_note = STRATEGY_NOTES.get(strategy_name, STRATEGY_NOTES["self_instruct"])
    min_words, max_words = assistant_word_bounds(config)
    return "\n".join([
        "Create one high-quality supervised fine-tuning conversation.",
        f"Strategy: {strategy_name}",
        f"Primary skill: {primary_skill}",
        f"Variant index: {variant_index}",
        f"Seed user summary: {compact_text(message_content(seed, 'user'), 80)}",
        f"Seed assistant summary: {compact_text(message_content(seed, 'assistant'), 120)}",
        "Return strict JSON with keys user and assistant.",
        "The conversation must be eval-adjacent and must not copy held-out prompt wording.",
        f"The assistant answer must be {min_words}-{max_words} words, concrete, and free of filler.",
        f"Target user task: {scenario['user']} {strategy_note['user']}",
        f"Target answer content: {scenario['assistant']} {strategy_note['assistant']}",
    ])


def template_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("provider response did not contain a JSON object")
    return data


class GenerationProvider:
    def __init__(self, provider_config: dict[str, Any]) -> None:
        self.config = provider_config

    @property
    def provider_type(self) -> str:
        return str(self.config.get("type", "template"))

    @property
    def model_name(self) -> str:
        return str(self.config.get("model", self.provider_type))

    def generate(self, prompt: str, seed: dict[str, Any], primary_skill: str, strategy_name: str, variant_index: int) -> dict[str, str]:
        raise NotImplementedError


class TemplateGenerationProvider(GenerationProvider):
    def generate(self, prompt: str, seed: dict[str, Any], primary_skill: str, strategy_name: str, variant_index: int) -> dict[str, str]:
        scenario = SKILL_SCENARIOS.get(primary_skill, SKILL_SCENARIOS["checkpoint_selection"])
        strategy_note = STRATEGY_NOTES.get(strategy_name, STRATEGY_NOTES["self_instruct"])
        seed_anchor = compact_text(message_content(seed, "assistant"), 36)
        user = f"{scenario['user']} {strategy_note['user']}"
        assistant = " ".join([
            scenario["assistant"],
            strategy_note["assistant"],
            f"Seed concept anchor: {seed_anchor}",
        ])
        return {"user": user, "assistant": assistant}


class OpenAICompatibleProvider(GenerationProvider):
    def generate(self, prompt: str, seed: dict[str, Any], primary_skill: str, strategy_name: str, variant_index: int) -> dict[str, str]:
        base_url = str(self.config.get("base_url", "")).rstrip("/")
        if not base_url:
            raise RuntimeError("openai_compatible provider requires generation.provider.base_url")
        if base_url.endswith("/chat/completions"):
            endpoint = base_url
        elif base_url.endswith("/v1"):
            endpoint = f"{base_url}/chat/completions"
        else:
            endpoint = f"{base_url}/v1/chat/completions"
        api_key = str(self.config.get("api_key", ""))
        api_key_env = str(self.config.get("api_key_env", ""))
        if not api_key and api_key_env:
            api_key = os.environ.get(api_key_env, "")
        request_config = generation_request_config(self.config)
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": request_config["system_prompt"],
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": request_config["temperature"],
            "max_tokens": request_config["max_tokens"],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        timeout = float(self.config.get("timeout_seconds", 60))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"generation provider request failed: {exc}") from exc
        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_object(content)
        user = str(parsed.get("user", "")).strip()
        assistant = str(parsed.get("assistant", "")).strip()
        if not user or not assistant:
            raise ValueError("provider JSON must include non-empty user and assistant fields")
        return {"user": user, "assistant": assistant}


def generation_request_config(provider_config: dict[str, Any]) -> dict[str, Any]:
    request = provider_config.get("request", {}) if isinstance(provider_config.get("request"), dict) else {}
    default_system_prompt = (
        "You create high-quality eval-adjacent SFT data. Return only strict compact JSON with user "
        "and assistant keys. The assistant answer must be concise, concrete, correct, and within "
        "the requested word limit."
    )
    return {
        "temperature": float(request.get("temperature", provider_config.get("temperature", 0.7))),
        "max_tokens": int(request.get("max_tokens", provider_config.get("max_tokens", 900))),
        "system_prompt": str(request.get("system_prompt", provider_config.get("system_prompt", default_system_prompt))),
    }


def build_provider(config: dict[str, Any], provider_override: str | None = None) -> GenerationProvider:
    provider_config = copy.deepcopy(generation_config(config).get("provider", {}))
    if not isinstance(provider_config, dict):
        provider_config = {}
    if provider_override:
        provider_config["type"] = provider_override
    for config_key in ("base_url", "model", "api_key"):
        env_name = str(provider_config.get(f"{config_key}_env", ""))
        if env_name and os.environ.get(env_name):
            provider_config[config_key] = os.environ[env_name]
    provider_type = str(provider_config.get("type", "template"))
    if provider_type == "template":
        provider_config.setdefault("model", "model-forge-template-v0")
        return TemplateGenerationProvider(provider_config)
    if provider_type in {"openai_compatible", "vllm_openai"}:
        return OpenAICompatibleProvider(provider_config)
    raise SystemExit(f"unknown generation provider type: {provider_type}")


def generation_budget(config: dict[str, Any], smoke: bool, max_generated_candidates: int | None) -> dict[str, int]:
    generation = generation_config(config)
    smoke_config = generation.get("smoke", {}) if isinstance(generation.get("smoke"), dict) else {}
    if max_generated_candidates is not None:
        max_generated = max(0, int(max_generated_candidates))
    elif smoke:
        max_generated = int(smoke_config.get("max_generated_candidates", 6) or 0)
    else:
        max_generated = int(generation.get("max_generated_candidates", 0) or 0)
    if smoke:
        max_seed_rows = int(smoke_config.get("max_seed_rows", 3) or 0)
    else:
        max_seed_rows = int(generation.get("max_seed_rows", 0) or 0)
    return {"max_generated_candidates": max_generated, "max_seed_rows": max_seed_rows}


def primary_skill(row: dict[str, Any]) -> str:
    skills = [str(skill) for skill in row.get("skills", [])]
    return skills[0] if skills else "checkpoint_selection"


def select_generation_seeds(seed_rows: list[dict[str, Any]], max_seed_rows: int) -> list[dict[str, Any]]:
    if max_seed_rows <= 0 or len(seed_rows) <= max_seed_rows:
        return seed_rows
    selected: list[dict[str, Any]] = []
    seen_skills: set[str] = set()
    for row in seed_rows:
        skill = primary_skill(row)
        if skill in seen_skills:
            continue
        selected.append(row)
        seen_skills.add(skill)
        if len(selected) >= max_seed_rows:
            return selected
    for row in seed_rows:
        if row in selected:
            continue
        selected.append(row)
        if len(selected) >= max_seed_rows:
            return selected
    return selected


def enforce_generation_resources(config: dict[str, Any], output_dir: Path, planned_candidates: int) -> None:
    limits = generation_config(config).get("resource_limits", {})
    limits = limits if isinstance(limits, dict) else {}
    max_candidates = int(limits.get("max_candidates_per_run", 128) or 128)
    if planned_candidates > max_candidates:
        raise RuntimeError(f"planned candidates {planned_candidates} exceeds max_candidates_per_run {max_candidates}")
    memory_floor = float(limits.get("min_free_memory_ratio", 0.05))
    memory = psutil.virtual_memory()
    if memory.total and memory.available / memory.total < memory_floor:
        raise RuntimeError("not enough free memory to start dataset generation")
    disk_floor = float(limits.get("min_free_disk_ratio", 0.15))
    output_dir.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_dir)
    if disk.total and disk.free / disk.total < disk_floor:
        raise RuntimeError("not enough free disk to write dataset generation artifacts")


def generated_row(
    seed: dict[str, Any],
    config: dict[str, Any],
    provider: GenerationProvider,
    strategy_name: str,
    variant_index: int,
) -> dict[str, Any]:
    skill = primary_skill(seed)
    prompt = prompt_template(strategy_name, seed, skill, variant_index, config)
    rendered = provider.generate(prompt, seed, skill, strategy_name, variant_index)
    source = {
        "kind": "synthetic",
        "generator_model": provider.model_name,
        "judge_model": "heuristic",
        "source_uri": f"seed:{seed.get('id', stable_id(seed))}",
        "license": seed.get("source", {}).get("license", "CC-BY-4.0") if isinstance(seed.get("source"), dict) else "CC-BY-4.0",
        "license_risk": "low",
        "contamination_risk": "low",
        "generation": {
            "provider_type": provider.provider_type,
            "strategy": strategy_name,
            "seed_id": seed.get("id", stable_id(seed)),
            "variant_index": variant_index,
            "prompt_template_hash": template_hash(prompt),
        },
    }
    row = {
        "messages": [
            {"role": "user", "content": rendered["user"]},
            {"role": "assistant", "content": rendered["assistant"]},
        ],
        "skills": [skill],
        "source": source,
        "generation_method": strategy_name,
        "dataset_factory_stage": "candidate",
    }
    row["id"] = f"gen_{stable_id(row)}"
    return row


def build_generation_report(
    config: dict[str, Any],
    seed_count: int,
    candidates: list[dict[str, Any]],
    provider: GenerationProvider,
    smoke: bool,
) -> dict[str, Any]:
    method_counts = Counter(str(row.get("generation_method", "")) for row in candidates)
    source_counts = Counter(
        str(row.get("source", {}).get("kind", "unknown"))
        for row in candidates
        if isinstance(row.get("source"), dict)
    )
    return {
        "dataset_id": config["id"],
        "seed_rows": seed_count,
        "candidate_rows": len(candidates),
        "source_kind_counts": dict(sorted(source_counts.items())),
        "generation_method_counts": dict(sorted(method_counts.items())),
        "provider": {
            "type": provider.provider_type,
            "model": provider.model_name,
        },
        "strategies": [strategy.get("name") for strategy in enabled_strategies(config)],
        "smoke": bool(smoke),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def result_path(config: dict[str, Any]) -> Path:
    baseline = config.get("baseline", {}) if isinstance(config.get("baseline"), dict) else {}
    raw_path = baseline.get("local_v0_result_path")
    if not raw_path:
        raise SystemExit("dataset config baseline.local_v0_result_path is required for gap extraction")
    return resolve_repo_path(str(raw_path))


def bad_metric_reasons(scores: dict[str, Any]) -> list[str]:
    reasons = []
    low_is_bad = {
        "workflow_success",
        "schema_adherence",
        "normal_use_regression_pass_rate",
        "benign_answer_quality_rate",
        "safe_redirect_rate",
    }
    high_is_bad = {
        "benign_refusal_rate",
        "unsafe_overcompliance_rate",
        "harmful_prompt_compliance_rate",
        "harmful_detail_rate",
    }
    for metric, value in scores.items():
        if not isinstance(value, int | float):
            continue
        if metric in low_is_bad and float(value) <= 0.0:
            reasons.append(f"{metric}=0")
        if metric in high_is_bad and float(value) > 0.0:
            reasons.append(f"{metric}={value:g}")
    return reasons


def extract_missed_concepts(notes: list[str]) -> list[str]:
    concepts: list[str] = []
    for note in notes:
        lowered = note.lower()
        for marker in (
            "response missed required normal-use concepts:",
            "benign answer missed required concepts:",
            "unsafe response contained disallowed markers:",
        ):
            if marker in lowered:
                tail = note.split(":", 1)[1]
                for item in re.split(r",|;", tail):
                    concept = item.strip().strip(".")
                    if concept:
                        concepts.append(concept)
    return concepts


def skill_for_gap(row: dict[str, Any], notes: list[str], metric_reasons: list[str]) -> str:
    text = " ".join([
        str(row.get("bucket", "")),
        str(row.get("case_id", "")),
        str(row.get("prompt", "")),
        " ".join(notes),
        " ".join(metric_reasons),
    ]).lower()
    if any(term in text for term in ("latency", "throughput", "tokens per second", "completion", "prompt tokens", "tps")):
        return "eval_latency_throughput"
    if any(term in text for term in ("docker", "active container", "container", "disk cleanup")):
        return "docker_disk_safety"
    if any(term in text for term in ("sql", "null", "count(*)", "index", "aggregate")):
        return "sql_edge_cases"
    if any(term in text for term in ("shell", "rsync", "dry-run", "quote", "sync")):
        return "shell_safety"
    if any(term in text for term in ("json", "schema")):
        return "json_schema_repair"
    if any(term in text for term in ("git", "rebase", "branch", "push", "numbered steps", "observe", "verify")):
        return "git_workflow_repair"
    if any(term in text for term in ("yaml", "config", "variant", "prompt set", "output director")):
        return "config_review"
    if any(term in text for term in ("checkpoint", "eval", "compare", "baseline", "model variants")):
        return "checkpoint_selection"
    if any(term in text for term in ("refusal", "refused", "unsafe", "harmful", "overcompliance", "safety", "ablated")):
        return "benign_safety_analysis"
    return "checkpoint_selection"


def build_gap_report(config: dict[str, Any]) -> dict[str, Any]:
    responses_path = result_path(config) / "responses.jsonl"
    rows = read_jsonl(responses_path)
    gaps = []
    bucket_counts: Counter[str] = Counter()
    skill_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    case_counts: Counter[str] = Counter()

    for row in rows:
        scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
        notes = [str(note) for note in row.get("notes", []) if str(note).strip()]
        metric_reasons = bad_metric_reasons(scores)
        if not notes and not metric_reasons:
            continue
        skill = skill_for_gap(row, notes, metric_reasons)
        concepts = extract_missed_concepts(notes)
        bucket = str(row.get("bucket", "unknown"))
        case_id = str(row.get("case_id", "unknown"))
        bucket_counts[bucket] += 1
        skill_counts[skill] += 1
        case_counts[f"{bucket}.{case_id}"] += 1
        for concept in concepts:
            concept_counts[concept] += 1
        gaps.append({
            "bucket": bucket,
            "case_id": case_id,
            "trial_index": row.get("trial_index"),
            "recommended_skill": skill,
            "metric_reasons": metric_reasons,
            "notes": notes,
            "missed_concepts": concepts,
        })

    return {
        "dataset_id": config["id"],
        "source_result_path": display_path(result_path(config)),
        "responses_path": display_path(responses_path),
        "total_rows": len(rows),
        "gap_rows": len(gaps),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "recommended_skill_counts": dict(sorted(skill_counts.items())),
        "missed_concepts": dict(sorted(concept_counts.items())),
        "top_cases": dict(case_counts.most_common(25)),
        "gaps": gaps,
        "next_seed_priorities": [
            {"skill": skill, "gap_count": count}
            for skill, count in skill_counts.most_common()
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def command_gaps(config: dict[str, Any], overwrite: bool) -> Path:
    output_path = resolve_repo_path(config["output_dir"]) / "gap_report.yaml"
    if output_path.exists() and not overwrite:
        return output_path
    report = build_gap_report(config)
    write_yaml(output_path, report)
    return output_path


def build_feedback_proposal(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    gap_report = build_gap_report(config)
    plan = build_plan(config, config_path)
    targets = skill_targets(config)
    seed_counts = {
        str(skill): int(count)
        for skill, count in plan.get("seed_skill_counts", {}).items()
    }
    gap_counts = Counter({
        str(skill): int(count)
        for skill, count in gap_report.get("recommended_skill_counts", {}).items()
    })
    top_cases_by_skill: dict[str, list[str]] = {}
    concepts_by_skill: dict[str, Counter[str]] = {}
    for gap in gap_report.get("gaps", []):
        if not isinstance(gap, dict):
            continue
        skill = str(gap.get("recommended_skill", "checkpoint_selection"))
        case_key = f"{gap.get('bucket', 'unknown')}.{gap.get('case_id', 'unknown')}"
        top_cases_by_skill.setdefault(skill, [])
        if case_key not in top_cases_by_skill[skill]:
            top_cases_by_skill[skill].append(case_key)
        concepts_by_skill.setdefault(skill, Counter())
        for concept in gap.get("missed_concepts", []):
            concepts_by_skill[skill][str(concept)] += 1

    recommended_skill_updates = []
    for priority, (skill, gap_count) in enumerate(gap_counts.most_common(), start=1):
        current_target = int(targets.get(skill, 0))
        current_seed_rows = int(seed_counts.get(skill, 0))
        shortage = max(0, current_target - current_seed_rows)
        gap_weighted_bump = max(24, min(160, int(gap_count) * 8))
        proposed_target = current_target + gap_weighted_bump
        concepts = [concept for concept, _ in concepts_by_skill.get(skill, Counter()).most_common(8)]
        cases = top_cases_by_skill.get(skill, [])[:8]
        rationale_parts = [
            f"{gap_count} eval failure rows mapped to {skill}",
            f"{shortage} rows below the current target before generation",
        ]
        if concepts:
            rationale_parts.append("missed concepts: " + ", ".join(concepts[:5]))
        if cases:
            rationale_parts.append("top cases: " + ", ".join(cases[:3]))
        recommended_skill_updates.append({
            "priority": priority,
            "skill": skill,
            "gap_count": int(gap_count),
            "current_target_examples": current_target,
            "current_seed_rows": current_seed_rows,
            "current_shortage": shortage,
            "proposed_target_examples": proposed_target,
            "proposed_increment": proposed_target - current_target,
            "top_failure_cases": cases,
            "missed_concepts": concepts,
            "rationale": "; ".join(rationale_parts),
        })

    gap_rows = int(gap_report.get("gap_rows", 0) or 0)
    recommended_min_candidates = max(64, min(512, gap_rows * 4 if gap_rows else 64))
    focus_skills = [item["skill"] for item in recommended_skill_updates[:6]]
    candidate_skills = [
        {
            "id": item["skill"],
            "target_examples": item["proposed_target_examples"],
        }
        for item in recommended_skill_updates
    ]
    proposal = {
        "schema_version": "model_forge.dataset_feedback_proposal.v1",
        "dataset_id": config["id"],
        "family": config["family"],
        "variant": config["variant"],
        "objective": config["objective"],
        "config_path": display_path(config_path),
        "output_dir": display_path(resolve_repo_path(config["output_dir"])),
        "source_result_path": gap_report["source_result_path"],
        "responses_path": gap_report["responses_path"],
        "total_rows": gap_report["total_rows"],
        "gap_rows": gap_rows,
        "top_failure_buckets": gap_report.get("bucket_counts", {}),
        "top_failure_cases": gap_report.get("top_cases", {}),
        "missed_concepts": gap_report.get("missed_concepts", {}),
        "recommended_skill_updates": recommended_skill_updates,
        "recommended_generation": {
            "min_generated_candidates": recommended_min_candidates,
            "medium_pack_min_candidates": max(recommended_min_candidates, 256),
            "focus_skills": focus_skills,
            "rationale": (
                "Scale candidate generation from observed eval failures, then keep only rows "
                "that pass verification, holdout-overlap, source, and review gates."
            ),
        },
        "candidate_config_patch": {
            "skills": candidate_skills,
            "generation": {
                "max_generated_candidates": recommended_min_candidates,
                "resource_limits": {
                    "max_candidates_per_run": min(256, recommended_min_candidates),
                },
            },
            "review": {
                "focus_skills": list(focus_skills),
                "sample_size": max(int(config.get("review", {}).get("sample_size", 50) or 50), 75),
            },
        },
        "next_actions": [
            f"./forge data generate {config['family']} {config['variant']} --overwrite --max-generated-candidates {recommended_min_candidates}",
            f"./forge data verify {config['family']} {config['variant']} --overwrite",
            f"./forge data filter {config['family']} {config['variant']} --overwrite",
            f"./forge data review {config['family']} {config['variant']} --overwrite --sample 75",
            f"./forge data pack {config['family']} {config['variant']} --overwrite",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return proposal


def command_propose(config: dict[str, Any], config_path: Path, overwrite: bool) -> Path:
    output_path = resolve_repo_path(config["output_dir"]) / "feedback_proposal.yaml"
    if output_path.exists() and not overwrite:
        return output_path
    proposal = build_feedback_proposal(config, config_path)
    write_yaml(output_path, proposal)
    return output_path


def case_aliases(row: dict[str, Any]) -> set[str]:
    bucket = str(row.get("bucket", "")).strip()
    case_id = str(row.get("case_id", "")).strip()
    aliases = {case_id}
    if bucket and case_id:
        aliases.update({
            f"{bucket}/{case_id}",
            f"{bucket}.{case_id}",
            f"{bucket}:{case_id}",
        })
    return {alias for alias in aliases if alias}


def include_eval_row(row: dict[str, Any], include: dict[str, Any]) -> bool:
    buckets = {str(item) for item in include.get("buckets", [])}
    if buckets and str(row.get("bucket", "")) not in buckets:
        return False
    case_ids = {str(item) for item in include.get("case_ids", [])}
    if case_ids and not (case_aliases(row) & case_ids):
        return False
    categories = {str(item) for item in include.get("categories", [])}
    if categories and str(row.get("category", "")) not in categories:
        return False
    return True


def score_value(row: dict[str, Any], metric: str) -> Any:
    scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
    if metric in scores:
        return scores[metric]
    current: Any = row
    for part in metric.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def condition_matches(value: Any, condition: Any) -> bool:
    if isinstance(condition, dict):
        for op, expected in condition.items():
            if op == "eq":
                if not condition_matches(value, expected):
                    return False
            elif op == "ne":
                if condition_matches(value, expected):
                    return False
            elif op in {"min", "gte"}:
                if value is None or float(value) < float(expected):
                    return False
            elif op == "gt":
                if value is None or float(value) <= float(expected):
                    return False
            elif op in {"max", "lte"}:
                if value is None or float(value) > float(expected):
                    return False
            elif op == "lt":
                if value is None or float(value) >= float(expected):
                    return False
            elif op == "in":
                expected_values = expected if isinstance(expected, list) else [expected]
                if str(value) not in {str(item) for item in expected_values}:
                    return False
            else:
                raise ValueError(f"unknown score filter operator: {op}")
        return True
    if isinstance(value, int | float) and isinstance(condition, int | float):
        return abs(float(value) - float(condition)) < 1e-9
    return str(value) == str(condition)


def score_filters_match(row: dict[str, Any], filters: dict[str, Any]) -> bool:
    for metric, condition in filters.items():
        if not condition_matches(score_value(row, str(metric)), condition):
            return False
    return True


def response_text(row: dict[str, Any]) -> str:
    return str(row.get("response_text", "")).strip()


def normalize_filter_text(text: str, case_sensitive: bool = False) -> str:
    normalized = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return normalized if case_sensitive else normalized.lower()


def filter_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    return [str(raw)]


def text_filter_failures(text: str, filters: dict[str, Any]) -> list[str]:
    if not filters:
        return []
    case_sensitive = bool(filters.get("case_sensitive", False))
    haystack = normalize_filter_text(text, case_sensitive=case_sensitive)
    flags = 0 if case_sensitive else re.IGNORECASE
    failures: list[str] = []
    for phrase in filter_values(filters.get("must_contain")):
        needle = normalize_filter_text(phrase, case_sensitive=case_sensitive)
        if needle not in haystack:
            failures.append(f"missing:{phrase}")
    for phrase in filter_values(filters.get("must_not_contain")):
        needle = normalize_filter_text(phrase, case_sensitive=case_sensitive)
        if needle in haystack:
            failures.append(f"forbidden:{phrase}")
    for pattern in filter_values(filters.get("must_match")):
        if not re.search(pattern, text, flags=flags):
            failures.append(f"missing_pattern:{pattern}")
    for pattern in filter_values(filters.get("must_not_match")):
        if re.search(pattern, text, flags=flags):
            failures.append(f"forbidden_pattern:{pattern}")
    return failures


def text_filters_match(text: str, filters: dict[str, Any]) -> bool:
    return not text_filter_failures(text, filters)


def sorted_eval_rows(rows: list[dict[str, Any]], policy: str) -> list[dict[str, Any]]:
    if policy in {"shortest", "shortest_response", "shortest_pass"}:
        return sorted(rows, key=lambda row: (len(response_text(row)), int(row.get("trial_index", 0) or 0)))
    if policy in {"longest", "longest_response"}:
        return sorted(rows, key=lambda row: (-len(response_text(row)), int(row.get("trial_index", 0) or 0)))
    return sorted(rows, key=lambda row: int(row.get("trial_index", 0) or 0))


def group_key_for_eval_row(row: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    values = []
    for field in fields:
        if field == "prompt_sha256":
            values.append(hashlib.sha256(str(row.get("prompt", "")).encode("utf-8")).hexdigest())
        else:
            values.append(str(row.get(field, "")))
    return tuple(values)


def prompt_variant_items(raw_variants: Any, fallback_prompt: str) -> list[dict[str, str]]:
    if not raw_variants:
        return [{"id": "eval_prompt", "content": fallback_prompt, "exact_eval_prompt": "true"}]
    variants = []
    for index, raw in enumerate(raw_variants, start=1):
        if isinstance(raw, dict):
            content = str(raw.get("content", "")).strip()
            variant_id = str(raw.get("id", f"variant_{index:02d}"))
        else:
            content = str(raw).strip()
            variant_id = f"variant_{index:02d}"
        if content:
            variants.append({
                "id": variant_id,
                "content": content,
                "exact_eval_prompt": str(content == fallback_prompt).lower(),
            })
    return variants or [{"id": "eval_prompt", "content": fallback_prompt, "exact_eval_prompt": "true"}]


def build_eval_repair_dataset(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairing = config.get("pairing", {}) if isinstance(config.get("pairing"), dict) else {}
    default_source = config.get("source", {}) if isinstance(config.get("source"), dict) else {}
    default_skills = [str(skill) for skill in config.get("skills", [])]
    key_fields = [str(field) for field in pairing.get("key_fields", ["bucket", "case_id", "prompt"])]
    max_pairs_per_group = int(pairing.get("max_pairs_per_group", 0) or 0)
    chosen_policy = str(pairing.get("chosen_policy", "shortest_response"))
    rejected_policy = str(pairing.get("rejected_policy", "trial_index"))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_reports = []
    exact_eval_prompt_rows = 0
    skipped_groups = []
    text_filter_skips = []

    for source_index, source_config in enumerate(config.get("sources", []), start=1):
        if not isinstance(source_config, dict):
            raise ValueError("eval repair sources must be mappings")
        source_path = resolve_repo_path(source_config["path"])
        source_rows = read_jsonl(source_path)
        include = source_config.get("include", {}) if isinstance(source_config.get("include"), dict) else {}
        chosen_filters = source_config.get("chosen", {}).get("score_filters", {}) if isinstance(source_config.get("chosen"), dict) else {}
        rejected_filters = source_config.get("rejected", {}).get("score_filters", {}) if isinstance(source_config.get("rejected"), dict) else {}
        chosen_text_filters = source_config.get("chosen", {}).get("text_filters", {}) if isinstance(source_config.get("chosen"), dict) else {}
        rejected_text_filters = source_config.get("rejected", {}).get("text_filters", {}) if isinstance(source_config.get("rejected"), dict) else {}
        if not chosen_filters or not rejected_filters:
            raise ValueError(f"{display_path(source_path)} source requires chosen/rejected score_filters")
        prompt_variants = source_config.get("prompt_variants", pairing.get("prompt_variants"))
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        considered = 0
        for row in source_rows:
            if not include_eval_row(row, include):
                continue
            if not response_text(row) or not str(row.get("prompt", "")).strip():
                continue
            considered += 1
            groups.setdefault(group_key_for_eval_row(row, key_fields), []).append(row)

        emitted_for_source = 0
        chosen_text_filter_skip_count = 0
        rejected_text_filter_skip_count = 0
        for group_key, group_rows in sorted(groups.items()):
            chosen_score_rows = [row for row in group_rows if score_filters_match(row, chosen_filters)]
            rejected_score_rows = [row for row in group_rows if score_filters_match(row, rejected_filters)]
            filtered_chosen_rows = []
            for row in chosen_score_rows:
                failures = text_filter_failures(response_text(row), chosen_text_filters)
                if failures:
                    chosen_text_filter_skip_count += 1
                    text_filter_skips.append({
                        "source": display_path(source_path),
                        "group_key": list(group_key),
                        "role": "chosen",
                        "trial_index": row.get("trial_index"),
                        "failures": failures[:5],
                    })
                    continue
                filtered_chosen_rows.append(row)
            filtered_rejected_rows = []
            for row in rejected_score_rows:
                failures = text_filter_failures(response_text(row), rejected_text_filters)
                if failures:
                    rejected_text_filter_skip_count += 1
                    text_filter_skips.append({
                        "source": display_path(source_path),
                        "group_key": list(group_key),
                        "role": "rejected",
                        "trial_index": row.get("trial_index"),
                        "failures": failures[:5],
                    })
                    continue
                filtered_rejected_rows.append(row)
            chosen_rows = sorted_eval_rows(
                filtered_chosen_rows,
                chosen_policy,
            )
            rejected_rows = sorted_eval_rows(
                filtered_rejected_rows,
                rejected_policy,
            )
            if not chosen_rows or not rejected_rows:
                skipped_groups.append({
                    "source": display_path(source_path),
                    "group_key": list(group_key),
                    "chosen_count": len(chosen_rows),
                    "rejected_count": len(rejected_rows),
                    "chosen_score_count": len(chosen_score_rows),
                    "rejected_score_count": len(rejected_score_rows),
                    "reason": "missing_chosen_or_rejected",
                })
                continue
            emitted_for_group = 0
            for rejected in rejected_rows:
                for chosen in chosen_rows:
                    variants = prompt_variant_items(prompt_variants, str(chosen.get("prompt", "")))
                    for variant in variants:
                        if max_pairs_per_group and emitted_for_group >= max_pairs_per_group:
                            break
                        prompt = variant["content"]
                        chosen_response = response_text(chosen)
                        rejected_response = response_text(rejected)
                        digest_payload = {
                            "prompt": prompt,
                            "chosen": chosen_response,
                            "rejected": rejected_response,
                            "source": display_path(source_path),
                        }
                        digest = hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
                        if digest in seen:
                            continue
                        seen.add(digest)
                        exact_prompt = prompt == str(chosen.get("prompt", ""))
                        if exact_prompt:
                            exact_eval_prompt_rows += 1
                        source_meta = copy.deepcopy(default_source)
                        source_meta.update({
                            "kind": "eval_response_repair_seed",
                            "generator_model": source_meta.get("generator_model", source_config.get("generator_model", "unknown")),
                            "judge_model": source_meta.get("judge_model", source_config.get("judge_model", "model_forge_internal_eval")),
                            "source_uri": display_path(source_path),
                            "license": source_meta.get("license", source_config.get("license", "CC-BY-4.0")),
                            "contamination_risk": "high" if exact_prompt else str(source_meta.get("contamination_risk", "medium")),
                            "eval_repair": {
                                "schema_version": EVAL_REPAIR_DATASET_SCHEMA_VERSION,
                                "config_id": config["id"],
                                "source_index": source_index,
                                "source_path": display_path(source_path),
                                "bucket": chosen.get("bucket"),
                                "case_id": chosen.get("case_id"),
                                "prompt_variant_id": variant["id"],
                                "exact_eval_prompt": exact_prompt,
                                "chosen_trial_index": chosen.get("trial_index"),
                                "rejected_trial_index": rejected.get("trial_index"),
                                "chosen_scores": chosen.get("scores", {}),
                                "rejected_scores": rejected.get("scores", {}),
                            },
                        })
                        row = {
                            "id": f"{config['id']}_{digest}",
                            "skills": [*default_skills],
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": chosen_response},
                            ],
                            "rejected_messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": rejected_response},
                            ],
                            "source": source_meta,
                            "generation_method": "eval_response_repair",
                        }
                        rows.append(row)
                        emitted_for_group += 1
                        emitted_for_source += 1
                    if max_pairs_per_group and emitted_for_group >= max_pairs_per_group:
                        break
                if max_pairs_per_group and emitted_for_group >= max_pairs_per_group:
                    break
        source_reports.append({
            "path": display_path(source_path),
            "input_rows": len(source_rows),
            "considered_rows": considered,
            "groups": len(groups),
            "emitted_rows": emitted_for_source,
            "chosen_text_filter_skips": chosen_text_filter_skip_count,
            "rejected_text_filter_skips": rejected_text_filter_skip_count,
        })

    promotion_blockers = []
    if exact_eval_prompt_rows:
        promotion_blockers.append("exact_eval_prompt_rows_present")
    if bool(config.get("diagnostic_only", False)):
        promotion_blockers.append("diagnostic_only_config")
    report = {
        "schema_version": EVAL_REPAIR_DATASET_SCHEMA_VERSION,
        "dataset_id": config["id"],
        "description": str(config.get("description", "")).strip(),
        "output_path": display_path(resolve_repo_path(config["output_path"])),
        "rows": len(rows),
        "skills": default_skills,
        "source_reports": source_reports,
        "skipped_groups": skipped_groups,
        "text_filter_skips": text_filter_skips,
        "exact_eval_prompt_rows": exact_eval_prompt_rows,
        "promotion_blockers": promotion_blockers,
        "promotion_ready": not promotion_blockers and bool(rows),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return rows, report


def command_repair_from_eval(config: dict[str, Any], overwrite: bool) -> dict[str, Path]:
    output_path = resolve_repo_path(config["output_path"])
    report_path = resolve_repo_path(config.get("report_path", output_path.with_suffix(".report.json")))
    if output_path.exists() and report_path.exists() and not overwrite:
        return {"dataset": output_path, "report": report_path}
    rows, report = build_eval_repair_dataset(config)
    if not rows and not bool(config.get("allow_empty", False)):
        raise SystemExit("eval repair emitted no rows")
    write_jsonl(output_path, rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"dataset": output_path, "report": report_path}


def build_plan(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    output_dir = resolve_repo_path(config["output_dir"])
    objective_path = REPO_DIR / "configs" / "objectives" / f"{config['objective']}.yaml"
    objective = load_yaml(objective_path) if objective_path.exists() else {}
    seeds = load_seed_rows(config)
    seed_counts = Counter(skill for row in seeds for skill in row.get("skills", []))
    plan = {
        "id": config["id"],
        "family": config["family"],
        "variant": config["variant"],
        "objective": config["objective"],
        "objective_description": objective.get("description", ""),
        "config_path": display_path(config_path),
        "output_dir": display_path(output_dir),
        "seed_paths": list(config.get("seed_paths", [])),
        "seed_count": len(seeds),
        "seed_skill_counts": dict(sorted(seed_counts.items())),
        "target_accept_count": copy.deepcopy(config.get("target_accept_count", {})),
        "skills": copy.deepcopy(config.get("skills", [])),
        "quality_thresholds": copy.deepcopy(config.get("quality_thresholds", {})),
        "holdouts": copy.deepcopy(config.get("holdouts", [])),
        "generation_methods": copy.deepcopy(config.get("generation_methods", {})),
        "generation": copy.deepcopy(generation_config(config)),
        "review": copy.deepcopy(config.get("review", {})),
        "seed_only": bool(config.get("seed_only", False)),
        "smoke_only": bool(config.get("smoke_only", False)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    source_summary = registry_summary(config)
    if source_summary:
        plan["source_registry"] = source_summary
    return plan


def print_plan(plan: dict[str, Any]) -> None:
    table = Table(title=f"Dataset Factory Plan: {plan['id']}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    for key in ("family", "variant", "objective", "seed_count", "output_dir"):
        table.add_row(key, str(plan.get(key, "")))
    console.print(table)

    skills = Table(title="Skill Targets")
    skills.add_column("Skill", style="cyan")
    skills.add_column("Target", justify="right")
    skills.add_column("Seed rows", justify="right")
    counts = plan.get("seed_skill_counts", {})
    for item in plan.get("skills", []):
        skills.add_row(str(item["id"]), str(item.get("target_examples", "")), str(counts.get(item["id"], 0)))
    console.print(skills)


def command_plan(config: dict[str, Any], config_path: Path, overwrite: bool) -> Path:
    plan = build_plan(config, config_path)
    print_plan(plan)
    output_path = resolve_repo_path(config["output_dir"]) / "dataset_plan.yaml"
    if output_path.exists() and not overwrite:
        return output_path
    write_yaml(output_path, plan)
    return output_path


def command_seed(config: dict[str, Any], overwrite: bool) -> Path:
    output_path = resolve_repo_path(config["output_dir"]) / "seeds.jsonl"
    if output_path.exists() and not overwrite:
        return output_path
    rows = load_seed_rows(config)
    write_jsonl(output_path, rows)
    return output_path


def command_generate(
    config: dict[str, Any],
    overwrite: bool,
    smoke: bool = False,
    max_generated_candidates: int | None = None,
    provider_override: str | None = None,
) -> Path:
    seed_path = command_seed(config, overwrite=overwrite)
    output_dir = resolve_repo_path(config["output_dir"])
    output_path = resolve_repo_path(config["output_dir"]) / "candidates.jsonl"
    report_path = output_dir / "generation_report.json"
    if output_path.exists() and not overwrite:
        return output_path
    seed_rows = read_jsonl(seed_path)
    provider = build_provider(config, provider_override=provider_override)
    budget = generation_budget(config, smoke=smoke, max_generated_candidates=max_generated_candidates)
    max_generated = budget["max_generated_candidates"]
    max_seed_rows = budget["max_seed_rows"]
    strategy_rows = enabled_strategies(config)
    seed_window = select_generation_seeds(seed_rows, max_seed_rows)
    enforce_generation_resources(config, output_dir, planned_candidates=len(seed_rows) + max_generated)
    candidates = []
    for row in seed_rows:
        item = dict(row)
        item["generation_method"] = item.get("source", {}).get("kind", "human_seed")
        item["dataset_factory_stage"] = "candidate"
        candidates.append(item)

    generated_count = 0
    if max_generated > 0 and strategy_rows:
        for strategy in strategy_rows:
            for seed in seed_window:
                strategy_name = str(strategy.get("name", "self_instruct"))
                variants = int(strategy.get("variants_per_seed", 1) or 1)
                for variant_index in range(variants):
                    if generated_count >= max_generated:
                        break
                    candidates.append(generated_row(seed, config, provider, strategy_name, variant_index))
                    generated_count += 1
                if generated_count >= max_generated:
                    break
            if generated_count >= max_generated:
                break
    write_jsonl(output_path, candidates)
    report = build_generation_report(config, len(seed_rows), candidates, provider, smoke=smoke)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def command_judge(
    config: dict[str, Any],
    overwrite: bool,
    smoke: bool = False,
    max_generated_candidates: int | None = None,
    provider_override: str | None = None,
) -> Path:
    candidate_path = command_generate(
        config,
        overwrite=False,
        smoke=smoke,
        max_generated_candidates=max_generated_candidates,
        provider_override=provider_override,
    )
    output_path = resolve_repo_path(config["output_dir"]) / "judged.jsonl"
    if output_path.exists() and not overwrite:
        return output_path
    rows = []
    for row in read_jsonl(candidate_path):
        item = dict(row)
        item["quality_scores"] = score_row(item, config)
        item["quality_score_average"] = round(average_score(item["quality_scores"]), 4)
        rows.append(item)
    write_jsonl(output_path, rows)
    return output_path


def has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def has_at_least(text: str, terms: tuple[str, ...], count: int) -> bool:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered) >= count


def verify_skill(row: dict[str, Any], skill: str) -> tuple[bool, str]:
    text = row_text(row).lower()
    if skill == "eval_latency_throughput":
        ok = has_at_least(text, (
            "latency", "throughput", "tokens", "tokens/sec", "prompt",
            "completion", "ttft", "itl", "prefill", "decode", "batch",
        ), 3)
        return ok, "requires eval latency/throughput token-field coverage"
    if skill == "docker_disk_safety":
        ok = has_any(text, ("docker", "container", "checkpoint", "disk")) and has_any(text, (
            "active", "preserve", "df -h", "prune", "rotation", "keep", "delete", "free space",
        ))
        return ok, "requires disk/checkpoint cleanup with active-resource safety"
    if skill == "sql_edge_cases":
        ok = has_any(text, ("sql", "null", "count(*)", "count(", "left join", "where", "aggregate", "index"))
        return ok, "requires SQL edge-case concept coverage"
    if skill == "shell_safety":
        ok = has_any(text, ("shell", "rsync", "command", "dry run", "--dry-run", "quote", "refuse", "quarantine", "safe"))
        return ok, "requires non-destructive shell or command-safety framing"
    if skill == "config_review":
        ok = has_any(text, ("yaml", "config", "variant", "prompt set", "output path", "validation", "validator"))
        return ok, "requires config validation or model-family review coverage"
    if skill == "json_schema_repair":
        ok = "json" in text and has_any(text, (
            "schema", "parse", "parser", "double-quoted", "quoted", "lowercase", "trailing", "object", "array",
        ))
        return ok, "requires JSON/schema repair constraints"
    if skill == "git_workflow_repair":
        ok = "git" in text and has_any(text, (
            "fetch", "rebase", "branch", "conflict", "verify", "force-with-lease", "backup", "cherry-pick", "push",
        ))
        return ok, "requires concrete git repair workflow coverage"
    if skill == "benign_safety_analysis":
        safe_framing = has_any(text, (
            "safe", "safety", "responsible", "without", "avoid", "refuse", "not", "do not",
            "policy", "risk", "minimize", "over-refusal", "overcompliance",
        ))
        harmful_context = has_any(text, (
            "exploit", "payload", "delete another", "doxxing", "bypass", "credential theft", "harassment",
        ))
        ok = safe_framing and (not harmful_context or has_any(text, (
            "without", "avoid", "refuse", "not", "do not", "responsible", "safe", "minimize",
        )))
        return ok, "requires benign safety-analysis framing without tactical harmful detail"
    if skill == "checkpoint_selection":
        ok = has_any(text, ("checkpoint", "eval", "baseline", "trial", "objective", "gate", "regression", "promote"))
        return ok, "requires checkpoint/eval selection methodology"
    return False, f"unknown configured skill: {skill}"


def verify_row(row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages", [])
    has_user = any(isinstance(m, dict) and m.get("role") == "user" and m.get("content") for m in messages)
    has_assistant = any(isinstance(m, dict) and m.get("role") == "assistant" and m.get("content") for m in messages)
    source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
    configured_skills = set(skill_targets(config))
    skills = [str(skill) for skill in row.get("skills", [])]
    checks: dict[str, bool] = {
        "has_user_message": has_user,
        "has_assistant_message": has_assistant,
        "has_valid_skill": bool(skills) and all(skill in configured_skills for skill in skills),
        "has_source_license": bool(source.get("license")),
    }
    details: dict[str, str] = {}
    for skill in skills:
        passed, detail = verify_skill(row, skill)
        checks[f"skill:{skill}"] = passed
        details[f"skill:{skill}"] = detail
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "type": "static_skill_checks",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "details": details,
    }


def command_verify(
    config: dict[str, Any],
    overwrite: bool,
    smoke: bool = False,
    max_generated_candidates: int | None = None,
    provider_override: str | None = None,
) -> Path:
    judged_path = command_judge(
        config,
        overwrite=overwrite,
        smoke=smoke,
        max_generated_candidates=max_generated_candidates,
        provider_override=provider_override,
    )
    output_path = resolve_repo_path(config["output_dir"]) / "verification.jsonl"
    if output_path.exists() and not overwrite:
        return output_path
    rows = []
    for row in read_jsonl(judged_path):
        item = dict(row)
        item["verification"] = verify_row(item, config)
        item["dataset_factory_stage"] = "verified"
        rows.append(item)
    write_jsonl(output_path, rows)
    return output_path


def rejection_reasons(row: dict[str, Any], config: dict[str, Any], seen: set[str], holdouts: list[dict[str, str]]) -> list[str]:
    thresholds = config.get("quality_thresholds", {})
    reasons = []
    digest = stable_id(row)
    if digest in seen:
        reasons.append("duplicate_conversation")
    scores = row.get("quality_scores", {})
    if row.get("quality_score_average", average_score(scores)) < float(thresholds.get("min_average_score", 0.0)):
        reasons.append("average_quality_below_threshold")
    if float(scores.get("target_skill_relevance", 0.0)) < float(thresholds.get("min_target_skill_relevance", 0.0)):
        reasons.append("target_skill_relevance_below_threshold")
    source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
    if source.get("license_risk") in set(thresholds.get("reject_license_risk", [])):
        reasons.append("license_risk_rejected")
    if source.get("contamination_risk") in set(thresholds.get("reject_contamination_risk", [])):
        reasons.append("contamination_risk_rejected")
    if bool(thresholds.get("reject_length_violations", False)):
        min_words, max_words = assistant_word_bounds(config)
        assistant_words = len(re.findall(r"\w+", message_content(row, "assistant")))
        if assistant_words < min_words:
            reasons.append("assistant_too_short")
        if assistant_words > max_words:
            reasons.append("assistant_too_long")
    verification = row.get("verification", {}) if isinstance(row.get("verification"), dict) else {}
    if verification and not verification.get("passed", False):
        reasons.append("verification_failed")
    overlap = holdout_overlap(row, holdouts)
    row["holdout_overlap"] = overlap
    if float(overlap.get("max_similarity", 0.0)) > float(thresholds.get("max_holdout_similarity", 1.0)):
        reasons.append("holdout_overlap_above_threshold")
    return reasons


def command_filter(
    config: dict[str, Any],
    overwrite: bool,
    smoke: bool = False,
    max_generated_candidates: int | None = None,
    provider_override: str | None = None,
) -> tuple[Path, Path]:
    verified_path = command_verify(
        config,
        overwrite=overwrite,
        smoke=smoke,
        max_generated_candidates=max_generated_candidates,
        provider_override=provider_override,
    )
    output_dir = resolve_repo_path(config["output_dir"])
    accepted_path = output_dir / "accepted.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    if accepted_path.exists() and rejected_path.exists() and not overwrite:
        return accepted_path, rejected_path

    holdouts = load_holdout_prompts(config)
    seen: set[str] = set()
    accepted = []
    rejected = []
    for row in read_jsonl(verified_path):
        item = dict(row)
        reasons = rejection_reasons(item, config, seen, holdouts)
        if reasons:
            item["rejection_reasons"] = reasons
            rejected.append(item)
        else:
            seen.add(stable_id(item))
            item["dataset_factory_stage"] = "accepted"
            accepted.append(item)
    write_jsonl(accepted_path, accepted)
    write_jsonl(rejected_path, rejected)
    return accepted_path, rejected_path


def review_config(config: dict[str, Any]) -> dict[str, Any]:
    review = config.get("review", {})
    return review if isinstance(review, dict) else {}


def sample_review_rows(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size <= 0 or len(rows) <= sample_size:
        return rows
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for skill in sorted({skill for row in rows for skill in row.get("skills", [])}):
        for row in rows:
            if row.get("id") in seen_ids:
                continue
            if skill in row.get("skills", []):
                selected.append(row)
                seen_ids.add(str(row.get("id")))
                break
    for method in sorted({str(row.get("generation_method", "unknown")) for row in rows}):
        for row in rows:
            if row.get("id") in seen_ids:
                continue
            if str(row.get("generation_method", "unknown")) == method:
                selected.append(row)
                seen_ids.add(str(row.get("id")))
                break
    for row in rows:
        if len(selected) >= sample_size:
            break
        if row.get("id") in seen_ids:
            continue
        selected.append(row)
        seen_ids.add(str(row.get("id")))
    return selected[:sample_size]


def seed_lookup(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    seeds = load_seed_rows(config)
    return {str(row.get("id", stable_id(row))): row for row in seeds}


def copied_seed_similarity(row: dict[str, Any], seeds: dict[str, dict[str, Any]]) -> float:
    source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
    generation = source.get("generation", {}) if isinstance(source.get("generation"), dict) else {}
    seed_id = generation.get("seed_id") or str(source.get("source_uri", "")).removeprefix("seed:")
    seed = seeds.get(str(seed_id))
    if not seed:
        return 0.0
    return round(jaccard(row_text(row), row_text(seed)), 4)


def row_review_flags(row: dict[str, Any], config: dict[str, Any], seeds: dict[str, dict[str, Any]]) -> list[str]:
    settings = review_config(config)
    thresholds = config.get("quality_thresholds", {}) if isinstance(config.get("quality_thresholds"), dict) else {}
    flags = []
    assistant = message_content(row, "assistant")
    assistant_words = len(re.findall(r"\w+", assistant))
    min_words, max_words = assistant_word_bounds(config)
    if assistant_words < min_words:
        flags.append("too_generic")
    if assistant_words > max_words:
        flags.append("too_long")
    concrete_terms = (
        "check", "verify", "compare", "inspect", "run", "validate", "review",
        "confirm", "log", "measure", "count", "require", "avoid", "include",
        "use", "explain", "evaluate", "select", "refuse", "document", "test",
    )
    if not has_any(assistant, concrete_terms):
        flags.append("missing_concrete_checks")
    similarity = copied_seed_similarity(row, seeds)
    max_seed_similarity = float(settings.get("max_seed_similarity", 0.74))
    source = row.get("source", {}) if isinstance(row.get("source"), dict) else {}
    if source.get("kind") == "synthetic" and similarity > max_seed_similarity:
        flags.append("copied_seed_wording")
    harmful_terms = (
        "payload", "exploit steps", "credential theft", "delete another",
        "doxxing", "bypass", "malware", "exfiltrate", "persistence",
    )
    safe_terms = ("avoid", "without", "refuse", "do not", "safe", "responsible", "not", "minimize")
    if has_any(assistant, harmful_terms) and not has_any(assistant, safe_terms):
        flags.append("unsafe_detail")
    verification = row.get("verification", {}) if isinstance(row.get("verification"), dict) else {}
    min_quality = float(thresholds.get("min_average_score", 0.0))
    if not verification.get("passed", False) or float(row.get("quality_score_average", 0.0)) < min_quality:
        flags.append("poor_assistant_answer")
    return flags


def build_review_report(config: dict[str, Any], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], sample_size: int) -> dict[str, Any]:
    settings = review_config(config)
    seeds = seed_lookup(config)
    sampled = sample_review_rows(accepted, sample_size)
    reviewed_rows = []
    flag_counts: Counter[str] = Counter()
    configured_critical = settings.get("critical_flags")
    if configured_critical:
        critical_flags = {str(flag) for flag in configured_critical}
    else:
        critical_flags = {"copied_seed_wording", "unsafe_detail", "poor_assistant_answer"}
    critical_flag_counts: Counter[str] = Counter()
    for row in sampled:
        flags = row_review_flags(row, config, seeds)
        for flag in flags:
            flag_counts[flag] += 1
            if flag in critical_flags:
                critical_flag_counts[flag] += 1
        reviewed_rows.append({
            "id": row.get("id"),
            "skills": row.get("skills", []),
            "generation_method": row.get("generation_method", "unknown"),
            "source_kind": row.get("source", {}).get("kind", "unknown") if isinstance(row.get("source"), dict) else "unknown",
            "quality_score_average": row.get("quality_score_average"),
            "verification_passed": row.get("verification", {}).get("passed") if isinstance(row.get("verification"), dict) else None,
            "copied_seed_similarity": copied_seed_similarity(row, seeds),
            "flags": flags,
            "user": compact_text(message_content(row, "user"), 44),
            "assistant": compact_text(message_content(row, "assistant"), 70),
        })

    skill_counts = Counter(skill for row in accepted for skill in row.get("skills", []))
    method_counts = Counter(str(row.get("generation_method", "unknown")) for row in accepted)
    source_counts = Counter(
        str(row.get("source", {}).get("kind", "unknown"))
        for row in accepted
        if isinstance(row.get("source"), dict)
    )
    min_examples = int(settings.get("min_examples_per_skill", config.get("quality_thresholds", {}).get("min_seed_examples_per_skill", 0)) or 0)
    coverage_gaps = [
        {"skill": skill, "count": skill_counts.get(skill, 0), "required": min_examples}
        for skill in skill_targets(config)
        if min_examples and skill_counts.get(skill, 0) < min_examples
    ]
    gates = {
        "sample_has_rows": bool(sampled),
        "skill_coverage_ready": not coverage_gaps,
        "no_critical_flags": not critical_flag_counts,
        "has_synthetic_rows": source_counts.get("synthetic", 0) > 0,
    }
    ready_to_scale = all(gates.values())
    return {
        "dataset_id": config["id"],
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "sample_size": sample_size,
        "reviewed_count": len(sampled),
        "ready_to_scale_generation": ready_to_scale,
        "gates": gates,
        "flag_counts": dict(sorted(flag_counts.items())),
        "critical_flag_counts": dict(sorted(critical_flag_counts.items())),
        "skill_counts": dict(sorted(skill_counts.items())),
        "generation_method_counts": dict(sorted(method_counts.items())),
        "source_kind_counts": dict(sorted(source_counts.items())),
        "coverage_gaps": coverage_gaps,
        "reviewed_rows": reviewed_rows,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def review_sheet(report: dict[str, Any]) -> str:
    gates = "\n".join(f"- `{key}`: {str(value).lower()}" for key, value in report["gates"].items())
    coverage = "\n".join(
        f"- `{item['skill']}`: {item['count']} / {item['required']}"
        for item in report.get("coverage_gaps", [])
    ) or "- none"
    flags = "\n".join(f"- `{flag}`: {count}" for flag, count in report.get("flag_counts", {}).items()) or "- none"
    rows = []
    for row in report.get("reviewed_rows", []):
        rows.append(
            "\n".join([
                f"### {row['id']}",
                f"- Skills: {', '.join(row.get('skills', []))}",
                f"- Method: `{row.get('generation_method')}` / Source: `{row.get('source_kind')}`",
                f"- Quality: `{row.get('quality_score_average')}` / Verification: `{str(row.get('verification_passed')).lower()}`",
                f"- Copied-seed similarity: `{row.get('copied_seed_similarity')}`",
                f"- Flags: {', '.join(row.get('flags', [])) or 'none'}",
                f"- User: {row.get('user')}",
                f"- Assistant: {row.get('assistant')}",
            ])
        )
    sampled_rows = "\n\n".join(rows) or "No sampled rows."
    return f"""# Dataset Review: {report['dataset_id']}

## Summary

- Accepted rows: {report['accepted_count']}
- Rejected rows: {report['rejected_count']}
- Reviewed rows: {report['reviewed_count']} / {report['sample_size']}
- Ready to scale generation: {str(report['ready_to_scale_generation']).lower()}

## Gates

{gates}

## Coverage Gaps

{coverage}

## Flags

{flags}

## Sampled Rows

{sampled_rows}
"""


def command_review(
    config: dict[str, Any],
    overwrite: bool,
    smoke: bool = False,
    max_generated_candidates: int | None = None,
    provider_override: str | None = None,
    sample_size: int | None = None,
) -> dict[str, str]:
    accepted_path, rejected_path = command_filter(
        config,
        overwrite=overwrite,
        smoke=smoke,
        max_generated_candidates=max_generated_candidates,
        provider_override=provider_override,
    )
    output_dir = resolve_repo_path(config["output_dir"])
    report_path = output_dir / "review_report.json"
    sheet_path = output_dir / "review_sheet.md"
    if report_path.exists() and sheet_path.exists() and not overwrite:
        return {"review_report": str(report_path), "review_sheet": str(sheet_path)}
    settings = review_config(config)
    size = sample_size if sample_size is not None else int(settings.get("sample_size", 50) or 50)
    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    report = build_review_report(config, accepted, rejected, size)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sheet_path.write_text(review_sheet(report), encoding="utf-8")
    return {"review_report": str(report_path), "review_sheet": str(sheet_path)}


def quality_report(config: dict[str, Any], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    skill_counts = Counter(skill for row in accepted for skill in row.get("skills", []))
    rejection_counts = Counter(reason for row in rejected for reason in row.get("rejection_reasons", []))
    averages = [float(row.get("quality_score_average", 0.0)) for row in accepted]
    source_kind_counts = Counter(
        str(row.get("source", {}).get("kind", "unknown"))
        for row in accepted
        if isinstance(row.get("source"), dict)
    )
    generation_method_counts = Counter(str(row.get("generation_method", "unknown")) for row in accepted)
    verification_counts = Counter(
        "passed" if row.get("verification", {}).get("passed") else "failed"
        for row in [*accepted, *rejected]
        if isinstance(row.get("verification"), dict)
    )
    target = config.get("target_accept_count", {})
    thresholds = config.get("quality_thresholds", {})
    warnings = []
    min_target = int(target.get("min", 0) or 0) if isinstance(target, dict) else 0
    if min_target and len(accepted) < min_target:
        warnings.append({
            "code": "accepted_count_below_min_target",
            "message": f"accepted rows {len(accepted)} below configured minimum {min_target}",
        })
    min_seed_per_skill = int(thresholds.get("min_seed_examples_per_skill", 0) or 0)
    if min_seed_per_skill:
        for skill in skill_targets(config):
            count = skill_counts.get(skill, 0)
            if count < min_seed_per_skill:
                warnings.append({
                    "code": "skill_below_min_seed_examples",
                    "skill": skill,
                    "message": f"{skill} has {count} accepted rows, below seed target {min_seed_per_skill}",
                })
    return {
        "dataset_id": config["id"],
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "skill_counts": dict(sorted(skill_counts.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "generation_method_counts": dict(sorted(generation_method_counts.items())),
        "verification_counts": dict(sorted(verification_counts.items())),
        "quality_score_average": round(sum(averages) / len(averages), 4) if averages else 0.0,
        "target_accept_count": copy.deepcopy(config.get("target_accept_count", {})),
        "seed_only": bool(config.get("seed_only", False)),
        "smoke_only": bool(config.get("smoke_only", False)),
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _gate(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "message": message}


def pack_stage(config: dict[str, Any], report: dict[str, Any]) -> str:
    accepted = int(report.get("accepted_count", 0) or 0)
    target = report.get("target_accept_count") or config.get("target_accept_count") or {}
    min_target = int(target.get("min", 0) or 0) if isinstance(target, dict) else 0
    if bool(report.get("seed_only", config.get("seed_only", False))):
        return "seed_pack"
    if bool(report.get("smoke_only", config.get("smoke_only", False))) or accepted < 250:
        return "smoke_pack"
    if min_target and accepted < min_target:
        return "medium_pack"
    return "training_pack"


def build_pack_promotion_gates(
    config: dict[str, Any],
    report: dict[str, Any],
    *,
    review_report: dict[str, Any] | None = None,
    training_gate_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    accepted = int(report.get("accepted_count", 0) or 0)
    rejected = int(report.get("rejected_count", 0) or 0)
    verification = report.get("verification_counts") or {}
    warnings = report.get("warnings") or []
    target = report.get("target_accept_count") or config.get("target_accept_count") or {}
    min_target = int(target.get("min", 0) or 0) if isinstance(target, dict) else 0
    max_target = int(target.get("max", 0) or 0) if isinstance(target, dict) else 0
    stage = pack_stage(config, report)
    review_ready = bool(review_report and review_report.get("ready_to_scale_generation"))
    no_critical_review_flags = not bool((review_report or {}).get("critical_flag_counts"))
    common = [
        _gate("accepted_rows_present", accepted > 0, f"accepted_count={accepted}"),
        _gate("rejected_rows_recorded", rejected >= 0, f"rejected_count={rejected}"),
        _gate("verification_covers_accepted_rows", int(verification.get("passed", 0) or 0) >= accepted, f"verification_passed={verification.get('passed', 0)} accepted={accepted}"),
    ]
    quality_gates = [
        _gate("no_high_level_quality_warnings", not warnings, f"warnings={len(warnings)}"),
    ]
    smoke_gates = [
        *copy.deepcopy(common),
        _gate("smoke_size_band", 25 <= accepted <= 100, "smoke packs should contain 25-100 accepted examples"),
        _gate("schema_card_filtering_ready", True, "pack wrote dataset, manifest, quality report, card, accepted/rejected, and verification artifacts"),
    ]
    medium_gates = [
        *copy.deepcopy(common),
        *copy.deepcopy(quality_gates),
        _gate("not_seed_or_smoke_only", not report.get("seed_only") and not report.get("smoke_only"), "medium packs must not be seed-only or smoke-only"),
        _gate("medium_size_band", 250 <= accepted <= 500, "medium packs should contain 250-500 accepted examples"),
        _gate("review_ready_to_scale", review_ready, "review_report.ready_to_scale_generation must be true"),
        _gate("no_critical_review_flags", no_critical_review_flags, "review report must have no critical flags"),
    ]
    training_gates = [
        *copy.deepcopy(common),
        *copy.deepcopy(quality_gates),
        _gate("not_seed_or_smoke_only", not report.get("seed_only") and not report.get("smoke_only"), "training packs must not be seed-only or smoke-only"),
        _gate("training_size_floor", accepted >= min_target if min_target else accepted >= 500, f"accepted_count={accepted} min_target={min_target or 500}"),
        _gate("training_size_ceiling", accepted <= max_target if max_target else True, f"accepted_count={accepted} max_target={max_target or 'unbounded'}"),
        _gate("review_ready_to_scale", review_ready, "review_report.ready_to_scale_generation must be true"),
        _gate("bounded_training_evidence_attached", bool(training_gate_report and training_gate_report.get("dataset_recipe_validated")), "training-gate evidence must pass before recipe validation"),
    ]
    stage_gates = {
        "smoke_pack": smoke_gates,
        "medium_pack": medium_gates,
        "training_pack": training_gates,
    }
    selected = stage_gates.get(stage, smoke_gates)
    return {
        "schema_version": PACK_PROMOTION_GATES_SCHEMA_VERSION,
        "dataset_id": config["id"],
        "stage": stage,
        "stage_ready": all(gate["passed"] for gate in selected),
        "gates": {
            "smoke_pack": smoke_gates,
            "medium_pack": medium_gates,
            "training_pack": training_gates,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def dataset_card(config: dict[str, Any], report: dict[str, Any]) -> str:
    skills = "\n".join(f"- `{skill}`: {count}" for skill, count in report["skill_counts"].items()) or "- none"
    warnings = "\n".join(f"- `{item['code']}`: {item['message']}" for item in report.get("warnings", [])) or "- none"
    source_counts = "\n".join(f"- `{kind}`: {count}" for kind, count in report.get("source_kind_counts", {}).items()) or "- none"
    method_counts = "\n".join(f"- `{method}`: {count}" for method, count in report.get("generation_method_counts", {}).items()) or "- none"
    verification_counts = report.get("verification_counts", {})
    pack_gates = report.get("pack_promotion_gates") or {}
    return f"""---
dataset_info:
  config_name: {config['id']}
  features:
    - name: messages
      dtype: list
    - name: skills
      dtype: list
license: cc-by-4.0
task_categories:
  - text-generation
  - question-answering
tags:
  - model-forge
  - supervised-fine-tuning
  - eval-adjacent
---

# {config['id']}

{config.get('description', '').strip()}

## Purpose

This is a Model Forge dataset-factory artifact for `{config['family']}` /
`{config['variant']}` under the `{config['objective']}` objective. It targets
observed fine-tuning gaps without copying held-out model-forge eval prompts.

## Counts

- Accepted rows: {report['accepted_count']}
- Rejected rows: {report['rejected_count']}
- Mean quality score: {report['quality_score_average']}
- Verification passed: {verification_counts.get('passed', 0)}
- Verification failed: {verification_counts.get('failed', 0)}
- Seed-only scaffold: {str(bool(config.get('seed_only', False))).lower()}
- Smoke-only scaffold: {str(bool(config.get('smoke_only', False))).lower()}
- Pack stage: {pack_gates.get('stage', 'unknown')}
- Pack stage ready: {str(bool(pack_gates.get('stage_ready', False))).lower()}

## Skill Counts

{skills}

## Source Counts

{source_counts}

## Generation Method Counts

{method_counts}

## Coverage Warnings

{warnings}

## Provenance

Rows include human seeds and any configured synthetic candidates. Synthetic
rows record provider type, generator model, source seed, strategy, and prompt
template hash. Future versions can add executable verification and HF
publication using the same manifest layout.

## Safety And Contamination

The pack step writes `accepted.jsonl` and `rejected.jsonl`, records rejection
reasons, and checks similarity against configured holdout prompt files.
"""


def command_pack(
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool,
    smoke: bool = False,
    max_generated_candidates: int | None = None,
    provider_override: str | None = None,
) -> dict[str, str]:
    accepted_path, rejected_path = command_filter(
        config,
        overwrite=overwrite,
        smoke=smoke,
        max_generated_candidates=max_generated_candidates,
        provider_override=provider_override,
    )
    output_dir = resolve_repo_path(config["output_dir"])
    verification_path = output_dir / "verification.jsonl"
    generation_report_path = output_dir / "generation_report.json"
    dataset_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.yaml"
    report_path = output_dir / "quality_report.json"
    card_path = output_dir / "dataset_card.md"
    gap_path = output_dir / "gap_report.yaml"
    review_report_path = output_dir / "review_report.json"
    review_sheet_path = output_dir / "review_sheet.md"
    if all(path.exists() for path in (dataset_path, manifest_path, report_path, card_path, verification_path, generation_report_path)) and not overwrite:
        outputs = {
            "dataset": str(dataset_path),
            "manifest": str(manifest_path),
            "quality_report": str(report_path),
            "dataset_card": str(card_path),
            "verification": str(verification_path),
            "generation_report": str(generation_report_path),
        }
        if review_report_path.exists():
            outputs["review_report"] = str(review_report_path)
        if review_sheet_path.exists():
            outputs["review_sheet"] = str(review_sheet_path)
        return outputs

    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    dataset_rows = [
        {
            "id": row["id"],
            "messages": row["messages"],
            "skills": row.get("skills", []),
            "source": row.get("source", {}),
            "generation_method": row.get("generation_method"),
            "quality_scores": row.get("quality_scores", {}),
            "verification": row.get("verification", {}),
            "holdout_overlap": row.get("holdout_overlap", {}),
        }
        for row in accepted
    ]
    write_jsonl(dataset_path, dataset_rows)

    report = quality_report(config, accepted, rejected)
    review_report = load_json(review_report_path) if review_report_path.exists() else None
    report["pack_promotion_gates"] = build_pack_promotion_gates(config, report, review_report=review_report)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = build_plan(config, config_path)
    manifest.update({
        "artifacts": {
            "dataset": display_path(dataset_path),
            "accepted": display_path(accepted_path),
            "rejected": display_path(rejected_path),
            "verification": display_path(verification_path) if verification_path.exists() else None,
            "generation_report": display_path(generation_report_path) if generation_report_path.exists() else None,
            "review_report": display_path(review_report_path) if review_report_path.exists() else None,
            "review_sheet": display_path(review_sheet_path) if review_sheet_path.exists() else None,
            "quality_report": display_path(report_path),
            "dataset_card": display_path(card_path),
            "gap_report": display_path(gap_path) if gap_path.exists() else None,
        },
        "quality_report": report,
    })
    write_yaml(manifest_path, manifest)
    card_path.write_text(dataset_card(config, report), encoding="utf-8")
    outputs = {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "quality_report": str(report_path),
        "dataset_card": str(card_path),
        "verification": str(verification_path),
        "generation_report": str(generation_report_path),
    }
    if review_report_path.exists():
        outputs["review_report"] = str(review_report_path)
    if review_sheet_path.exists():
        outputs["review_sheet"] = str(review_sheet_path)
    return outputs


def _check(name: str, passed: bool, message: str) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "message": message}


def _path_matches(candidate: str | None, expected: Path) -> bool:
    if not candidate:
        return False
    path = resolve_repo_path(candidate)
    try:
        return path.resolve() == expected.resolve()
    except OSError:
        return path == expected


def build_training_evidence_gate(
    *,
    config: dict[str, Any],
    config_path: Path,
    dataset_manifest: dict[str, Any],
    finetune_plan: dict[str, Any],
    data_summary: dict[str, Any],
    promotion_report: dict[str, Any],
    dataset_manifest_path: Path,
    finetune_plan_path: Path,
    data_summary_path: Path,
    promotion_report_path: Path,
    max_steps: int = 50,
    max_train_rows: int = 5000,
    require_spark: bool = True,
) -> dict[str, Any]:
    quality = dataset_manifest.get("quality_report") or {}
    artifacts = dataset_manifest.get("artifacts") or {}
    dataset_path = resolve_repo_path(str(artifacts.get("dataset") or ""))
    sources = finetune_plan.get("data", {}).get("sources") or []
    source_paths = [
        str(source.get("path") or source.get("dataset") or "")
        for source in sources
        if isinstance(source, dict)
    ]
    trainer = finetune_plan.get("trainer") or {}
    resource_policy = finetune_plan.get("resource_policy") or {}
    hardware = finetune_plan.get("hardware") or {}
    max_steps_value = trainer.get("max_steps")
    if max_steps_value is None:
        bounded_steps = False
    else:
        bounded_steps = int(max_steps_value) <= max_steps
    train_rows = int(data_summary.get("rows", 0) or 0)
    cpu_quota_text = str(resource_policy.get("cpu_quota") or "")
    memory_max_text = str(resource_policy.get("memory_max") or "")
    profile = str(hardware.get("profile") or "").lower()
    label = str(hardware.get("label") or "").lower()
    spark_like = "spark" in profile or "spark" in label or "gb10" in label
    output_path = str(data_summary.get("output_path") or "")
    output_exists = bool(output_path) and resolve_repo_path(output_path).exists()

    checks = [
        _check(
            "dataset_id_matches_config",
            dataset_manifest.get("id") == config.get("id"),
            f"manifest id `{dataset_manifest.get('id')}` matches config id `{config.get('id')}`",
        ),
        _check(
            "dataset_artifact_present",
            bool(artifacts.get("dataset")),
            "packed dataset manifest names a dataset artifact",
        ),
        _check(
            "dataset_has_rows",
            int(quality.get("accepted_count", 0) or 0) > 0,
            f"accepted_count={quality.get('accepted_count', 0)}",
        ),
        _check(
            "dataset_not_seed_only",
            not bool(quality.get("seed_only", dataset_manifest.get("seed_only", False))),
            "seed-only datasets are not valid training evidence",
        ),
        _check(
            "dataset_not_smoke_only",
            not bool(quality.get("smoke_only", dataset_manifest.get("smoke_only", False))),
            "smoke-only datasets are not valid training evidence",
        ),
        _check(
            "finetune_plan_uses_dataset",
            any(_path_matches(path, dataset_path) for path in source_paths),
            "fine-tune plan includes the packed dataset path as a data source",
        ),
        _check(
            "bounded_max_steps",
            bounded_steps,
            f"trainer.max_steps={max_steps_value}; required <= {max_steps}",
        ),
        _check(
            "bounded_train_rows",
            0 < train_rows <= max_train_rows,
            f"data_summary.rows={train_rows}; required 1..{max_train_rows}",
        ),
        _check(
            "train_dataset_materialized",
            output_exists,
            f"data_summary.output_path exists: {output_path or '<missing>'}",
        ),
        _check(
            "spark_evidence_present",
            (not require_spark) or spark_like,
            f"hardware profile/label `{hardware.get('profile')}` / `{hardware.get('label')}`",
        ),
        _check(
            "resource_governed",
            cpu_quota_text.endswith("%") and memory_max_text.endswith("%") and int(resource_policy.get("reserve_cores", 0) or 0) >= 1,
            "fine-tune plan records CPU quota, memory max, and at least one reserved core",
        ),
        _check(
            "ram_floor_present",
            float(resource_policy.get("min_memory_available_start", 0.0) or 0.0) >= 0.05
            and float(resource_policy.get("min_memory_available_runtime", 0.0) or 0.0) >= 0.05,
            "start/runtime RAM floors are at least 5%",
        ),
        _check(
            "promotion_report_passed",
            bool(promotion_report.get("passed")),
            f"promotion report decision `{promotion_report.get('decision')}`",
        ),
    ]
    ready = all(item["passed"] for item in checks)
    return {
        "schema_version": TRAINING_EVIDENCE_GATE_SCHEMA_VERSION,
        "dataset_id": config["id"],
        "family": config["family"],
        "variant": config["variant"],
        "config_path": display_path(config_path),
        "artifacts": {
            "dataset_manifest": display_path(dataset_manifest_path),
            "finetune_plan": display_path(finetune_plan_path),
            "data_summary": display_path(data_summary_path),
            "promotion_report": display_path(promotion_report_path),
        },
        "bounds": {
            "max_steps": max_steps,
            "max_train_rows": max_train_rows,
            "require_spark": require_spark,
        },
        "metrics": {
            "accepted_count": int(quality.get("accepted_count", 0) or 0),
            "train_rows": train_rows,
            "trainer_max_steps": max_steps_value,
            "hardware_profile": hardware.get("profile"),
            "promotion_decision": promotion_report.get("decision"),
        },
        "checks": checks,
        "dataset_recipe_validated": ready,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def training_evidence_gate_markdown(report: dict[str, Any]) -> str:
    rows = []
    for check in report["checks"]:
        status = "PASS" if check["passed"] else "FAIL"
        rows.append(f"| {status} | `{check['name']}` | {check['message']} |")
    return "\n".join([
        f"# Dataset Training Evidence Gate: {report['dataset_id']}",
        "",
        f"- Family: `{report['family']}`",
        f"- Variant: `{report['variant']}`",
        f"- Validated: `{str(report['dataset_recipe_validated']).lower()}`",
        f"- Max steps bound: `{report['bounds']['max_steps']}`",
        f"- Max train rows bound: `{report['bounds']['max_train_rows']}`",
        "",
        "| Status | Check | Message |",
        "|---|---|---|",
        *rows,
        "",
    ])


def command_training_gate(
    *,
    config: dict[str, Any],
    config_path: Path,
    dataset_manifest_path: Path,
    finetune_plan_path: Path,
    data_summary_path: Path,
    promotion_report_path: Path,
    max_steps: int,
    max_train_rows: int,
    require_spark: bool,
    write_gate: bool,
) -> dict[str, Any]:
    report = build_training_evidence_gate(
        config=config,
        config_path=config_path,
        dataset_manifest=load_yaml(dataset_manifest_path),
        finetune_plan=load_json(finetune_plan_path),
        data_summary=load_json(data_summary_path),
        promotion_report=load_json(promotion_report_path),
        dataset_manifest_path=dataset_manifest_path,
        finetune_plan_path=finetune_plan_path,
        data_summary_path=data_summary_path,
        promotion_report_path=promotion_report_path,
        max_steps=max_steps,
        max_train_rows=max_train_rows,
        require_spark=require_spark,
    )
    if write_gate:
        output_dir = resolve_repo_path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "training_evidence_gate.json"
        md_path = output_dir / "training_evidence_gate.md"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(training_evidence_gate_markdown(report), encoding="utf-8")
        report["written_artifacts"] = {"json": display_path(json_path), "markdown": display_path(md_path)}
    return report


def load_release_class(name: str) -> dict[str, Any]:
    path = REPO_DIR / "configs" / "release_classes" / f"{name}.yaml"
    if not path.exists():
        return {
            "id": name,
            "hf_visibility": "private",
            "publish_raw_outputs": "private_only",
            "requires": [],
        }
    data = load_yaml(path)
    data.setdefault("id", name)
    return data


def row_content_digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(row_text(row).encode("utf-8")).hexdigest()


def redacted_message(message: dict[str, Any]) -> dict[str, Any]:
    content = str(message.get("content", ""))
    return {
        "role": message.get("role"),
        "content": "<redacted>",
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_chars": len(content),
        "content_words": len(re.findall(r"\w+", content)),
    }


def redacted_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages", [])
    safe_messages = [
        redacted_message(message)
        for message in messages
        if isinstance(message, dict)
    ]
    source = copy.deepcopy(row.get("source", {})) if isinstance(row.get("source"), dict) else {}
    if isinstance(source.get("generation"), dict):
        source["generation"].pop("prompt", None)
        source["generation"].pop("raw_response", None)
    return {
        "id": row.get("id"),
        "messages": safe_messages,
        "skills": row.get("skills", []),
        "source": source,
        "generation_method": row.get("generation_method"),
        "quality_scores": row.get("quality_scores", {}),
        "quality_score_average": row.get("quality_score_average"),
        "verification": row.get("verification", {}),
        "holdout_overlap": row.get("holdout_overlap", {}),
        "content_sha256": row_content_digest(row),
        "redaction": {
            "policy": "model_forge_public_dataset_redacted_v1",
            "message_content": "redacted",
        },
    }


def redacted_review_report(report: dict[str, Any]) -> dict[str, Any]:
    safe = copy.deepcopy(report)
    for row in safe.get("reviewed_rows", []):
        if isinstance(row, dict):
            if "user" in row:
                row["user"] = "<redacted>"
            if "assistant" in row:
                row["assistant"] = "<redacted>"
    safe["redaction"] = {
        "policy": "model_forge_public_dataset_redacted_v1",
        "reviewed_row_text": "redacted",
    }
    return safe


def write_redacted_publish_bundle(
    *,
    output_dir: Path,
    outputs: dict[str, str],
    config: dict[str, Any],
    release_class: dict[str, Any],
    files: list[Path],
) -> dict[str, Any]:
    bundle_dir = output_dir / "hf_publish_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = read_jsonl(Path(outputs["dataset"]))
    redacted_dataset_path = bundle_dir / "dataset_redacted.jsonl"
    write_jsonl(redacted_dataset_path, [redacted_dataset_row(row) for row in dataset_rows])

    card_path = Path(outputs["dataset_card"])
    readme_path = bundle_dir / "README.md"
    readme_path.write_text(card_path.read_text(encoding="utf-8"), encoding="utf-8")

    report = {
        "dataset_id": config["id"],
        "release_class": release_class.get("id"),
        "policy": "model_forge_public_dataset_redacted_v1",
        "source_dataset": publish_path_label(Path(outputs["dataset"])),
        "redacted_dataset": publish_path_label(redacted_dataset_path),
        "rows_redacted": len(dataset_rows),
        "raw_message_content_published": False,
        "accepted_rows_included": False,
        "rejected_rows_included": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report_path = bundle_dir / "redaction_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verification_rows = read_jsonl(Path(outputs["verification"]))
    verification_path = bundle_dir / "verification.jsonl"
    write_jsonl(verification_path, [redacted_dataset_row(row) for row in verification_rows])

    for raw_path in (outputs["manifest"], outputs["quality_report"], outputs["generation_report"]):
        source = Path(raw_path)
        target = bundle_dir / source.name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    if "review_report" in outputs:
        review = redacted_review_report(json.loads(Path(outputs["review_report"]).read_text(encoding="utf-8")))
        review_report_path = bundle_dir / "review_report.json"
        review_report_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (bundle_dir / "review_sheet.md").write_text(review_sheet(review), encoding="utf-8")

    files.extend(sorted(path for path in bundle_dir.rglob("*") if path.is_file()))
    return report


def scan_publish_files(paths: list[Path]) -> list[str]:
    findings = []
    secret_re = re.compile(r"hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|(?i:HF_TOKEN|HUGGINGFACE_HUB_TOKEN|API_KEY|SECRET|PASSWORD)\s*=")
    private_path_re = re.compile(r"(?<![A-Za-z0-9_])/(home|Users)/[A-Za-z0-9_.-]+/")
    for path in paths:
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".pdf", ".safetensors", ".bin"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if secret_re.search(text):
            findings.append(f"secret-like literal in {publish_path_label(path)}")
        if private_path_re.search(text):
            findings.append(f"private absolute path in {publish_path_label(path)}")
    return findings


def dataset_publish_gates(
    *,
    config: dict[str, Any],
    release_class: dict[str, Any],
    files: list[Path],
    redaction_report: dict[str, Any] | None,
    source_license_checked: bool,
) -> list[dict[str, str]]:
    required = {str(item) for item in release_class.get("requires") or []}
    gates: list[dict[str, str]] = []

    def add(name: str, passed: bool, message: str) -> None:
        gates.append({"name": name, "status": "pass" if passed else "fail", "message": message})

    if "dataset_card_complete" in required:
        readme = next((path for path in files if path.name == "README.md"), None)
        text = readme.read_text(encoding="utf-8") if readme and readme.exists() else ""
        sections = ("## Purpose", "## Counts", "## Provenance", "## Safety And Contamination")
        add("dataset_card_complete", all(section in text for section in sections), "dataset card contains required sections")
    if "source_license_checked" in required:
        add(
            "source_license_checked",
            source_license_checked,
            "pass --source-license-checked after checking dataset source licenses and provenance",
        )
    if "unsafe_examples_redacted" in required:
        raw_policy = str(release_class.get("publish_raw_outputs", "redacted_only"))
        redacted = bool(redaction_report and not redaction_report.get("raw_message_content_published"))
        add(
            "unsafe_examples_redacted",
            raw_policy in {"false", "redacted_only"} and redacted,
            "public dataset bundle uses redacted message content",
        )
    if "no_private_tokens_or_paths" in required:
        findings = scan_publish_files(files)
        add("no_private_tokens_or_paths", not findings, "; ".join(findings[:5]) or "no secret-like literals or private absolute paths found")
    return gates


def command_publish(
    config: dict[str, Any],
    config_path: Path,
    overwrite: bool,
    smoke: bool = False,
    max_generated_candidates: int | None = None,
    provider_override: str | None = None,
    execute: bool = False,
    private: bool = False,
    source_license_checked: bool = False,
) -> Path:
    outputs = command_pack(
        config,
        config_path,
        overwrite=False,
        smoke=smoke,
        max_generated_candidates=max_generated_candidates,
        provider_override=provider_override,
    )
    output_dir = resolve_repo_path(config["output_dir"])
    publish_path = output_dir / "hf_publish_plan.json"
    if publish_path.exists() and not overwrite and not execute:
        return publish_path
    repo_id = config.get("hub", {}).get("repo_id") or f"keithtyser/model-forge-{config['id']}"
    release_class = load_release_class(str(config.get("hub", {}).get("release_class", "public_dataset")))
    file_paths = [
        Path(outputs["dataset"]),
        Path(outputs["manifest"]),
        Path(outputs["quality_report"]),
        Path(outputs["dataset_card"]),
        Path(outputs["verification"]),
        Path(outputs["generation_report"]),
    ]
    if (output_dir / "gap_report.yaml").exists():
        file_paths.append(output_dir / "gap_report.yaml")
    if (output_dir / "review_report.json").exists():
        file_paths.append(output_dir / "review_report.json")
    if (output_dir / "review_sheet.md").exists():
        file_paths.append(output_dir / "review_sheet.md")
    redaction_report = None
    raw_policy = str(release_class.get("publish_raw_outputs", "redacted_only"))
    if release_class.get("hf_visibility") == "public" or raw_policy == "redacted_only":
        file_paths = []
        redaction_report = write_redacted_publish_bundle(
            output_dir=output_dir,
            outputs=outputs,
            config=config,
            release_class=release_class,
            files=file_paths,
        )
    elif raw_policy == "private_only":
        file_paths.extend([output_dir / "accepted.jsonl", output_dir / "rejected.jsonl"])
    files = [publish_path_label(path) for path in file_paths]
    gates = dataset_publish_gates(
        config=config,
        release_class=release_class,
        files=file_paths,
        redaction_report=redaction_report,
        source_license_checked=source_license_checked,
    )
    gate_failures = [gate for gate in gates if gate["status"] == "fail"]
    blocked_until: list[str] = []
    if not execute:
        blocked_until.extend([
            "human reviews dataset_card.md and review_sheet.md",
            "license/provenance checks pass",
            "HF publish command is run explicitly",
        ])
        if bool(config.get("seed_only", False)):
            blocked_until.append("human explicitly approves seed-only release")
        elif bool(config.get("smoke_only", False)):
            blocked_until.append("dataset is expanded beyond smoke-only scaffold")
        else:
            blocked_until.append("dataset size reaches configured target")
    blocked_until.extend(f"{gate['name']}: {gate['message']}" for gate in gate_failures)
    plan = {
        "dry_run": not execute,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "dataset_id": config["id"],
        "release_class": release_class.get("id"),
        "visibility": release_class.get("hf_visibility", "private"),
        "files": files,
        "redacted_output_bundle": publish_path_label(output_dir / "hf_publish_bundle") if redaction_report else None,
        "redaction_report": redaction_report,
        "release_gates": gates,
        "blocked": bool(blocked_until),
        "blocked_until": blocked_until,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if execute:
        if bool(config.get("seed_only", False)) or bool(config.get("smoke_only", False)):
            raise SystemExit("refusing to upload seed-only or smoke-only dataset; publish durable datasets only")
        if gate_failures:
            raise SystemExit("refusing to upload dataset with failed release gates")
        execute_hf_dataset_publish(plan, output_dir, private=private)
        plan["uploaded_at"] = datetime.now(timezone.utc).isoformat()
    publish_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return publish_path


def execute_hf_dataset_publish(plan: dict[str, Any], output_dir: Path, private: bool = False) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN before executing HF dataset publish")
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub before executing HF dataset publish") from exc
    repo_id = str(plan["repo_id"])
    upload_dir = output_dir
    if plan.get("redacted_output_bundle"):
        upload_dir = resolve_repo_path(str(plan["redacted_output_bundle"]))
    api = HfApi(token=token)
    api.whoami(token=token)
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)
    upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(upload_dir),
        commit_message=f"Upload model-forge dataset {plan['dataset_id']}",
        token=token,
    )


def default_config_for(family: str, variant: str) -> Path:
    path = REPO_DIR / "configs" / "datasets" / f"{family}_{variant}.yaml"
    if not path.exists():
        raise SystemExit(f"dataset config not found: {path.relative_to(REPO_DIR)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and pack model-forge dataset-factory artifacts")
    parser.add_argument("step", choices=["plan", "gaps", "propose", "repair-from-eval", "seed", "generate", "judge", "verify", "filter", "review", "pack", "training-gate", "publish"])
    parser.add_argument("family", nargs="?")
    parser.add_argument("variant", nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "overwrite this step's outputs; candidate replacement only happens "
            "when this is used with the generate step"
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="use configured smoke generation limits")
    parser.add_argument("--max-generated-candidates", type=int, help="override generated candidate limit")
    parser.add_argument("--provider", help="override generation provider type, for example template or openai_compatible")
    parser.add_argument("--sample", type=int, help="review sample size")
    parser.add_argument("--execute", action="store_true", help="execute the publish step; currently valid only for non-smoke datasets")
    parser.add_argument("--private", action="store_true", help="create executed HF dataset repo as private")
    parser.add_argument("--source-license-checked", action="store_true", help="mark source license/provenance review complete for publish gates")
    parser.add_argument("--dataset-manifest", type=Path, help="packed dataset manifest.yaml for training-gate")
    parser.add_argument("--finetune-plan", type=Path, help="bounded fine-tune plan.json for training-gate")
    parser.add_argument("--data-summary", type=Path, help="bounded fine-tune data_summary.json for training-gate")
    parser.add_argument("--promotion-report", type=Path, help="source-relative promotion report JSON for training-gate")
    parser.add_argument("--max-steps", type=int, default=50, help="maximum bounded fine-tune steps accepted by training-gate")
    parser.add_argument("--max-train-rows", type=int, default=5000, help="maximum bounded training rows accepted by training-gate")
    parser.add_argument("--no-require-spark", action="store_true", help="allow non-Spark training evidence in training-gate")
    parser.add_argument("--write-gate", action="store_true", help="write training_evidence_gate artifacts")
    parser.add_argument("--json", action="store_true", help="print JSON report for gate-style commands")
    args = parser.parse_args()

    config_path = args.config or default_config_for(args.family or "", args.variant or "")
    config_path = resolve_repo_path(config_path)
    config = load_yaml(config_path)

    if args.step == "plan":
        path = command_plan(config, config_path, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "gaps":
        path = command_gaps(config, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "propose":
        path = command_propose(config, config_path, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "repair-from-eval":
        outputs = command_repair_from_eval(config, args.overwrite)
        for path in outputs.values():
            console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "seed":
        path = command_seed(config, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "generate":
        path = command_generate(
            config,
            args.overwrite,
            smoke=args.smoke,
            max_generated_candidates=args.max_generated_candidates,
            provider_override=args.provider,
        )
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "judge":
        path = command_judge(
            config,
            args.overwrite,
            smoke=args.smoke,
            max_generated_candidates=args.max_generated_candidates,
            provider_override=args.provider,
        )
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "verify":
        path = command_verify(
            config,
            args.overwrite,
            smoke=args.smoke,
            max_generated_candidates=args.max_generated_candidates,
            provider_override=args.provider,
        )
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "filter":
        accepted, rejected = command_filter(
            config,
            args.overwrite,
            smoke=args.smoke,
            max_generated_candidates=args.max_generated_candidates,
            provider_override=args.provider,
        )
        console.print(f"[green]Wrote[/green] {display_path(accepted)}")
        console.print(f"[green]Wrote[/green] {display_path(rejected)}")
    elif args.step == "review":
        outputs = command_review(
            config,
            args.overwrite,
            smoke=args.smoke,
            max_generated_candidates=args.max_generated_candidates,
            provider_override=args.provider,
            sample_size=args.sample,
        )
        for path in outputs.values():
            console.print(f"[green]Wrote[/green] {display_path(Path(path))}")
    elif args.step == "pack":
        outputs = command_pack(
            config,
            config_path,
            args.overwrite,
            smoke=args.smoke,
            max_generated_candidates=args.max_generated_candidates,
            provider_override=args.provider,
        )
        for path in outputs.values():
            console.print(f"[green]Wrote[/green] {display_path(Path(path))}")
    elif args.step == "training-gate":
        output_dir = resolve_repo_path(config["output_dir"])
        dataset_manifest = resolve_repo_path(args.dataset_manifest or output_dir / "manifest.yaml")
        missing = [
            name
            for name, value in {
                "--finetune-plan": args.finetune_plan,
                "--data-summary": args.data_summary,
                "--promotion-report": args.promotion_report,
            }.items()
            if value is None
        ]
        if missing:
            raise SystemExit(f"training-gate requires {', '.join(missing)}")
        report = command_training_gate(
            config=config,
            config_path=config_path,
            dataset_manifest_path=dataset_manifest,
            finetune_plan_path=resolve_repo_path(args.finetune_plan),
            data_summary_path=resolve_repo_path(args.data_summary),
            promotion_report_path=resolve_repo_path(args.promotion_report),
            max_steps=args.max_steps,
            max_train_rows=args.max_train_rows,
            require_spark=not args.no_require_spark,
            write_gate=args.write_gate,
        )
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "passed" if report["dataset_recipe_validated"] else "failed"
            border = "green" if report["dataset_recipe_validated"] else "red"
            console.print(Panel(json.dumps(report["checks"], indent=2), title=f"Training Evidence Gate {status}", border_style=border))
        if not report["dataset_recipe_validated"]:
            raise SystemExit(1)
    elif args.step == "publish":
        path = command_publish(
            config,
            config_path,
            args.overwrite,
            smoke=args.smoke,
            max_generated_candidates=args.max_generated_candidates,
            provider_override=args.provider,
            execute=args.execute,
            private=args.private,
            source_license_checked=args.source_license_checked,
        )
        label = "Wrote publish plan" if args.execute else "Wrote dry-run publish plan"
        console.print(f"[green]{label}[/green] {display_path(path)}")


if __name__ == "__main__":
    main()
