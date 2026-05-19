from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table

REPO_DIR = Path(__file__).resolve().parents[3]
console = Console()


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


def row_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    if not isinstance(messages, list):
        return ""
    parts = []
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            parts.append(message["content"])
    return "\n".join(parts)


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


def build_plan(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    output_dir = resolve_repo_path(config["output_dir"])
    objective_path = REPO_DIR / "configs" / "objectives" / f"{config['objective']}.yaml"
    objective = load_yaml(objective_path) if objective_path.exists() else {}
    seeds = load_seed_rows(config)
    seed_counts = Counter(skill for row in seeds for skill in row.get("skills", []))
    return {
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
        "seed_only": bool(config.get("seed_only", False)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


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


def command_generate(config: dict[str, Any], overwrite: bool) -> Path:
    seed_path = command_seed(config, overwrite=overwrite)
    output_path = resolve_repo_path(config["output_dir"]) / "candidates.jsonl"
    if output_path.exists() and not overwrite:
        return output_path
    rows = read_jsonl(seed_path)
    candidates = []
    for row in rows:
        item = dict(row)
        item["generation_method"] = item.get("source", {}).get("kind", "human_seed")
        item["dataset_factory_stage"] = "candidate"
        candidates.append(item)
    write_jsonl(output_path, candidates)
    return output_path


def command_judge(config: dict[str, Any], overwrite: bool) -> Path:
    candidate_path = command_generate(config, overwrite=overwrite)
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


def command_verify(config: dict[str, Any], overwrite: bool) -> Path:
    judged_path = command_judge(config, overwrite=overwrite)
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
    verification = row.get("verification", {}) if isinstance(row.get("verification"), dict) else {}
    if verification and not verification.get("passed", False):
        reasons.append("verification_failed")
    overlap = holdout_overlap(row, holdouts)
    row["holdout_overlap"] = overlap
    if float(overlap.get("max_similarity", 0.0)) > float(thresholds.get("max_holdout_similarity", 1.0)):
        reasons.append("holdout_overlap_above_threshold")
    return reasons


def command_filter(config: dict[str, Any], overwrite: bool) -> tuple[Path, Path]:
    verified_path = command_verify(config, overwrite=overwrite)
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


def quality_report(config: dict[str, Any], accepted: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    skill_counts = Counter(skill for row in accepted for skill in row.get("skills", []))
    rejection_counts = Counter(reason for row in rejected for reason in row.get("rejection_reasons", []))
    averages = [float(row.get("quality_score_average", 0.0)) for row in accepted]
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
        "verification_counts": dict(sorted(verification_counts.items())),
        "quality_score_average": round(sum(averages) / len(averages), 4) if averages else 0.0,
        "target_accept_count": copy.deepcopy(config.get("target_accept_count", {})),
        "seed_only": bool(config.get("seed_only", False)),
        "warnings": warnings,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def dataset_card(config: dict[str, Any], report: dict[str, Any]) -> str:
    skills = "\n".join(f"- `{skill}`: {count}" for skill, count in report["skill_counts"].items()) or "- none"
    warnings = "\n".join(f"- `{item['code']}`: {item['message']}" for item in report.get("warnings", [])) or "- none"
    verification_counts = report.get("verification_counts", {})
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

## Skill Counts

{skills}

## Coverage Warnings

{warnings}

## Provenance

Rows are currently human-seeded and heuristically judged. Future versions can
add teacher-model generation, executable verification, and HF publication using
the same manifest layout.

## Safety And Contamination

The pack step writes `accepted.jsonl` and `rejected.jsonl`, records rejection
reasons, and checks similarity against configured holdout prompt files.
"""


def command_pack(config: dict[str, Any], config_path: Path, overwrite: bool) -> dict[str, str]:
    accepted_path, rejected_path = command_filter(config, overwrite=overwrite)
    output_dir = resolve_repo_path(config["output_dir"])
    verification_path = output_dir / "verification.jsonl"
    dataset_path = output_dir / "dataset.jsonl"
    manifest_path = output_dir / "manifest.yaml"
    report_path = output_dir / "quality_report.json"
    card_path = output_dir / "dataset_card.md"
    gap_path = output_dir / "gap_report.yaml"
    if all(path.exists() for path in (dataset_path, manifest_path, report_path, card_path, verification_path)) and not overwrite:
        return {
            "dataset": str(dataset_path),
            "manifest": str(manifest_path),
            "quality_report": str(report_path),
            "dataset_card": str(card_path),
            "verification": str(verification_path),
        }

    accepted = read_jsonl(accepted_path)
    rejected = read_jsonl(rejected_path)
    dataset_rows = [
        {
            "id": row["id"],
            "messages": row["messages"],
            "skills": row.get("skills", []),
            "source": row.get("source", {}),
            "quality_scores": row.get("quality_scores", {}),
            "verification": row.get("verification", {}),
            "holdout_overlap": row.get("holdout_overlap", {}),
        }
        for row in accepted
    ]
    write_jsonl(dataset_path, dataset_rows)

    report = quality_report(config, accepted, rejected)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = build_plan(config, config_path)
    manifest.update({
        "artifacts": {
            "dataset": display_path(dataset_path),
            "accepted": display_path(accepted_path),
            "rejected": display_path(rejected_path),
            "verification": display_path(verification_path) if verification_path.exists() else None,
            "quality_report": display_path(report_path),
            "dataset_card": display_path(card_path),
            "gap_report": display_path(gap_path) if gap_path.exists() else None,
        },
        "quality_report": report,
    })
    write_yaml(manifest_path, manifest)
    card_path.write_text(dataset_card(config, report), encoding="utf-8")
    return {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "quality_report": str(report_path),
        "dataset_card": str(card_path),
        "verification": str(verification_path),
    }


def command_publish(config: dict[str, Any], config_path: Path, overwrite: bool) -> Path:
    outputs = command_pack(config, config_path, overwrite=False)
    output_dir = resolve_repo_path(config["output_dir"])
    publish_path = output_dir / "hf_publish_plan.json"
    if publish_path.exists() and not overwrite:
        return publish_path
    repo_id = config.get("hub", {}).get("repo_id") or f"keithtyser/model-forge-{config['id']}"
    files = [
        display_path(Path(outputs["dataset"])),
        display_path(Path(outputs["manifest"])),
        display_path(Path(outputs["quality_report"])),
        display_path(Path(outputs["dataset_card"])),
        display_path(Path(outputs["verification"])),
        display_path(output_dir / "accepted.jsonl"),
        display_path(output_dir / "rejected.jsonl"),
    ]
    if (output_dir / "gap_report.yaml").exists():
        files.append(display_path(output_dir / "gap_report.yaml"))
    blocked_until = [
        "human reviews dataset_card.md",
        "license/provenance checks pass",
        "HF publish command is run explicitly",
    ]
    if bool(config.get("seed_only", False)):
        blocked_until.append("human explicitly approves seed-only release")
    else:
        blocked_until.append("dataset size reaches configured target")
    plan = {
        "dry_run": True,
        "repo_id": repo_id,
        "repo_type": "dataset",
        "dataset_id": config["id"],
        "release_class": config.get("hub", {}).get("release_class", "public_dataset_candidate"),
        "files": files,
        "blocked_until": blocked_until,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    publish_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return publish_path


def default_config_for(family: str, variant: str) -> Path:
    path = REPO_DIR / "configs" / "datasets" / f"{family}_{variant}.yaml"
    if not path.exists():
        raise SystemExit(f"dataset config not found: {path.relative_to(REPO_DIR)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and pack model-forge dataset-factory artifacts")
    parser.add_argument("step", choices=["plan", "gaps", "seed", "generate", "judge", "verify", "filter", "pack", "publish"])
    parser.add_argument("family", nargs="?")
    parser.add_argument("variant", nargs="?")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
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
    elif args.step == "seed":
        path = command_seed(config, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "generate":
        path = command_generate(config, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "judge":
        path = command_judge(config, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "verify":
        path = command_verify(config, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(path)}")
    elif args.step == "filter":
        accepted, rejected = command_filter(config, args.overwrite)
        console.print(f"[green]Wrote[/green] {display_path(accepted)}")
        console.print(f"[green]Wrote[/green] {display_path(rejected)}")
    elif args.step == "pack":
        outputs = command_pack(config, config_path, args.overwrite)
        for path in outputs.values():
            console.print(f"[green]Wrote[/green] {display_path(Path(path))}")
    elif args.step == "publish":
        path = command_publish(config, config_path, args.overwrite)
        console.print(f"[green]Wrote dry-run publish plan[/green] {display_path(path)}")


if __name__ == "__main__":
    main()
