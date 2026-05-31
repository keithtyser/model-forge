from __future__ import annotations

import argparse
import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from model_forge.benchmarks.sweep import DEFAULT_CONFIG as DEFAULT_SERVING_SWEEP_CONFIG
from model_forge.benchmarks.sweep import build_sweep_plan, load_sweep_config
from model_forge.pipelines.abliterate import build_plan as build_abliteration_plan
from model_forge.quantization.cli import DEFAULT_CONFIG as DEFAULT_QUANTIZATION_CONFIG
from model_forge.quantization.cli import filter_matrix_entries, load_quantization_config, matrix_entries
from model_forge.runs.manifest import REPO_DIR, display_path, sanitize_run_id


SCHEMA_CONFIG = REPO_DIR / "configs" / "agents" / "experiment_schema.yaml"
DEFAULT_TEMPLATE = REPO_DIR / "recipes" / "agents" / "agent_experiment_template.yaml"
AGENT_EXPERIMENT_VERSION = "model_forge.agent_experiment.v1"


@dataclass(frozen=True)
class Finding:
    check: str
    message: str
    path: str | None = None
    field: str | None = None


def shlex_quote(value: Any) -> str:
    return shlex.quote(str(value))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(path)}")
    return data


def load_schema(path: Path = SCHEMA_CONFIG) -> dict[str, Any]:
    return load_yaml(path)


def dotted_get(data: Mapping[str, Any], field: str) -> Any:
    current: Any = data
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def field_missing(data: Mapping[str, Any], field: str) -> bool:
    value = dotted_get(data, field)
    return value in (None, "", [])


def validate_agent_experiment(
    plan: Mapping[str, Any],
    *,
    schema: Mapping[str, Any] | None = None,
    path: str | None = None,
    repo_dir: Path = REPO_DIR,
) -> list[Finding]:
    schema = schema or load_schema()
    findings: list[Finding] = []
    if plan.get("schema_version") != AGENT_EXPERIMENT_VERSION:
        findings.append(Finding("agent_experiment", f"schema_version must be {AGENT_EXPERIMENT_VERSION}", path, "schema_version"))
    for field in schema.get("required_fields") or []:
        if field_missing(plan, str(field)):
            findings.append(Finding("agent_experiment", "required field is missing", path, str(field)))

    allowed_statuses = set(str(item) for item in schema.get("allowed_statuses") or [])
    if plan.get("status") and str(plan["status"]) not in allowed_statuses:
        findings.append(Finding("agent_experiment", "unsupported status", path, "status"))
    allowed_types = set(str(item) for item in schema.get("allowed_experiment_types") or [])
    if plan.get("experiment_type") and str(plan["experiment_type"]) not in allowed_types:
        findings.append(Finding("agent_experiment", "unsupported experiment_type", path, "experiment_type"))

    family = str(plan.get("family") or "")
    variant = str(plan.get("variant") or "")
    objective_profile = str(plan.get("objective_profile") or "")
    if family:
        family_path = repo_dir / "configs" / "model_families" / f"{family}.yaml"
        if not family_path.exists():
            findings.append(Finding("agent_experiment", "family config does not exist", path, "family"))
        elif variant:
            family_config = load_yaml(family_path)
            if variant not in (family_config.get("variants") or {}):
                findings.append(Finding("agent_experiment", "variant is not defined for family", path, "variant"))
    if objective_profile:
        objective_path = repo_dir / "configs" / "objectives" / f"{objective_profile}.yaml"
        if not objective_path.exists():
            findings.append(Finding("agent_experiment", "objective profile config does not exist", path, "objective_profile"))

    commands = plan.get("planned_commands") or []
    if not isinstance(commands, list):
        findings.append(Finding("agent_experiment", "planned_commands must be a list", path, "planned_commands"))
        commands = []
    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            findings.append(Finding("agent_experiment", "planned command must be a mapping", path, f"planned_commands.{index}"))
            continue
        for field in schema.get("planned_command_required_fields") or []:
            if field_missing(command, str(field)):
                findings.append(Finding("agent_experiment", "planned command field is missing", path, f"planned_commands.{index}.{field}"))
        raw_command = str(command.get("command") or "")
        for pattern in schema.get("secret_patterns") or []:
            if re.search(str(pattern), raw_command):
                findings.append(Finding("agent_experiment", "planned command contains a secret-like value", path, f"planned_commands.{index}.command"))
        if bool(command.get("starts_heavy_job")) and not bool((plan.get("resource_policy") or {}).get("checkpoint_or_write_plan_before_execute")):
            findings.append(
                Finding(
                    "agent_experiment",
                    "heavy planned commands require checkpoint_or_write_plan_before_execute",
                    path,
                    f"planned_commands.{index}.starts_heavy_job",
                )
            )

    resource_policy = plan.get("resource_policy") or {}
    if not isinstance(resource_policy, Mapping):
        findings.append(Finding("agent_experiment", "resource_policy must be a mapping", path, "resource_policy"))
        resource_policy = {}
    for field in schema.get("required_resource_policy_fields") or []:
        if field_missing(resource_policy, str(field)):
            findings.append(Finding("agent_experiment", "resource policy field is missing", path, f"resource_policy.{field}"))

    evidence_plan = plan.get("evidence_plan") or {}
    if not isinstance(evidence_plan, Mapping):
        findings.append(Finding("agent_experiment", "evidence_plan must be a mapping", path, "evidence_plan"))
        evidence_plan = {}
    for field in schema.get("required_evidence_plan_fields") or []:
        if field_missing(evidence_plan, str(field)):
            findings.append(Finding("agent_experiment", "evidence plan field is missing", path, f"evidence_plan.{field}"))

    handoff = plan.get("handoff") or {}
    if not isinstance(handoff, Mapping):
        findings.append(Finding("agent_experiment", "handoff must be a mapping", path, "handoff"))
        handoff = {}
    for field in schema.get("required_handoff_fields") or []:
        if field_missing(handoff, str(field)):
            findings.append(Finding("agent_experiment", "handoff field is missing", path, f"handoff.{field}"))

    return findings


def tracked_agent_experiment_paths(repo_dir: Path = REPO_DIR) -> list[Path]:
    roots = [repo_dir / "recipes" / "agents"]
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.glob("*.yaml")))
    return paths


def audit_agent_experiments(repo_dir: Path = REPO_DIR) -> list[Finding]:
    findings: list[Finding] = []
    schema = load_schema(repo_dir / "configs" / "agents" / "experiment_schema.yaml")
    for path in tracked_agent_experiment_paths(repo_dir):
        relative = display_path(path)
        try:
            plan = load_yaml(path)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            findings.append(Finding("agent_experiment", f"could not parse plan: {exc}", relative))
            continue
        findings.extend(validate_agent_experiment(plan, schema=schema, path=relative, repo_dir=repo_dir))
    return findings


def init_agent_experiment(args: argparse.Namespace) -> dict[str, Any]:
    template = load_yaml(args.template)
    experiment_id = sanitize_run_id(args.experiment_id)
    template["experiment_id"] = experiment_id
    template["title"] = args.title
    template["family"] = args.family
    template["variant"] = args.variant
    template["experiment_type"] = args.experiment_type
    template["objective_profile"] = args.objective_profile
    if args.hypothesis:
        template["hypothesis"] = args.hypothesis
    return template


def optimize_serving_plan(args: argparse.Namespace) -> dict[str, Any]:
    sweep_config, sweep_path = load_sweep_config(args.sweep_config)
    sweep_plan = build_sweep_plan(
        sweep_config,
        sweep_path,
        family=args.family,
        variant=args.variant,
        base_url=args.base_url,
        cluster_config=args.cluster_config,
    )
    experiment_id = sanitize_run_id(args.experiment_id or f"{args.family}_{args.variant}_serving_optimization")
    target = sweep_plan["target"]
    planned_commands: list[dict[str, Any]] = [
        {
            "command": f"./forge bench sweep doctor --config {display_path(sweep_path)} --strict",
            "purpose": "Validate the serving sweep config before planning or running cases.",
            "starts_heavy_job": False,
            "requires_execute": False,
            "expected_artifacts": ["terminal_output"],
        },
        {
            "command": (
                f"./forge bench sweep plan --config {display_path(sweep_path)} "
                f"--family {args.family} --variant {args.variant} --write-plan"
            ),
            "purpose": "Write the concrete serving sweep plan and per-case benchmark commands.",
            "starts_heavy_job": False,
            "requires_execute": False,
            "expected_artifacts": ["sweep_plan.json", "bench_commands.sh"],
        },
    ]
    if args.cluster_config:
        planned_commands.append(
            {
                "command": f"./forge cluster health --config {display_path(args.cluster_config)}",
                "purpose": "Verify the configured Spark cluster is reachable before serving optimization.",
                "starts_heavy_job": False,
                "requires_execute": False,
                "expected_artifacts": ["terminal_output"],
            }
        )
    for case in sweep_plan.get("cases") or []:
        env_prefix = " ".join(f"{key}={shlex_quote(value)}" for key, value in sorted((case.get("server_env") or {}).items()))
        planned_commands.append(
            {
                "command": f"{env_prefix} ./forge serve {args.family} {args.variant}".strip(),
                "purpose": f"Start exactly one vLLM server for serving sweep case {case['id']}.",
                "starts_heavy_job": True,
                "requires_execute": True,
                "expected_artifacts": ["running_openai_compatible_endpoint"],
            }
        )
        planned_commands.append(
            {
                "command": str(case["bench_command"]),
                "purpose": f"Benchmark serving sweep case {case['id']} after its server is healthy.",
                "starts_heavy_job": False,
                "requires_execute": True,
                "expected_artifacts": [f"{case['output_dir']}/summary.json", f"{case['output_dir']}/serving_card.md"],
            }
        )
    for command in (sweep_plan.get("quality_gate") or {}).get("required_after_sweep") or []:
        planned_commands.append(
            {
                "command": str(command).replace("<family>", args.family).replace("<variant>", args.variant),
                "purpose": "Run sampled quality/behavior validation under the selected serving configuration before promotion.",
                "starts_heavy_job": False,
                "requires_execute": True,
                "expected_artifacts": ["eval_manifest", "scores.csv", "eval_provenance_card.json"],
            }
        )

    resource_policy = sweep_plan.get("resource_policy") or {}
    return {
        "schema_version": AGENT_EXPERIMENT_VERSION,
        "experiment_id": experiment_id,
        "title": args.title or f"Optimize serving for {args.family}/{args.variant}",
        "hypothesis": args.hypothesis
        or (
            "A bounded serving sweep can improve throughput or latency for the target model "
            "without losing sampled quality/behavior once the selected case is re-evaluated."
        ),
        "owner_agent": args.owner_agent,
        "experiment_type": "serving",
        "status": "planned",
        "family": args.family,
        "variant": args.variant,
        "objective_profile": args.objective_profile,
        "planned_commands": planned_commands,
        "resource_policy": {
            "max_concurrent_large_jobs": int(resource_policy.get("max_concurrent_servers") or 1),
            "start_if_memory_available_above_fraction": float(resource_policy.get("start_if_memory_available_above_fraction") or 0.05),
            "stop_if_memory_available_below_fraction": float(resource_policy.get("stop_if_memory_available_below_fraction") or 0.05),
            "require_disk_free_fraction": float(resource_policy.get("require_disk_free_fraction") or 0.15),
            "use_cluster_when_heavy": True,
            "checkpoint_or_write_plan_before_execute": True,
        },
        "evidence_plan": {
            "manifest_required": True,
            "ledger_update_required": True,
            "expected_reports": [
                "sweep_plan.json",
                "bench_commands.sh",
                "summary.json",
                "serving_card.md",
                "manifest.json",
            ],
            "required_validation_commands": [
                f"./forge bench sweep doctor --config {display_path(sweep_path)} --strict",
                "./forge doctor --json",
            ],
        },
        "success_criteria": [
            "At least one candidate case improves the selected serving metric versus baseline.",
            "The selected case has a serving card and canonical manifest.",
            "Sampled quality/behavior validation under the selected case has no unacceptable regression.",
            "The selected command, metrics, and caveats are recorded in docs/experiment-ledger.md.",
        ],
        "rollback_plan": [
            "Stop the active vLLM server before starting another case.",
            "Return to the baseline serving environment from the sweep plan if candidate quality or stability regresses.",
            "Do not promote serving settings without matching quality/behavior evidence.",
        ],
        "handoff": {
            "push_to_github": True,
            "update_status_docs": True,
            "hf_upload_policy": "not_applicable",
            "raw_artifact_policy": "keep_large_outputs_untracked",
            "notes": "This plan does not execute serving. Run one server case at a time and attach serving cards before promotion.",
        },
        "metadata": {
            "sweep": sweep_plan["sweep"],
            "target": target,
            "cluster": sweep_plan.get("cluster"),
            "case_count": len(sweep_plan.get("cases") or []),
        },
    }


def _quantization_variants(config: Any, variants: str | None) -> list[dict[str, Any]]:
    entries = matrix_entries(config)
    if entries:
        return filter_matrix_entries(entries, variants)
    source_variants = [
        item.strip()
        for item in str(variants or config.source_variant or "base").split(",")
        if item.strip()
    ]
    return [
        {
            "source_variant": source_variant,
            "target_variant": config.target_variant or f"{source_variant}_{config.method}_{config.backend}",
        }
        for source_variant in source_variants
    ]


def optimize_quantization_plan(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config)
    config = load_quantization_config(config_path)
    family = args.family or config.family
    if not family:
        raise ValueError("optimize-quantization needs --family or config.family")
    entries = _quantization_variants(config, args.variants)
    primary_variant = str(entries[0].get("source_variant") or config.source_variant or "base")
    experiment_id = sanitize_run_id(args.experiment_id or f"{family}_{primary_variant}_{config.method}_quantization_optimization")
    has_checkpoint_export = bool(config.export or matrix_entries(config))
    planned_commands: list[dict[str, Any]] = [
        {
            "command": f"./forge quantize plan --config {display_path(config_path)} --family {family} --variant {primary_variant} --write-plan",
            "purpose": "Resolve the quantization runtime/export contract without loading model weights.",
            "starts_heavy_job": False,
            "requires_execute": False,
            "expected_artifacts": ["quantization_plan.json", "quantization_plan.md"],
        },
    ]
    if matrix_entries(config):
        variants_arg = f" --variants {args.variants}" if args.variants else ""
        planned_commands.append(
            {
                "command": f"./forge quantize matrix-plan --config {display_path(config_path)}{variants_arg} --write-plan",
                "purpose": "Write export plans for the configured source-variant matrix.",
                "starts_heavy_job": False,
                "requires_execute": False,
                "expected_artifacts": ["quantization_export_plan.json"],
            }
        )
    for entry in entries:
        source_variant = str(entry.get("source_variant") or config.source_variant or primary_variant)
        target_variant = str(entry.get("target_variant") or config.target_variant or f"{source_variant}_{config.method}_{config.backend}")
        if has_checkpoint_export:
            planned_commands.append(
                {
                    "command": (
                        f"./forge quantize export {family} {source_variant} "
                        f"--config {display_path(config_path)} --target-variant {target_variant} --write-plan"
                    ),
                    "purpose": f"Write the guarded export plan for {family}/{source_variant} -> {target_variant}.",
                    "starts_heavy_job": False,
                    "requires_execute": False,
                    "expected_artifacts": ["quantization_export_plan.json"],
                }
            )
            planned_commands.append(
                {
                    "command": (
                        f"./forge quantize export {family} {source_variant} "
                        f"--config {display_path(config_path)} --target-variant {target_variant} --execute"
                    ),
                    "purpose": f"Run the guarded quantization export for {target_variant}.",
                    "starts_heavy_job": True,
                    "requires_execute": True,
                    "expected_artifacts": ["quantized_checkpoint", "quantization_export_plan.json"],
                }
            )
        planned_commands.append(
            {
                "command": f"./forge serve {family} {target_variant}",
                "purpose": f"Serve the quantized candidate {target_variant} for loader and throughput validation.",
                "starts_heavy_job": True,
                "requires_execute": True,
                "expected_artifacts": ["running_openai_compatible_endpoint"],
            }
        )
        planned_commands.append(
            {
                "command": f"./forge eval {family} {target_variant} --smoke",
                "purpose": f"Run sampled quality/behavior validation for {target_variant}.",
                "starts_heavy_job": False,
                "requires_execute": True,
                "expected_artifacts": ["manifest.json", "scores.csv", "eval_provenance_card.json"],
            }
        )
        source_eval = str(entry.get("baseline_eval") or "<source_eval>")
        planned_commands.append(
            {
                "command": (
                    f"./forge quantize card --config {display_path(config_path)} "
                    "--source-serving-summary <source>/summary.json "
                    "--candidate-serving-summary <candidate>/summary.json "
                    f"--source-serving-eval {source_eval} "
                    "--candidate-serving-eval <candidate_eval> "
                    f"--run-id {sanitize_run_id(target_variant + '_quantization_card')} --write-card"
                ),
                "purpose": f"Write the source-vs-quantized promotion card for {target_variant}.",
                "starts_heavy_job": False,
                "requires_execute": False,
                "expected_artifacts": ["quantization_card.json", "quantization_card.md"],
            }
        )
    export = config.export or {}
    return {
        "schema_version": AGENT_EXPERIMENT_VERSION,
        "experiment_id": experiment_id,
        "title": args.title or f"Optimize quantization for {family}/{primary_variant}",
        "hypothesis": args.hypothesis
        or (
            "A guarded quantization workflow can improve serving throughput or memory efficiency "
            "while preserving source-variant behavior within the quantized_quality_retention gates."
        ),
        "owner_agent": args.owner_agent,
        "experiment_type": "quantization",
        "status": "planned",
        "family": family,
        "variant": primary_variant,
        "objective_profile": args.objective_profile,
        "planned_commands": planned_commands,
        "resource_policy": {
            "max_concurrent_large_jobs": 1,
            "start_if_memory_available_above_fraction": float(export.get("start_if_memory_available_above_fraction") or 0.05),
            "stop_if_memory_available_below_fraction": float(export.get("stop_if_memory_available_below_fraction") or 0.05),
            "require_disk_free_fraction": float(export.get("require_disk_free_fraction") or 0.15),
            "use_cluster_when_heavy": True,
            "checkpoint_or_write_plan_before_execute": True,
        },
        "evidence_plan": {
            "manifest_required": True,
            "ledger_update_required": True,
            "expected_reports": [
                "quantization_plan.json",
                *(["quantization_export_plan.json"] if has_checkpoint_export else []),
                "serving_card.md",
                "quantization_card.md",
                "manifest.json",
            ],
            "required_validation_commands": [
                f"./forge quantize plan --config {display_path(config_path)} --family {family} --variant {primary_variant} --json",
                "./forge doctor --json",
            ],
        },
        "success_criteria": [
            "Quantized candidate loads and serves through the target backend.",
            "Serving card shows the expected throughput or memory improvement versus source.",
            "Quantization card shows no unacceptable sampled quality or behavior regression.",
            "Tokenizer and chat-template preservation checks pass for the quantized candidate before release.",
            "The selected command, metrics, and caveats are recorded in docs/experiment-ledger.md.",
        ],
        "rollback_plan": [
            "Stop active export or serving jobs through the guarded launcher if system health degrades.",
            "Keep the unquantized source variant as the serving fallback.",
            "Do not publish quantized checkpoints that lack a quantization card and release-class approval.",
        ],
        "handoff": {
            "push_to_github": True,
            "update_status_docs": True,
            "hf_upload_policy": "upload_completed_quantized_checkpoints_only_after_release_gate",
            "raw_artifact_policy": "keep_large_outputs_untracked_or_upload_to_hf",
            "notes": (
                "This plan does not execute quantization. When checkpoint export is present, "
                "run one export per node and attach cards before promotion."
            ),
        },
        "metadata": {
            "config": display_path(config_path),
            "method": config.method,
            "backend": config.backend,
            "hardware_profile": config.hardware_profile,
            "has_checkpoint_export": has_checkpoint_export,
            "variant_count": len(entries),
            "variants": entries,
            "workers_env": (config.matrix or {}).get("workers_env"),
        },
    }


def _default_ablation_config_for_family(family: str) -> Path:
    return REPO_DIR / "configs" / "abliteration" / f"{family}_local_abli.yaml"


def _infer_family_from_ablation_config(config_path: Path) -> str | None:
    stem = config_path.stem
    suffixes = (
        "_ft_local_abli",
        "_local_abli",
        "_huihui_like",
        "_huihui_shaped",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return None


def _ablation_config_path(args: argparse.Namespace) -> Path:
    if args.config:
        return Path(args.config)
    if not args.family:
        raise ValueError("optimize-behavior-edit needs --family or --config")
    return _default_ablation_config_for_family(args.family)


def _family_variants(family: str) -> dict[str, Any]:
    family_path = REPO_DIR / "configs" / "model_families" / f"{family}.yaml"
    if not family_path.exists():
        return {}
    family_config = load_yaml(family_path)
    return dict(family_config.get("variants") or {})


def _infer_behavior_source_variant(family: str, plan: Mapping[str, Any], config_path: Path) -> str:
    variants = _family_variants(family)
    model = plan.get("model") or {}
    source_model = str(model.get("source") or "")
    local_dir = str(model.get("local_dir") or "")
    for name, variant in variants.items():
        if not isinstance(variant, Mapping):
            continue
        if source_model and source_model == str(variant.get("repo_id") or ""):
            return str(name)
        if local_dir and local_dir.endswith(str(variant.get("local_dir") or "\0")):
            return str(name)
    if "_ft_" in config_path.stem:
        return "local_ft" if "local_ft" in variants else "ft"
    return "base"


def _infer_behavior_target_variant(family: str, source_variant: str) -> str:
    variants = _family_variants(family)
    if source_variant == "base":
        for candidate in ("local_abli_sota", "local_abli", "abli"):
            if candidate in variants:
                return candidate
    scored: list[tuple[int, str]] = []
    for name, variant in variants.items():
        if not isinstance(variant, Mapping):
            continue
        if "abli" not in name:
            continue
        score = 0
        if variant.get("base_variant") == source_variant:
            score += 4
        if name.startswith(source_variant):
            score += 2
        if "sota" in name:
            score += 1
        scored.append((score, name))
    if scored:
        return sorted(scored, key=lambda item: (-item[0], item[1]))[0][1]
    return f"{source_variant}_local_abli"


def optimize_behavior_edit_plan(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _ablation_config_path(args)
    config = load_yaml(config_path)
    plan = build_abliteration_plan(config, config_path)
    family = args.family or _infer_family_from_ablation_config(config_path)
    if not family:
        raise ValueError("optimize-behavior-edit needs --family when it cannot infer one from --config")
    source_variant = args.source_variant
    target_variant = args.target_variant
    if not source_variant:
        source_variant = _infer_behavior_source_variant(family, plan, config_path)
    if not target_variant:
        target_variant = _infer_behavior_target_variant(family, source_variant)
    backend_arg = f" --backend {args.backend}" if args.backend else ""
    experiment_id = sanitize_run_id(args.experiment_id or f"{family}_{source_variant}_behavior_edit_optimization")
    config_display = display_path(config_path)
    planned_commands: list[dict[str, Any]] = [
        {
            "command": f"./forge ablate --config {config_display} plan",
            "purpose": "Validate the behavior-edit config, prompt counts, hardware profile, and memory guard without loading model weights.",
            "starts_heavy_job": False,
            "requires_execute": False,
            "expected_artifacts": ["terminal_output"],
        },
        {
            "command": f"./forge ablate --config {config_display} sota-plan{backend_arg}",
            "purpose": "Inspect the configured SOTA behavior-edit backend and selected recipe before writing runner artifacts.",
            "starts_heavy_job": False,
            "requires_execute": False,
            "expected_artifacts": ["terminal_output"],
        },
        {
            "command": f"./forge ablate --config {config_display} sota-prepare{backend_arg}",
            "purpose": "Write backend-specific behavior-edit runner/config artifacts without executing the edit.",
            "starts_heavy_job": False,
            "requires_execute": False,
            "expected_artifacts": ["sota_runner", "sota_config", "README.md"],
        },
        {
            "command": f"./forge ablate --config {config_display} sota-run{backend_arg} --execute",
            "purpose": f"Run the guarded behavior-edit backend for {family}/{source_variant} -> {target_variant}.",
            "starts_heavy_job": True,
            "requires_execute": True,
            "expected_artifacts": ["edited_checkpoint", "model_forge_sota_summary.json"],
        },
        {
            "command": f"./forge serve {family} {target_variant}",
            "purpose": f"Serve the edited candidate {target_variant} for loader and sampled quality validation.",
            "starts_heavy_job": True,
            "requires_execute": True,
            "expected_artifacts": ["running_openai_compatible_endpoint"],
        },
        {
            "command": f"./forge eval {family} {target_variant} --internal",
            "purpose": f"Run internal refusal, capability, paired-quality, artifact, and regression evals for {target_variant}.",
            "starts_heavy_job": False,
            "requires_execute": True,
            "expected_artifacts": ["manifest.json", "scores.csv", "eval_provenance_card.json"],
        },
        {
            "command": f"./forge compare {family}",
            "purpose": "Refresh saved comparisons against the unedited source and downloaded references.",
            "starts_heavy_job": False,
            "requires_execute": True,
            "expected_artifacts": ["comparison.json", "comparison.md"],
        },
        {
            "command": f"./forge promote {family}",
            "purpose": "Apply objective-profile promotion gates before publishing or calling the behavior edit successful.",
            "starts_heavy_job": False,
            "requires_execute": True,
            "expected_artifacts": ["promotion_report.json", "promotion_report.md"],
        },
    ]
    safety = plan.get("safety") or {}
    return {
        "schema_version": AGENT_EXPERIMENT_VERSION,
        "experiment_id": experiment_id,
        "title": args.title or f"Optimize behavior edit for {family}/{source_variant}",
        "hypothesis": args.hypothesis
        or (
            "A guarded behavior-edit workflow can reduce invalid refusals and target noncompliance "
            "while preserving the source model's normal-use, challenge, paired-benign, and artifact quality."
        ),
        "owner_agent": args.owner_agent,
        "experiment_type": "ablation",
        "status": "planned",
        "family": family,
        "variant": source_variant,
        "objective_profile": args.objective_profile,
        "planned_commands": planned_commands,
        "resource_policy": {
            "max_concurrent_large_jobs": 1,
            "start_if_memory_available_above_fraction": args.start_memory_fraction,
            "stop_if_memory_available_below_fraction": args.stop_memory_fraction,
            "require_disk_free_fraction": args.disk_free_fraction,
            "use_cluster_when_heavy": True,
            "checkpoint_or_write_plan_before_execute": True,
        },
        "evidence_plan": {
            "manifest_required": True,
            "ledger_update_required": True,
            "expected_reports": [
                "sota_runner",
                "model_forge_sota_summary.json",
                "eval_provenance_card.json",
                "comparison.json",
                "promotion_report.md",
                "manifest.json",
            ],
            "required_validation_commands": [
                f"./forge ablate --config {config_display} plan",
                f"./forge variants tokenizer-audit {family} --variant {source_variant} --json",
                f"./forge variants tokenizer-audit {family} --variant {target_variant} --json",
                "./forge doctor --json",
            ],
        },
        "success_criteria": [
            "Invalid refusal and target noncompliance metrics improve versus the unedited source checkpoint.",
            "Normal-use, challenge, paired-benign, structured-output, and artifact-quality regressions stay within the objective profile gates.",
            "Unsafe overcompliance is reported as an expected risk metric for refusal-ablated models, not used as a blanket failure label.",
            "Tokenizer, chat-template, and architecture audits pass for the edited candidate before promotion.",
            "The selected recipe, eval deltas, caveats, and publish decision are recorded in docs/experiment-ledger.md.",
        ],
        "rollback_plan": [
            "Stop active behavior-edit or serving jobs through the guarded launcher if system health degrades.",
            "Keep the unedited source variant as the serving and release fallback.",
            "Reject edits that remove refusals by destroying normal-use or challenge capability.",
            "Do not publish edited checkpoints without matching eval, comparison, and promotion evidence.",
        ],
        "handoff": {
            "push_to_github": True,
            "update_status_docs": True,
            "hf_upload_policy": "upload_completed_behavior_edited_checkpoints_only_after_release_gate",
            "raw_artifact_policy": "keep_large_outputs_untracked_or_upload_to_hf",
            "notes": "This plan does not execute behavior editing. Run one heavy edit or server at a time and attach objective evidence before promotion.",
        },
        "metadata": {
            "config": config_display,
            "method": plan.get("method"),
            "source_model": plan["model"].get("source"),
            "source_variant": source_variant,
            "target_variant": target_variant,
            "output_dir": plan["model"].get("output_dir"),
            "sota_output_dir": ((config.get("sota") or {}).get("output_dir")),
            "backend": args.backend or ((config.get("sota") or {}).get("preferred_backend")),
            "usable_pairs": (plan.get("data") or {}).get("usable_pairs"),
            "high_parallelism_c": (plan.get("activation_collection") or {}).get("high_parallelism_c"),
            "min_free_cuda_gb": safety.get("min_free_cuda_gb"),
            "review_required_before_export": bool((plan.get("edit") or {}).get("review_required_before_export", True)),
        },
    }


def render_findings(findings: list[Finding]) -> None:
    if not findings:
        print("model-forge agent audit: OK")
        return
    print(f"model-forge agent audit: {len(findings)} issue(s) found")
    for finding in findings:
        location = finding.path or "<plan>"
        if finding.field:
            location = f"{location}:{finding.field}"
        print(f"- [{finding.check}] {location}: {finding.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and validate AI-agent experiment plans")
    sub = parser.add_subparsers(dest="command", required=True)

    schema_cmd = sub.add_parser("schema", help="Print the agent experiment schema")
    schema_cmd.add_argument("--json", action="store_true")

    audit = sub.add_parser("audit", help="Validate tracked or supplied agent experiment plans")
    audit.add_argument("paths", nargs="*", type=Path)
    audit.add_argument("--json", action="store_true")

    init = sub.add_parser("init", help="Create an agent experiment plan from the template")
    init.add_argument("--experiment-id", required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--family", required=True)
    init.add_argument("--variant", default="base")
    init.add_argument("--objective-profile", required=True)
    init.add_argument("--experiment-type", default="repo_maintenance")
    init.add_argument("--hypothesis")
    init.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    init.add_argument("--output", type=Path)

    optimize_serving = sub.add_parser("optimize-serving", help="Create a serving optimization agent experiment plan")
    optimize_serving.add_argument("--family", required=True)
    optimize_serving.add_argument("--variant", default="base")
    optimize_serving.add_argument("--objective-profile", default="dgx_spark_latency_throughput")
    optimize_serving.add_argument("--sweep-config", type=Path, default=DEFAULT_SERVING_SWEEP_CONFIG)
    optimize_serving.add_argument("--cluster-config", type=Path)
    optimize_serving.add_argument("--base-url")
    optimize_serving.add_argument("--experiment-id")
    optimize_serving.add_argument("--title")
    optimize_serving.add_argument("--hypothesis")
    optimize_serving.add_argument("--owner-agent", default="codex")
    optimize_serving.add_argument("--output", type=Path)
    optimize_serving.add_argument("--json", action="store_true")

    optimize_quantization = sub.add_parser("optimize-quantization", help="Create a quantization optimization agent experiment plan")
    optimize_quantization.add_argument("--config", type=Path, default=DEFAULT_QUANTIZATION_CONFIG)
    optimize_quantization.add_argument("--family")
    optimize_quantization.add_argument("--variants", help="Comma-separated source variants from the quantization matrix")
    optimize_quantization.add_argument("--objective-profile", default="quantized_quality_retention")
    optimize_quantization.add_argument("--experiment-id")
    optimize_quantization.add_argument("--title")
    optimize_quantization.add_argument("--hypothesis")
    optimize_quantization.add_argument("--owner-agent", default="codex")
    optimize_quantization.add_argument("--output", type=Path)
    optimize_quantization.add_argument("--json", action="store_true")

    optimize_behavior = sub.add_parser("optimize-behavior-edit", help="Create a refusal-ablation behavior-edit agent experiment plan")
    optimize_behavior.add_argument("--family", help="Model family to edit; also discovers configs/abliteration/<family>_local_abli.yaml")
    optimize_behavior.add_argument("--config", type=Path, help="Abliteration config to plan from")
    optimize_behavior.add_argument("--source-variant", help="Source variant being edited; defaults from config shape")
    optimize_behavior.add_argument("--target-variant", help="Expected edited candidate variant")
    optimize_behavior.add_argument("--backend", choices=["obliteratus", "heretic"], help="SOTA backend to plan")
    optimize_behavior.add_argument("--objective-profile", default="zero_refusal_capability_retention")
    optimize_behavior.add_argument("--experiment-id")
    optimize_behavior.add_argument("--title")
    optimize_behavior.add_argument("--hypothesis")
    optimize_behavior.add_argument("--owner-agent", default="codex")
    optimize_behavior.add_argument("--start-memory-fraction", type=float, default=0.05)
    optimize_behavior.add_argument("--stop-memory-fraction", type=float, default=0.05)
    optimize_behavior.add_argument("--disk-free-fraction", type=float, default=0.15)
    optimize_behavior.add_argument("--output", type=Path)
    optimize_behavior.add_argument("--json", action="store_true")

    args = parser.parse_args()
    if args.command == "schema":
        schema = load_schema()
        if args.json:
            print(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        else:
            print(yaml.safe_dump(schema, sort_keys=False))
        return

    if args.command == "audit":
        if args.paths:
            schema = load_schema()
            findings: list[Finding] = []
            for path in args.paths:
                try:
                    plan = load_yaml(path)
                except (OSError, yaml.YAMLError, ValueError) as exc:
                    findings.append(Finding("agent_experiment", f"could not parse plan: {exc}", display_path(path)))
                    continue
                findings.extend(validate_agent_experiment(plan, schema=schema, path=display_path(path)))
        else:
            findings = audit_agent_experiments()
        if args.json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2) + "\n")
        else:
            render_findings(findings)
        raise SystemExit(1 if findings else 0)

    if args.command == "init":
        plan = init_agent_experiment(args)
        findings = validate_agent_experiment(plan, path=str(args.output) if args.output else None)
        if findings:
            render_findings(findings)
            raise SystemExit(1)
        text = yaml.safe_dump(plan, sort_keys=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)
        return

    if args.command == "optimize-serving":
        plan = optimize_serving_plan(args)
        findings = validate_agent_experiment(plan, path=str(args.output) if args.output else None)
        if findings:
            if args.json:
                print(json.dumps({"findings": [asdict(finding) for finding in findings]}, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        text = yaml.safe_dump(plan, sort_keys=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            if not args.output:
                print(text)
        return

    if args.command == "optimize-quantization":
        try:
            plan = optimize_quantization_plan(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        findings = validate_agent_experiment(plan, path=str(args.output) if args.output else None)
        if findings:
            if args.json:
                print(json.dumps({"findings": [asdict(finding) for finding in findings]}, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        text = yaml.safe_dump(plan, sort_keys=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            if not args.output:
                print(text)
        return

    if args.command == "optimize-behavior-edit":
        try:
            plan = optimize_behavior_edit_plan(args)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        findings = validate_agent_experiment(plan, path=str(args.output) if args.output else None)
        if findings:
            if args.json:
                print(json.dumps({"findings": [asdict(finding) for finding in findings]}, indent=2, sort_keys=True) + "\n")
            else:
                render_findings(findings)
            raise SystemExit(1)
        text = yaml.safe_dump(plan, sort_keys=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        else:
            if not args.output:
                print(text)


if __name__ == "__main__":
    main()
