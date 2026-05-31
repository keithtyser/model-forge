from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from model_forge.runs.manifest import REPO_DIR, display_path, redact_value


SCHEMA_VERSION = "model_forge.kernel_card.v1"


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_DIR / candidate


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {display_path(resolved)}")
    return data


def benchmark_label(summary: Mapping[str, Any]) -> str:
    labels = {
        "rmsnorm": "RMSNorm",
        "rope": "RoPE",
        "dequant": "Dequant",
        "kv-layout": "KV Layout",
    }
    return labels.get(str(summary.get("benchmark")), str(summary.get("benchmark") or "unknown"))


def card_paths(summary: Mapping[str, Any]) -> dict[str, str]:
    output_dir = resolve_repo_path(str(summary["output_dir"]))
    return {
        "json": display_path(output_dir / "kernel_card.json"),
        "markdown": display_path(output_dir / "kernel_card.md"),
    }


def result_pair(summary: Mapping[str, Any]) -> tuple[str, str, Mapping[str, Any], Mapping[str, Any]]:
    results = summary.get("results") or {}
    benchmark = summary.get("benchmark")
    if benchmark == "rope":
        return "interleaved_rope_reference", "complex_rope", results.get("interleaved_reference") or {}, results.get("complex_candidate") or {}
    if benchmark == "dequant":
        return "unpack_plus_dequant", "dequant_from_unpacked", results.get("unpack_plus_dequant") or {}, results.get("dequant_from_unpacked") or {}
    if benchmark == "kv-layout":
        return "contiguous_read", "paged_gather_read", results.get("contiguous_read") or {}, results.get("paged_gather_read") or {}
    return "torch.nn.functional.rms_norm", "manual_rmsnorm", results.get("torch_native") or {}, results.get("manual_reference") or {}


def shape_summary(summary: Mapping[str, Any]) -> str:
    params = summary.get("parameters") or {}
    benchmark = summary.get("benchmark")
    if benchmark == "dequant":
        return f"num_elements={params.get('num_elements')}, block_size={params.get('block_size')}, format={params.get('format')}"
    if benchmark == "kv-layout":
        return (
            f"batch={params.get('batch')}, seq_len={params.get('seq_len')}, heads={params.get('heads')}, "
            f"head_dim={params.get('head_dim')}, page_size={params.get('page_size')}"
        )
    if benchmark == "rope":
        return f"batch={params.get('batch')}, seq_len={params.get('seq_len')}, heads={params.get('heads')}, head_dim={params.get('head_dim')}"
    return f"batch={params.get('batch')}, seq_len={params.get('seq_len')}, hidden_size={params.get('hidden_size')}"


def serving_relevance(summary: Mapping[str, Any]) -> str:
    benchmark = summary.get("benchmark")
    if benchmark == "dequant":
        return "Packed low-bit weight/activation path; pair with quantized serving profiles before optimization claims."
    if benchmark == "kv-layout":
        return "KV-cache layout and gather/copy path; pair with serving memory and decode profiles before optimization claims."
    if benchmark == "rope":
        return "Rotary position embedding path; pair with Nsight serving profiles before optimization claims."
    return "Transformer norm path; pair with Nsight serving profiles before optimization claims."


def result_status(summary: Mapping[str, Any]) -> str:
    if summary.get("dry_run_only"):
        return "planned"
    correctness = summary.get("correctness") or {}
    if correctness.get("passed") is True:
        return "correctness_passed"
    if correctness.get("passed") is False:
        return "correctness_failed"
    return "unknown"


def build_kernel_card(
    summary: Mapping[str, Any],
    *,
    summary_path: str | Path | None = None,
    profile_summary: Mapping[str, Any] | None = None,
    profile_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    baseline_name, candidate_name, baseline, candidate = result_pair(summary)
    runtime = summary.get("runtime") or {}
    correctness = summary.get("correctness") or {}
    artifacts = card_paths(summary)
    if summary_path:
        artifacts["summary"] = display_path(resolve_repo_path(summary_path))
    if profile_summary_path:
        artifacts["profile_summary"] = display_path(resolve_repo_path(profile_summary_path))

    card = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_timestamp(),
        "kernel": {
            "name": benchmark_label(summary),
            "benchmark": summary.get("benchmark"),
            "run_id": summary.get("run_id"),
            "shape": shape_summary(summary),
            "parameters": dict(summary.get("parameters") or {}),
        },
        "research_basis": {
            "purpose": summary.get("purpose"),
            "source_notes": list(summary.get("source_notes") or []),
            "limitations": list(summary.get("limitations") or []),
        },
        "baseline": {
            "name": baseline_name,
            "metrics": dict(baseline),
        },
        "optimized_path": {
            "name": candidate_name,
            "metrics": dict(candidate),
            "status": "candidate_or_proxy",
        },
        "hardware": {
            "device": runtime.get("device"),
            "device_name": runtime.get("device_name"),
            "dtype": runtime.get("dtype") or runtime.get("output_dtype"),
            "torch_version": runtime.get("torch_version"),
            "system": runtime.get("system"),
        },
        "correctness": {
            "reference": correctness.get("reference"),
            "candidate": correctness.get("candidate"),
            "tolerance": correctness.get("tolerance"),
            "max_abs_error": correctness.get("max_abs_error"),
            "passed": correctness.get("passed"),
        },
        "microbenchmark": {
            "bytes_processed_per_call": runtime.get("bytes_processed_per_call"),
            "baseline_p50_ms": baseline.get("p50_ms"),
            "baseline_p95_ms": baseline.get("p95_ms"),
            "baseline_effective_bandwidth_gbps_p50": baseline.get("effective_bandwidth_gbps_p50"),
            "candidate_p50_ms": candidate.get("p50_ms"),
            "candidate_p95_ms": candidate.get("p95_ms"),
            "candidate_effective_bandwidth_gbps_p50": candidate.get("effective_bandwidth_gbps_p50"),
        },
        "profiler_summary": {
            "attached": profile_summary is not None,
            "source": display_path(resolve_repo_path(profile_summary_path)) if profile_summary_path else None,
            "summary": dict((profile_summary or {}).get("summary") or {}),
        },
        "roofline_estimate": {
            "status": "not_computed",
            "note": "Add achieved bandwidth versus hardware roofline after Nsight or dedicated bandwidth evidence is attached.",
        },
        "end_to_end_serving_relevance": serving_relevance(summary),
        "result": result_status(summary),
        "next_action": "Attach profiler and serving evidence before making end-to-end performance claims.",
        "artifacts": artifacts,
    }
    return redact_value(card)


def render_kernel_card_markdown(card: Mapping[str, Any]) -> str:
    kernel = card.get("kernel") or {}
    research = card.get("research_basis") or {}
    baseline = card.get("baseline") or {}
    optimized = card.get("optimized_path") or {}
    hardware = card.get("hardware") or {}
    correctness = card.get("correctness") or {}
    micro = card.get("microbenchmark") or {}
    profiler = card.get("profiler_summary") or {}
    roofline = card.get("roofline_estimate") or {}
    artifacts = card.get("artifacts") or {}
    lines = [
        f"# Kernel Card: {kernel.get('run_id')}",
        "",
        "## Kernel",
        "",
        f"- Kernel: {kernel.get('name')}",
        f"- Shape: {kernel.get('shape')}",
        f"- Result: `{card.get('result')}`",
        "",
        "## Research Basis",
        "",
        f"- Purpose: {research.get('purpose')}",
    ]
    lines.extend(f"- Source note: {note}" for note in research.get("source_notes") or [])
    lines.extend(f"- Limitation: {note}" for note in research.get("limitations") or [])
    lines.extend(
        [
            "",
            "## Baseline",
            "",
            f"- Name: `{baseline.get('name')}`",
            f"- p50 ms: {(baseline.get('metrics') or {}).get('p50_ms')}",
            f"- p95 ms: {(baseline.get('metrics') or {}).get('p95_ms')}",
            f"- Effective GB/s p50: {(baseline.get('metrics') or {}).get('effective_bandwidth_gbps_p50')}",
            "",
            "## Optimized Path",
            "",
            f"- Name: `{optimized.get('name')}`",
            f"- Status: `{optimized.get('status')}`",
            f"- p50 ms: {(optimized.get('metrics') or {}).get('p50_ms')}",
            f"- p95 ms: {(optimized.get('metrics') or {}).get('p95_ms')}",
            f"- Effective GB/s p50: {(optimized.get('metrics') or {}).get('effective_bandwidth_gbps_p50')}",
            "",
            "## Hardware",
            "",
            f"- Device: {hardware.get('device')}",
            f"- Device name: {hardware.get('device_name')}",
            f"- DType: {hardware.get('dtype')}",
            f"- Torch: {hardware.get('torch_version')}",
            "",
            "## Correctness",
            "",
            f"- Reference: `{correctness.get('reference')}`",
            f"- Candidate: `{correctness.get('candidate')}`",
            f"- Tolerance: {correctness.get('tolerance')}",
            f"- Max abs error: {correctness.get('max_abs_error')}",
            f"- Passed: {correctness.get('passed')}",
            "",
            "## Microbenchmark",
            "",
            f"- Bytes processed per call: {micro.get('bytes_processed_per_call')}",
            f"- Baseline p50 ms: {micro.get('baseline_p50_ms')}",
            f"- Candidate p50 ms: {micro.get('candidate_p50_ms')}",
            "",
            "## Profiler Summary",
            "",
            f"- Attached: {profiler.get('attached')}",
            f"- Source: `{profiler.get('source')}`",
            f"- Summary: `{json.dumps(profiler.get('summary') or {}, sort_keys=True)}`",
            "",
            "## Roofline Estimate",
            "",
            f"- Status: `{roofline.get('status')}`",
            f"- Note: {roofline.get('note')}",
            "",
            "## End-To-End Serving Relevance",
            "",
            f"- {card.get('end_to_end_serving_relevance')}",
            "",
            "## Next Action",
            "",
            f"- {card.get('next_action')}",
            "",
            "## Artifacts",
            "",
        ]
    )
    lines.extend(f"- {key}: `{value}`" for key, value in artifacts.items())
    lines.append("")
    return "\n".join(lines)


def write_kernel_card(card: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = resolve_repo_path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "kernel_card.json"
    md_path = root / "kernel_card.md"
    json_path.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_kernel_card_markdown(card), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
