from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from rich.console import Console
from rich.table import Table

from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id, system_snapshot


SCHEMA_VERSION = "model_forge.kernel_benchmark.v1"
DEFAULT_OUTPUT_ROOT = REPO_DIR / "reports" / "generated" / "kernel_benchmarks"

console = Console(stderr=True)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(str(path)).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_DIR / candidate


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def timing_summary(times_ms: list[float]) -> dict[str, Any]:
    return {
        "count": len(times_ms),
        "min_ms": min(times_ms) if times_ms else None,
        "p50_ms": percentile(times_ms, 0.50),
        "p95_ms": percentile(times_ms, 0.95),
        "max_ms": max(times_ms) if times_ms else None,
    }


def rmsnorm_plan(args: argparse.Namespace) -> dict[str, Any]:
    run_id = sanitize_run_id(args.run_id or f"rmsnorm_{args.device}_{args.dtype}_b{args.batch}_s{args.seq_len}_h{args.hidden_size}")
    output_dir = resolve_repo_path(args.output_dir or DEFAULT_OUTPUT_ROOT / run_id)
    return redact_value(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "benchmark": "rmsnorm",
            "run_id": run_id,
            "output_dir": display_path(output_dir),
            "parameters": {
                "batch": args.batch,
                "seq_len": args.seq_len,
                "hidden_size": args.hidden_size,
                "eps": args.eps,
                "dtype": args.dtype,
                "device": args.device,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
            },
            "purpose": "Measure RMSNorm latency and bandwidth for transformer inference kernel work.",
            "limitations": [
                "This is a microbenchmark; promotion requires end-to-end serving evidence.",
                "The initial implementation compares Torch/native and manual reference paths.",
            ],
            "outputs": {
                "summary": display_path(output_dir / "summary.json"),
                "card": display_path(output_dir / "kernel_card.md"),
            },
        }
    )


def rope_plan(args: argparse.Namespace) -> dict[str, Any]:
    run_id = sanitize_run_id(args.run_id or f"rope_{args.device}_{args.dtype}_b{args.batch}_s{args.seq_len}_h{args.heads}_d{args.head_dim}")
    output_dir = resolve_repo_path(args.output_dir or DEFAULT_OUTPUT_ROOT / run_id)
    return redact_value(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "benchmark": "rope",
            "run_id": run_id,
            "output_dir": display_path(output_dir),
            "parameters": {
                "batch": args.batch,
                "seq_len": args.seq_len,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "theta": args.theta,
                "dtype": args.dtype,
                "device": args.device,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
            },
            "purpose": "Measure rotary position embedding latency and bandwidth for transformer inference kernel work.",
            "limitations": [
                "This is a microbenchmark; promotion requires end-to-end serving evidence.",
                "The initial implementation compares an interleaved Torch reference with a complex-number Torch candidate.",
            ],
            "outputs": {
                "summary": display_path(output_dir / "summary.json"),
                "card": display_path(output_dir / "kernel_card.md"),
            },
        }
    )


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for non-dry-run kernel benchmarks. Install the finetuning or abliteration extras.") from exc
    return torch


def torch_dtype(torch: Any, name: str) -> Any:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in aliases:
        raise ValueError(f"unsupported dtype: {name}")
    return aliases[name]


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device but torch.cuda.is_available() is false")
    return requested


def rmsnorm_manual(torch: Any, x: Any, weight: Any, eps: float) -> Any:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rmsnorm_native(torch: Any, x: Any, weight: Any, eps: float) -> Any:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def build_rope_angles(torch: Any, seq_len: int, head_dim: int, theta: float, *, device: str) -> Any:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    return torch.outer(positions, inv_freq)


def rope_interleaved_reference(torch: Any, x: Any, angles: Any) -> Any:
    cos = angles.cos()[None, :, None, :]
    sin = angles.sin()[None, :, None, :]
    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    out = torch.empty_like(x.float())
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out.to(dtype=x.dtype)


def rope_complex_candidate(torch: Any, x: Any, angles: Any) -> Any:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
    rotation = torch.polar(torch.ones_like(angles), angles)[None, :, None, :]
    return torch.view_as_real(x_complex * rotation).flatten(-2).to(dtype=x.dtype)


def synchronize(torch: Any, device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_impl(torch: Any, fn: Callable[[], Any], *, device: str, warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    synchronize(torch, device)
    times_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        synchronize(torch, device)
        times_ms.append((time.perf_counter() - started) * 1000.0)
    return times_ms


def effective_bandwidth_gbps(bytes_processed: int, latency_ms: float | None) -> float | None:
    if latency_ms is None or latency_ms <= 0:
        return None
    return bytes_processed / (latency_ms / 1000.0) / 1_000_000_000


def run_rmsnorm(plan: Mapping[str, Any]) -> dict[str, Any]:
    torch = import_torch()
    params = dict(plan["parameters"])
    device = resolve_device(torch, str(params["device"]))
    dtype = torch_dtype(torch, str(params["dtype"]))
    torch.manual_seed(int(params["seed"]))
    shape = (int(params["batch"]), int(params["seq_len"]), int(params["hidden_size"]))
    x = torch.randn(shape, device=device, dtype=dtype)
    weight = torch.randn((shape[-1],), device=device, dtype=dtype)
    eps = float(params["eps"])

    with torch.no_grad():
        native = rmsnorm_native(torch, x, weight, eps)
        manual = rmsnorm_manual(torch, x, weight, eps)
        max_abs_error = float((native.float() - manual.float()).abs().max().item())
        native_times = benchmark_impl(
            torch,
            lambda: rmsnorm_native(torch, x, weight, eps),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )
        manual_times = benchmark_impl(
            torch,
            lambda: rmsnorm_manual(torch, x, weight, eps),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )

    element_size = x.element_size()
    # Approximate read input + read weight + write output.
    bytes_processed = (x.numel() * 2 + weight.numel()) * element_size
    native_summary = timing_summary(native_times)
    manual_summary = timing_summary(manual_times)
    return redact_value(
        {
            **dict(plan),
            "created_at": utc_timestamp(),
            "dry_run_only": False,
            "runtime": {
                "torch_version": torch.__version__,
                "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "dtype": str(dtype).replace("torch.", ""),
                "shape": list(shape),
                "bytes_processed_per_call": bytes_processed,
                "system": system_snapshot(resolve_repo_path(str(plan["output_dir"]))),
            },
            "correctness": {
                "reference": "torch.nn.functional.rms_norm",
                "candidate": "manual_rmsnorm",
                "max_abs_error": max_abs_error,
                "tolerance": 1e-3 if str(params["dtype"]) in {"float16", "fp16", "bfloat16", "bf16"} else 1e-5,
                "passed": max_abs_error <= (1e-3 if str(params["dtype"]) in {"float16", "fp16", "bfloat16", "bf16"} else 1e-5),
            },
            "results": {
                "torch_native": {
                    **native_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, native_summary["p50_ms"]),
                },
                "manual_reference": {
                    **manual_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, manual_summary["p50_ms"]),
                },
            },
        }
    )


def run_rope(plan: Mapping[str, Any]) -> dict[str, Any]:
    torch = import_torch()
    params = dict(plan["parameters"])
    device = resolve_device(torch, str(params["device"]))
    dtype = torch_dtype(torch, str(params["dtype"]))
    torch.manual_seed(int(params["seed"]))
    head_dim = int(params["head_dim"])
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for interleaved RoPE")
    shape = (int(params["batch"]), int(params["seq_len"]), int(params["heads"]), head_dim)
    x = torch.randn(shape, device=device, dtype=dtype)
    angles = build_rope_angles(torch, int(params["seq_len"]), head_dim, float(params["theta"]), device=device)

    with torch.no_grad():
        reference = rope_interleaved_reference(torch, x, angles)
        candidate = rope_complex_candidate(torch, x, angles)
        max_abs_error = float((reference.float() - candidate.float()).abs().max().item())
        reference_times = benchmark_impl(
            torch,
            lambda: rope_interleaved_reference(torch, x, angles),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )
        candidate_times = benchmark_impl(
            torch,
            lambda: rope_complex_candidate(torch, x, angles),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )

    element_size = x.element_size()
    bytes_processed = x.numel() * element_size * 2
    reference_summary = timing_summary(reference_times)
    candidate_summary = timing_summary(candidate_times)
    tolerance = 2e-3 if str(params["dtype"]) in {"float16", "fp16", "bfloat16", "bf16"} else 1e-5
    return redact_value(
        {
            **dict(plan),
            "created_at": utc_timestamp(),
            "dry_run_only": False,
            "runtime": {
                "torch_version": torch.__version__,
                "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "dtype": str(dtype).replace("torch.", ""),
                "shape": list(shape),
                "bytes_processed_per_call": bytes_processed,
                "system": system_snapshot(resolve_repo_path(str(plan["output_dir"]))),
            },
            "correctness": {
                "reference": "interleaved_rope_reference",
                "candidate": "complex_rope",
                "max_abs_error": max_abs_error,
                "tolerance": tolerance,
                "passed": max_abs_error <= tolerance,
            },
            "results": {
                "interleaved_reference": {
                    **reference_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, reference_summary["p50_ms"]),
                },
                "complex_candidate": {
                    **candidate_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, candidate_summary["p50_ms"]),
                },
            },
        }
    )


def render_card(summary: Mapping[str, Any]) -> str:
    params = summary.get("parameters") or {}
    runtime = summary.get("runtime") or {}
    correctness = summary.get("correctness") or {}
    results = summary.get("results") or {}
    if summary.get("benchmark") == "rope":
        benchmark_label = "RoPE"
        baseline_name = "interleaved_rope_reference"
        candidate_name = "complex_rope"
        baseline = results.get("interleaved_reference") or {}
        candidate = results.get("complex_candidate") or {}
        scope = "rotary position embedding path; use alongside Nsight serving profiles before optimization claims."
        shape = f"batch={params.get('batch')}, seq_len={params.get('seq_len')}, heads={params.get('heads')}, head_dim={params.get('head_dim')}"
    else:
        benchmark_label = "RMSNorm"
        baseline_name = "torch.nn.functional.rms_norm"
        candidate_name = "manual_rmsnorm"
        baseline = results.get("torch_native") or {}
        candidate = results.get("manual_reference") or {}
        scope = "transformer norm path; use alongside Nsight serving profiles before optimization claims."
        shape = f"batch={params.get('batch')}, seq_len={params.get('seq_len')}, hidden_size={params.get('hidden_size')}"
    lines = [
        f"# Kernel Card: {summary.get('run_id')}",
        "",
        "## Scope",
        "",
        f"- Benchmark: {benchmark_label}",
        f"- End-to-end relevance: {scope}",
        f"- Baseline: `{baseline_name}`",
        f"- Candidate: `{candidate_name}`",
        "",
        "## Parameters",
        "",
        f"- Shape: {shape}",
        f"- Device: {runtime.get('device', params.get('device'))}",
        f"- DType: {runtime.get('dtype', params.get('dtype'))}",
        f"- Repeats: warmup={params.get('warmup')}, measured={params.get('repeats')}",
        "",
        "## Correctness",
        "",
        f"- Passed: {correctness.get('passed')}",
        f"- Max abs error: {correctness.get('max_abs_error')}",
        f"- Tolerance: {correctness.get('tolerance')}",
        "",
        "## Microbenchmark",
        "",
        f"- Baseline p50 ms: {baseline.get('p50_ms')}",
        f"- Baseline p95 ms: {baseline.get('p95_ms')}",
        f"- Baseline effective GB/s p50: {baseline.get('effective_bandwidth_gbps_p50')}",
        f"- Candidate p50 ms: {candidate.get('p50_ms')}",
        f"- Candidate p95 ms: {candidate.get('p95_ms')}",
        f"- Candidate effective GB/s p50: {candidate.get('effective_bandwidth_gbps_p50')}",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in summary.get("limitations") or [])
    lines.append("")
    return "\n".join(lines)


def write_outputs(summary: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(summary["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "kernel_card.md").write_text(render_card(summary), encoding="utf-8")
    return summary_path


def render_table(summary: Mapping[str, Any]) -> None:
    table = Table(title=f"Kernel Benchmark: {summary.get('run_id')}")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("benchmark", str(summary.get("benchmark")))
    table.add_row("dry run", str(summary.get("dry_run_only", False)))
    table.add_row("output", str(summary.get("output_dir")))
    results = summary.get("results") or {}
    if results:
        if summary.get("benchmark") == "rope":
            baseline = results.get("interleaved_reference") or {}
        else:
            baseline = results.get("torch_native") or {}
        table.add_row("baseline p50 ms", str(baseline.get("p50_ms")))
        table.add_row("baseline GB/s p50", str(baseline.get("effective_bandwidth_gbps_p50")))
        table.add_row("correctness", str((summary.get("correctness") or {}).get("passed")))
    console.print(table)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kernel microbenchmarks for model-forge")
    sub = parser.add_subparsers(dest="benchmark", required=True)
    rms = sub.add_parser("rmsnorm", help="Run or plan an RMSNorm microbenchmark")
    rms.add_argument("--batch", type=int, default=1)
    rms.add_argument("--seq-len", type=int, default=1024)
    rms.add_argument("--hidden-size", type=int, default=4096)
    rms.add_argument("--eps", type=float, default=1e-6)
    rms.add_argument("--dtype", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], default="bfloat16")
    rms.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    rms.add_argument("--warmup", type=int, default=5)
    rms.add_argument("--repeats", type=int, default=20)
    rms.add_argument("--seed", type=int, default=17)
    rms.add_argument("--run-id")
    rms.add_argument("--output-dir", type=Path)
    rms.add_argument("--dry-run", action="store_true")
    rms.add_argument("--write", action="store_true")
    rms.add_argument("--json", action="store_true")
    rope = sub.add_parser("rope", help="Run or plan a RoPE microbenchmark")
    rope.add_argument("--batch", type=int, default=1)
    rope.add_argument("--seq-len", type=int, default=1024)
    rope.add_argument("--heads", type=int, default=16)
    rope.add_argument("--head-dim", type=int, default=128)
    rope.add_argument("--theta", type=float, default=10_000.0)
    rope.add_argument("--dtype", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], default="bfloat16")
    rope.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    rope.add_argument("--warmup", type=int, default=5)
    rope.add_argument("--repeats", type=int, default=20)
    rope.add_argument("--seed", type=int, default=17)
    rope.add_argument("--run-id")
    rope.add_argument("--output-dir", type=Path)
    rope.add_argument("--dry-run", action="store_true")
    rope.add_argument("--write", action="store_true")
    rope.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.benchmark == "rmsnorm":
        summary = rmsnorm_plan(args)
    elif args.benchmark == "rope":
        if args.head_dim % 2 != 0:
            raise SystemExit("--head-dim must be even")
        summary = rope_plan(args)
    else:
        raise SystemExit(f"unknown kernel benchmark: {args.benchmark}")
    if args.dry_run:
        summary = {**summary, "dry_run_only": True}
    elif args.benchmark == "rmsnorm":
        summary = run_rmsnorm(summary)
    elif args.benchmark == "rope":
        summary = run_rope(summary)
    if args.write:
        write_outputs(summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        render_table(summary)


if __name__ == "__main__":
    main()
