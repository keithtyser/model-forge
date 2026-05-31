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
from model_forge.reports.kernel_card import build_kernel_card, load_json, render_kernel_card_markdown, write_kernel_card


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
                "card_json": display_path(output_dir / "kernel_card.json"),
                "card_markdown": display_path(output_dir / "kernel_card.md"),
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
                "card_json": display_path(output_dir / "kernel_card.json"),
                "card_markdown": display_path(output_dir / "kernel_card.md"),
            },
        }
    )


def dequant_plan(args: argparse.Namespace) -> dict[str, Any]:
    run_id = sanitize_run_id(args.run_id or f"dequant_{args.format}_{args.device}_{args.output_dtype}_n{args.num_elements}_b{args.block_size}")
    output_dir = resolve_repo_path(args.output_dir or DEFAULT_OUTPUT_ROOT / run_id)
    return redact_value(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "benchmark": "dequant",
            "run_id": run_id,
            "output_dir": display_path(output_dir),
            "parameters": {
                "format": args.format,
                "num_elements": args.num_elements,
                "block_size": args.block_size,
                "output_dtype": args.output_dtype,
                "device": args.device,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
            },
            "purpose": "Measure packed 4-bit dequantization latency and bandwidth for quantized inference kernel work.",
            "limitations": [
                "This is a Torch microbenchmark/proxy, not a native Blackwell Tensor Core NVFP4 path.",
                "NVFP4 scale encoding is represented as FP32 scales for benchmark portability.",
                "Promotion requires real quantized serving evidence.",
            ],
            "source_notes": [
                "NVIDIA documents NVFP4 as E2M1 values with local E4M3 scale per 16 values and a global FP32 scale.",
            ],
            "outputs": {
                "summary": display_path(output_dir / "summary.json"),
                "card": display_path(output_dir / "kernel_card.md"),
                "card_json": display_path(output_dir / "kernel_card.json"),
                "card_markdown": display_path(output_dir / "kernel_card.md"),
            },
        }
    )


def kv_layout_plan(args: argparse.Namespace) -> dict[str, Any]:
    run_id = sanitize_run_id(
        args.run_id
        or f"kv_layout_{args.device}_{args.dtype}_b{args.batch}_s{args.seq_len}_h{args.heads}_d{args.head_dim}_p{args.page_size}"
    )
    output_dir = resolve_repo_path(args.output_dir or DEFAULT_OUTPUT_ROOT / run_id)
    return redact_value(
        {
            "schema_version": SCHEMA_VERSION,
            "created_at": utc_timestamp(),
            "benchmark": "kv-layout",
            "run_id": run_id,
            "output_dir": display_path(output_dir),
            "parameters": {
                "batch": args.batch,
                "seq_len": args.seq_len,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "page_size": args.page_size,
                "dtype": args.dtype,
                "device": args.device,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "seed": args.seed,
            },
            "purpose": "Measure contiguous versus paged/gathered KV-cache layout access for decode-path memory work.",
            "limitations": [
                "This is a Torch microbenchmark/proxy, not a vLLM PagedAttention kernel.",
                "Promotion requires backend-specific serving profiles and memory evidence.",
            ],
            "outputs": {
                "summary": display_path(output_dir / "summary.json"),
                "card": display_path(output_dir / "kernel_card.md"),
                "card_json": display_path(output_dir / "kernel_card.json"),
                "card_markdown": display_path(output_dir / "kernel_card.md"),
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


def nvfp4_e2m1_codebook(torch: Any, *, device: str) -> Any:
    # Positive E2M1 magnitudes documented for NVFP4 are approximately
    # 0, 0.5, 1, 1.5, 2, 3, 4, and 6, mirrored by the sign bit.
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    return torch.tensor(values, device=device, dtype=torch.float32)


def unpack_low_high_nibbles(torch: Any, packed: Any, total_values: int) -> Any:
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    stacked = torch.stack((low, high), dim=-1).flatten()
    return stacked[:total_values].to(dtype=torch.long)


def dequant_from_unpacked(torch: Any, codes: Any, scales: Any, global_scale: float, codebook: Any, block_size: int, output_dtype: Any) -> Any:
    values = codebook[codes]
    block_ids = torch.div(torch.arange(codes.numel(), device=codes.device), block_size, rounding_mode="floor")
    return (values * scales[block_ids] * global_scale).to(dtype=output_dtype)


def dequant_unpack_plus_lut(torch: Any, packed: Any, scales: Any, global_scale: float, codebook: Any, block_size: int, total_values: int, output_dtype: Any) -> Any:
    codes = unpack_low_high_nibbles(torch, packed, total_values)
    return dequant_from_unpacked(torch, codes, scales, global_scale, codebook, block_size, output_dtype)


def dequant_python_reference(raw_codes: list[int], raw_scales: list[float], global_scale: float, block_size: int) -> list[float]:
    codebook = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    return [codebook[int(code)] * raw_scales[index // block_size] * global_scale for index, code in enumerate(raw_codes)]


def build_paged_kv(torch: Any, contiguous: Any, page_size: int, permutation: Any) -> tuple[Any, Any]:
    batch, seq_len, heads, head_dim = contiguous.shape
    page_count = (seq_len + page_size - 1) // page_size
    padded_len = page_count * page_size
    if padded_len != seq_len:
        pad = torch.zeros((batch, padded_len - seq_len, heads, head_dim), device=contiguous.device, dtype=contiguous.dtype)
        contiguous = torch.cat((contiguous, pad), dim=1)
    logical_pages = contiguous.reshape(batch, page_count, page_size, heads, head_dim)
    physical_pages = logical_pages[:, permutation].contiguous()
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(page_count, device=permutation.device)
    return physical_pages, inverse


def kv_contiguous_read(torch: Any, key: Any, value: Any, seq_len: int) -> Any:
    return key[:, :seq_len].float().sum(dim=(1, 2, 3)) + value[:, :seq_len].float().sum(dim=(1, 2, 3))


def kv_paged_gather_read(torch: Any, key_pages: Any, value_pages: Any, page_table: Any, seq_len: int) -> Any:
    gathered_key = key_pages.index_select(1, page_table).flatten(1, 2)[:, :seq_len]
    gathered_value = value_pages.index_select(1, page_table).flatten(1, 2)[:, :seq_len]
    return gathered_key.float().sum(dim=(1, 2, 3)) + gathered_value.float().sum(dim=(1, 2, 3))


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


def run_dequant(plan: Mapping[str, Any]) -> dict[str, Any]:
    torch = import_torch()
    params = dict(plan["parameters"])
    if params["format"] != "nvfp4-e2m1":
        raise ValueError(f"unsupported dequant format: {params['format']}")
    device = resolve_device(torch, str(params["device"]))
    output_dtype = torch_dtype(torch, str(params["output_dtype"]))
    torch.manual_seed(int(params["seed"]))
    total_values = int(params["num_elements"])
    block_size = int(params["block_size"])
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    packed_count = (total_values + 1) // 2
    block_count = (total_values + block_size - 1) // block_size
    low = torch.randint(0, 16, (packed_count,), device=device, dtype=torch.uint8)
    high = torch.randint(0, 16, (packed_count,), device=device, dtype=torch.uint8)
    packed = low | (high << 4)
    scales = torch.rand((block_count,), device=device, dtype=torch.float32) + 0.5
    global_scale = 0.75
    codebook = nvfp4_e2m1_codebook(torch, device=device)
    unpacked = unpack_low_high_nibbles(torch, packed, total_values)

    with torch.no_grad():
        sample_count = min(total_values, 256)
        sample_codes = unpacked[:sample_count].detach().cpu().tolist()
        sample_scales = scales[: (sample_count + block_size - 1) // block_size].detach().cpu().tolist()
        reference = torch.tensor(
            dequant_python_reference(sample_codes, sample_scales, global_scale, block_size),
            device=device,
            dtype=torch.float32,
        )
        candidate_sample = dequant_from_unpacked(torch, unpacked[:sample_count], scales, global_scale, codebook, block_size, torch.float32)
        max_abs_error = float((reference - candidate_sample).abs().max().item())
        unpack_times = benchmark_impl(
            torch,
            lambda: dequant_unpack_plus_lut(torch, packed, scales, global_scale, codebook, block_size, total_values, output_dtype),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )
        dequant_only_times = benchmark_impl(
            torch,
            lambda: dequant_from_unpacked(torch, unpacked, scales, global_scale, codebook, block_size, output_dtype),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )

    output_bytes = total_values * torch.tensor([], dtype=output_dtype).element_size()
    input_bytes = packed.numel() * packed.element_size() + scales.numel() * scales.element_size()
    bytes_processed = input_bytes + output_bytes
    unpack_summary = timing_summary(unpack_times)
    dequant_summary = timing_summary(dequant_only_times)
    return redact_value(
        {
            **dict(plan),
            "created_at": utc_timestamp(),
            "dry_run_only": False,
            "runtime": {
                "torch_version": torch.__version__,
                "device": device,
                "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
                "output_dtype": str(output_dtype).replace("torch.", ""),
                "packed_values": int(packed.numel()),
                "block_count": block_count,
                "bytes_processed_per_call": bytes_processed,
                "system": system_snapshot(resolve_repo_path(str(plan["output_dir"]))),
            },
            "correctness": {
                "reference": "python_nvfp4_e2m1_sample",
                "candidate": "torch_vectorized_lut",
                "sample_values": sample_count,
                "max_abs_error": max_abs_error,
                "tolerance": 1e-6,
                "passed": max_abs_error <= 1e-6,
            },
            "results": {
                "unpack_plus_dequant": {
                    **unpack_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, unpack_summary["p50_ms"]),
                },
                "dequant_from_unpacked": {
                    **dequant_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, dequant_summary["p50_ms"]),
                },
            },
        }
    )


def run_kv_layout(plan: Mapping[str, Any]) -> dict[str, Any]:
    torch = import_torch()
    params = dict(plan["parameters"])
    device = resolve_device(torch, str(params["device"]))
    dtype = torch_dtype(torch, str(params["dtype"]))
    torch.manual_seed(int(params["seed"]))
    batch = int(params["batch"])
    seq_len = int(params["seq_len"])
    heads = int(params["heads"])
    head_dim = int(params["head_dim"])
    page_size = int(params["page_size"])
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    shape = (batch, seq_len, heads, head_dim)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    page_count = (seq_len + page_size - 1) // page_size
    permutation = torch.randperm(page_count, device=device)
    key_pages, page_table = build_paged_kv(torch, key, page_size, permutation)
    value_pages, _ = build_paged_kv(torch, value, page_size, permutation)

    with torch.no_grad():
        contiguous_output = kv_contiguous_read(torch, key, value, seq_len)
        paged_output = kv_paged_gather_read(torch, key_pages, value_pages, page_table, seq_len)
        max_abs_error = float((contiguous_output.float() - paged_output.float()).abs().max().item())
        contiguous_times = benchmark_impl(
            torch,
            lambda: kv_contiguous_read(torch, key, value, seq_len),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )
        paged_times = benchmark_impl(
            torch,
            lambda: kv_paged_gather_read(torch, key_pages, value_pages, page_table, seq_len),
            device=device,
            warmup=int(params["warmup"]),
            repeats=int(params["repeats"]),
        )

    element_size = key.element_size()
    bytes_processed = key.numel() * element_size * 2
    contiguous_summary = timing_summary(contiguous_times)
    paged_summary = timing_summary(paged_times)
    tolerance = 1e-2 if str(params["dtype"]) in {"float16", "fp16", "bfloat16", "bf16"} else 1e-4
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
                "page_count": page_count,
                "bytes_processed_per_call": bytes_processed,
                "system": system_snapshot(resolve_repo_path(str(plan["output_dir"]))),
            },
            "correctness": {
                "reference": "contiguous_kv_read",
                "candidate": "paged_gather_kv_read",
                "max_abs_error": max_abs_error,
                "tolerance": tolerance,
                "passed": max_abs_error <= tolerance,
            },
            "results": {
                "contiguous_read": {
                    **contiguous_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, contiguous_summary["p50_ms"]),
                },
                "paged_gather_read": {
                    **paged_summary,
                    "effective_bandwidth_gbps_p50": effective_bandwidth_gbps(bytes_processed, paged_summary["p50_ms"]),
                },
            },
        }
    )


def render_card(summary: Mapping[str, Any]) -> str:
    return render_kernel_card_markdown(build_kernel_card(summary))


def write_outputs(summary: Mapping[str, Any]) -> Path:
    output_dir = resolve_repo_path(str(summary["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    card = build_kernel_card(summary, summary_path=summary_path)
    write_kernel_card(card, output_dir)
    return summary_path


def build_card_from_summary(summary_path: str | Path, profile_summary_path: str | Path | None = None) -> dict[str, Any]:
    summary_resolved = resolve_repo_path(summary_path)
    summary = load_json(summary_resolved)
    profile_summary = load_json(profile_summary_path) if profile_summary_path else None
    return build_kernel_card(
        summary,
        summary_path=summary_resolved,
        profile_summary=profile_summary,
        profile_summary_path=profile_summary_path,
    )


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
        elif summary.get("benchmark") == "dequant":
            baseline = results.get("unpack_plus_dequant") or {}
        elif summary.get("benchmark") == "kv-layout":
            baseline = results.get("contiguous_read") or {}
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
    dequant = sub.add_parser("dequant", help="Run or plan a packed 4-bit dequantization microbenchmark")
    dequant.add_argument("--format", choices=["nvfp4-e2m1"], default="nvfp4-e2m1")
    dequant.add_argument("--num-elements", type=int, default=1_048_576)
    dequant.add_argument("--block-size", type=int, default=16)
    dequant.add_argument("--output-dtype", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], default="bfloat16")
    dequant.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    dequant.add_argument("--warmup", type=int, default=5)
    dequant.add_argument("--repeats", type=int, default=20)
    dequant.add_argument("--seed", type=int, default=17)
    dequant.add_argument("--run-id")
    dequant.add_argument("--output-dir", type=Path)
    dequant.add_argument("--dry-run", action="store_true")
    dequant.add_argument("--write", action="store_true")
    dequant.add_argument("--json", action="store_true")
    kv = sub.add_parser("kv-layout", help="Run or plan a KV-cache layout/copy microbenchmark")
    kv.add_argument("--batch", type=int, default=1)
    kv.add_argument("--seq-len", type=int, default=4096)
    kv.add_argument("--heads", type=int, default=16)
    kv.add_argument("--head-dim", type=int, default=128)
    kv.add_argument("--page-size", type=int, default=16)
    kv.add_argument("--dtype", choices=["float32", "fp32", "float16", "fp16", "bfloat16", "bf16"], default="bfloat16")
    kv.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    kv.add_argument("--warmup", type=int, default=5)
    kv.add_argument("--repeats", type=int, default=20)
    kv.add_argument("--seed", type=int, default=17)
    kv.add_argument("--run-id")
    kv.add_argument("--output-dir", type=Path)
    kv.add_argument("--dry-run", action="store_true")
    kv.add_argument("--write", action="store_true")
    kv.add_argument("--json", action="store_true")
    card = sub.add_parser("card", help="Generate a Kernel Card from a kernel benchmark summary")
    card.add_argument("--summary", type=Path, required=True)
    card.add_argument("--profile-summary", type=Path)
    card.add_argument("--output-dir", type=Path)
    card.add_argument("--write-card", action="store_true")
    card.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.benchmark == "card":
        card = build_card_from_summary(args.summary, args.profile_summary)
        if args.write_card:
            output_dir = resolve_repo_path(args.output_dir or str(load_json(args.summary)["output_dir"]))
            write_kernel_card(card, output_dir)
        if args.json:
            print(json.dumps(card, indent=2, sort_keys=True) + "\n")
        else:
            print(render_kernel_card_markdown(card))
        return

    if args.benchmark == "rmsnorm":
        summary = rmsnorm_plan(args)
    elif args.benchmark == "rope":
        if args.head_dim % 2 != 0:
            raise SystemExit("--head-dim must be even")
        summary = rope_plan(args)
    elif args.benchmark == "dequant":
        if args.num_elements <= 0:
            raise SystemExit("--num-elements must be positive")
        if args.block_size <= 0:
            raise SystemExit("--block-size must be positive")
        summary = dequant_plan(args)
    elif args.benchmark == "kv-layout":
        if args.seq_len <= 0:
            raise SystemExit("--seq-len must be positive")
        if args.page_size <= 0:
            raise SystemExit("--page-size must be positive")
        summary = kv_layout_plan(args)
    else:
        raise SystemExit(f"unknown kernel benchmark: {args.benchmark}")
    if args.dry_run:
        summary = {**summary, "dry_run_only": True}
    elif args.benchmark == "rmsnorm":
        summary = run_rmsnorm(summary)
    elif args.benchmark == "rope":
        summary = run_rope(summary)
    elif args.benchmark == "dequant":
        summary = run_dequant(summary)
    elif args.benchmark == "kv-layout":
        summary = run_kv_layout(summary)
    if args.write:
        write_outputs(summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    else:
        render_table(summary)


if __name__ == "__main__":
    main()
