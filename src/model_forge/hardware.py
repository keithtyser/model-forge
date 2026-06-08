from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field, replace
from multiprocessing import cpu_count
from typing import Mapping


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total_mb: int
    compute_capability: str = ""


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    label: str
    gpus: tuple[GpuInfo, ...] = ()
    vllm_env: Mapping[str, str] = field(default_factory=dict)
    training_env: Mapping[str, str] = field(default_factory=dict)
    quantization_env: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
    compute_capability: str = ""
    supported_quant: tuple[str, ...] = ()


def arch_from_cc(cc: str) -> str:
    """Map an NVIDIA compute capability (e.g. '8.6', '9.0', '12.1') to an arch class."""
    cc = (cc or "").strip()
    try:
        major, minor = (int(x) for x in cc.split(".")[:2])
    except (ValueError, TypeError):
        return "unknown"
    if major >= 10:
        return "blackwell"          # sm_100 (B100/B200), sm_120/121 (RTX 50xx / GB10)
    if (major, minor) == (9, 0):
        return "hopper"             # H100 / H200
    if (major, minor) == (8, 9):
        return "ada"                # RTX 40xx / L40
    if major == 8:
        return "ampere"             # A100 / A6000 / RTX 30xx
    if major == 7:
        return "volta_turing"       # V100 / T4 / RTX 20xx
    return "unknown"


def supported_quant_for_cc(cc: str) -> tuple[str, ...]:
    """Quantization methods this compute capability can actually run, best-first. NVFP4 needs
    Blackwell FP4 tensor cores; FP8 needs Hopper/Ada (sm_89+); INT8/AWQ/GPTQ work broadly."""
    arch = arch_from_cc(cc)
    if arch == "blackwell":
        return ("nvfp4", "fp8", "int8", "awq")
    if arch in ("hopper", "ada"):
        return ("fp8", "int8", "awq")
    if arch == "ampere":
        return ("int8", "awq", "gptq")
    if arch == "volta_turing":
        return ("int8", "awq")
    return ()


def _query_nvidia_smi() -> tuple[GpuInfo, ...]:
    def _run(fields: str):
        try:
            return subprocess.run(
                ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                check=False, capture_output=True, text=True, timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

    result = _run("name,memory.total,compute_cap")
    if result is None or result.returncode != 0:   # older drivers may lack compute_cap
        result = _run("name,memory.total")
        if result is None or result.returncode != 0:
            return ()

    gpus: list[GpuInfo] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        raw_memory = parts[1].strip("[]")
        memory = 0 if raw_memory in ("N/A", "") else int(raw_memory)
        cc = parts[2].strip("[]") if len(parts) > 2 else ""
        if cc in ("N/A", "Not Supported", ""):
            cc = ""
        gpus.append(GpuInfo(name=parts[0], memory_total_mb=memory, compute_capability=cc))
    return tuple(gpus)


def _profile_from_name(name: str, gpus: tuple[GpuInfo, ...]) -> HardwareProfile:
    normalized = name.lower().replace("-", "_")
    if normalized in {"dgx_spark", "gb10", "spark"}:
        return HardwareProfile(
            name="dgx_spark",
            label="DGX Spark / GB10 unified memory",
            gpus=gpus,
            vllm_env={
                "GPU_MEMORY_UTILIZATION": "0.85",
                "MAX_MODEL_LEN": "32768",
                "MAX_NUM_BATCHED_TOKENS": "32768",
                "VLLM_MAX_NUM_SEQS": "4",
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
                "VLLM_ENABLE_PREFIX_CACHING": "1",
                "VLLM_KV_CACHE_DTYPE": "fp8_e4m3",
                "VLLM_DTYPE": "auto",
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "VLLM_TEST_FORCE_FP8_MARLIN": "1",
                "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
                "VLLM_NVFP4_GEMM_BACKEND": "cutlass",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_MATMUL_PRECISION": "high",
                "NVIDIA_FORWARD_COMPAT": "1",
                "NVIDIA_DISABLE_REQUIRE": "1",
            },
            training_env={
                "MODEL_FORGE_PARALLELISM": "32",
                "MODEL_FORGE_HIGH_PARALLELISM": "192",
                "TOKENIZERS_PARALLELISM": "true",
                "OMP_NUM_THREADS": "1",
            },
            quantization_env={
                "MODEL_FORGE_QUANT_CALIBRATION_SAMPLES": "512",
                "MODEL_FORGE_QUANT_CALIBRATION_SEQ_LEN": "4096",
                "MODEL_FORGE_QUANT_BATCH_SIZE": "auto",
                "MODEL_FORGE_QUANT_KEEP_BF16_PATTERNS": "router,gate,vision,visual,embed_vision,multi_modal_projector",
                "MODEL_FORGE_MOE_FAST_CALIBRATION": "1",
                "MODEL_FORGE_NVFP4_WEIGHT_BACKEND": "modelopt_or_native_nvfp4",
                "MODEL_FORGE_NVFP4_SERVING_BACKEND": "vllm_blackwell_cutlass",
            },
            notes=(
                "Use a Spark/GB10-native vLLM build; stock wheels are usually not compiled for SM 12.1.",
                "Conservative DGX Spark cap; raise GPU_MEMORY_UTILIZATION only after smoke serving passes.",
                "Use FP8 KV cache, chunked prefill, prefix caching, and low max-num-seqs to preserve unified-memory headroom.",
                "For NVFP4 MoE, keep routers and multimodal projection/vision modules in BF16 unless model-specific evals prove otherwise.",
                "For CPU/input-pipeline bottlenecks, DGX Spark can benefit from high parallelism such as c=192, but gate it behind an explicit override.",
            ),
        )
    if normalized in {"blackwell_dedicated", "blackwell", "rtx_pro_6000"}:
        return HardwareProfile(
            name="blackwell_dedicated",
            label="Blackwell dedicated VRAM",
            gpus=gpus,
            vllm_env={
                "GPU_MEMORY_UTILIZATION": "0.92",
                "MAX_MODEL_LEN": "32768",
                "MAX_NUM_BATCHED_TOKENS": "32768",
                "VLLM_MAX_NUM_SEQS": "8",
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
                "VLLM_ENABLE_PREFIX_CACHING": "1",
                "VLLM_KV_CACHE_DTYPE": "auto",
                "VLLM_DTYPE": "auto",
                "VLLM_NVFP4_GEMM_BACKEND": "cutlass",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_MATMUL_PRECISION": "high",
            },
            training_env={
                "MODEL_FORGE_PARALLELISM": "48",
                "MODEL_FORGE_HIGH_PARALLELISM": "192",
                "TOKENIZERS_PARALLELISM": "true",
                "OMP_NUM_THREADS": "1",
            },
            quantization_env={
                "MODEL_FORGE_QUANT_CALIBRATION_SAMPLES": "512",
                "MODEL_FORGE_QUANT_CALIBRATION_SEQ_LEN": "4096",
                "MODEL_FORGE_QUANT_BATCH_SIZE": "auto",
                "MODEL_FORGE_QUANT_KEEP_BF16_PATTERNS": "router,gate,vision,visual,embed_vision,multi_modal_projector",
                "MODEL_FORGE_MOE_FAST_CALIBRATION": "1",
                "MODEL_FORGE_NVFP4_WEIGHT_BACKEND": "modelopt_or_native_nvfp4",
                "MODEL_FORGE_NVFP4_SERVING_BACKEND": "vllm_blackwell_cutlass",
            },
            notes=("Higher cap is for dedicated VRAM Blackwell cards; lower it if CUDA graph capture fails.",),
        )
    if normalized in {"cuda_small_vram", "small_cuda"}:
        return HardwareProfile(
            name="cuda_small_vram",
            label="CUDA small VRAM",
            gpus=gpus,
            vllm_env={
                "GPU_MEMORY_UTILIZATION": "0.75",
                "MAX_MODEL_LEN": "8192",
                "MAX_NUM_BATCHED_TOKENS": "4096",
                "VLLM_MAX_NUM_SEQS": "2",
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
                "VLLM_ENABLE_PREFIX_CACHING": "1",
            },
            training_env={
                "MODEL_FORGE_PARALLELISM": "8",
                "MODEL_FORGE_HIGH_PARALLELISM": "32",
                "TOKENIZERS_PARALLELISM": "true",
                "OMP_NUM_THREADS": "1",
            },
            notes=("Small VRAM profile prioritizes successful startup over throughput.",),
        )
    if normalized in {"cuda_large_vram", "large_cuda", "cuda"}:
        return HardwareProfile(
            name="cuda_large_vram",
            label="CUDA large VRAM",
            gpus=gpus,
            vllm_env={
                "GPU_MEMORY_UTILIZATION": "0.88",
                "MAX_MODEL_LEN": "32768",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "VLLM_MAX_NUM_SEQS": "8",
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
                "VLLM_ENABLE_PREFIX_CACHING": "1",
                "VLLM_KV_CACHE_DTYPE": "auto",
            },
            training_env={
                "MODEL_FORGE_PARALLELISM": "32",
                "MODEL_FORGE_HIGH_PARALLELISM": "96",
                "TOKENIZERS_PARALLELISM": "true",
                "OMP_NUM_THREADS": "1",
            },
            notes=("Generic CUDA profile; tune per model after smoke tests.",),
        )
    return HardwareProfile(
        name="cpu",
        label="CPU / no CUDA detected",
        gpus=gpus,
        vllm_env={
            "MAX_MODEL_LEN": "4096",
            "MAX_NUM_BATCHED_TOKENS": "1024",
        },
        training_env={
            "MODEL_FORGE_PARALLELISM": str(max(1, min(8, cpu_count()))),
            "MODEL_FORGE_HIGH_PARALLELISM": str(max(1, min(16, cpu_count()))),
            "TOKENIZERS_PARALLELISM": "true",
            "OMP_NUM_THREADS": "1",
        },
        notes=("No NVIDIA GPU detected; serving large models is not recommended.",),
    )


def _select_profile_name(gpus: tuple[GpuInfo, ...], cc: str) -> str:
    names = " ".join(gpu.name.lower() for gpu in gpus)
    max_mb = max((gpu.memory_total_mb for gpu in gpus), default=0)
    if "gb10" in names or "dgx spark" in names:
        return "dgx_spark"
    if arch_from_cc(cc) == "blackwell" or "blackwell" in names or "rtx pro 6000" in names \
            or "b200" in names or "b300" in names:
        return "blackwell_dedicated"
    if max_mb >= 70_000:
        return "cuda_large_vram"
    if gpus:
        return "cuda_small_vram"
    return "cpu"


def detect_hardware_profile(env: Mapping[str, str] | None = None) -> HardwareProfile:
    env = env or os.environ
    gpus = _query_nvidia_smi()
    # MODEL_FORGE_COMPUTE_CAP lets you simulate other archs (e.g. 8.0 for A100, 9.0 for H100)
    cc = env.get("MODEL_FORGE_COMPUTE_CAP") or next(
        (g.compute_capability for g in gpus if g.compute_capability), ""
    )
    forced = env.get("MODEL_FORGE_HARDWARE_PROFILE")
    profile = _profile_from_name(forced or _select_profile_name(gpus, cc), gpus)
    return replace(profile, compute_capability=cc, supported_quant=supported_quant_for_cc(cc))


def recommended_vllm_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = env or os.environ
    profile = detect_hardware_profile(env)
    recommendations = dict(profile.vllm_env)
    for key in (
        "GPU_MEMORY_UTILIZATION",
        "MAX_MODEL_LEN",
        "MAX_NUM_BATCHED_TOKENS",
        "VLLM_MAX_NUM_SEQS",
        "VLLM_CPU_OFFLOAD_GB",
        "VLLM_SWAP_SPACE",
        "VLLM_KV_CACHE_DTYPE",
        "VLLM_DTYPE",
        "VLLM_QUANTIZATION",
        "VLLM_SPECULATIVE_CONFIG",
        "VLLM_ENABLE_CHUNKED_PREFILL",
        "VLLM_ENABLE_PREFIX_CACHING",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
        "VLLM_ENABLE_AUTO_TOOL_CHOICE",
        "VLLM_TOOL_CALL_PARSER",
        "VLLM_REASONING_PARSER",
        "VLLM_TEST_FORCE_FP8_MARLIN",
        "VLLM_MARLIN_USE_ATOMIC_ADD",
        "VLLM_NVFP4_GEMM_BACKEND",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_MATMUL_PRECISION",
        "NVIDIA_FORWARD_COMPAT",
        "NVIDIA_DISABLE_REQUIRE",
    ):
        if key in env:
            recommendations[key] = env[key]
    return recommendations


def recommended_training_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = env or os.environ
    profile = detect_hardware_profile(env)
    recommendations = dict(profile.training_env)
    if env.get("MODEL_FORGE_ENABLE_HIGH_PARALLELISM") == "1":
        recommendations["MODEL_FORGE_PARALLELISM"] = recommendations["MODEL_FORGE_HIGH_PARALLELISM"]
    for key in (
        "MODEL_FORGE_PARALLELISM",
        "MODEL_FORGE_HIGH_PARALLELISM",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if key in env:
            recommendations[key] = env[key]
    return recommendations


def recommended_quantization_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = env or os.environ
    profile = detect_hardware_profile(env)
    recommendations = dict(profile.quantization_env)
    for key in (
        "MODEL_FORGE_QUANT_CALIBRATION_SAMPLES",
        "MODEL_FORGE_QUANT_CALIBRATION_SEQ_LEN",
        "MODEL_FORGE_QUANT_BATCH_SIZE",
        "MODEL_FORGE_QUANT_KEEP_BF16_PATTERNS",
        "MODEL_FORGE_MOE_FAST_CALIBRATION",
        "MODEL_FORGE_NVFP4_WEIGHT_BACKEND",
        "MODEL_FORGE_NVFP4_SERVING_BACKEND",
    ):
        if key in env:
            recommendations[key] = env[key]
    return recommendations
