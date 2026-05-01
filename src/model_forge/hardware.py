from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from typing import Mapping


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total_mb: int


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    label: str
    gpus: tuple[GpuInfo, ...] = ()
    vllm_env: Mapping[str, str] = field(default_factory=dict)
    training_env: Mapping[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


def _query_nvidia_smi() -> tuple[GpuInfo, ...]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if result.returncode != 0:
        return ()

    gpus: list[GpuInfo] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"(.+),\s*(\d+|\[?N/A\]?)$", line)
        if not match:
            continue
        raw_memory = match.group(2).strip("[]")
        memory = 0 if raw_memory == "N/A" else int(raw_memory)
        gpus.append(GpuInfo(name=match.group(1).strip(), memory_total_mb=memory))
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
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
                "VLLM_KV_CACHE_DTYPE": "auto",
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
            notes=(
                "Conservative DGX Spark cap; AEON reports 0.88+ can thrash unified memory.",
                "Keep one server at a time and prefer shorter contexts while developing.",
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
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
                "VLLM_KV_CACHE_DTYPE": "auto",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_MATMUL_PRECISION": "high",
            },
            training_env={
                "MODEL_FORGE_PARALLELISM": "48",
                "MODEL_FORGE_HIGH_PARALLELISM": "192",
                "TOKENIZERS_PARALLELISM": "true",
                "OMP_NUM_THREADS": "1",
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
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
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
                "VLLM_ENABLE_CHUNKED_PREFILL": "1",
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


def detect_hardware_profile(env: Mapping[str, str] | None = None) -> HardwareProfile:
    env = env or os.environ
    gpus = _query_nvidia_smi()
    forced = env.get("MODEL_FORGE_HARDWARE_PROFILE")
    if forced:
        return _profile_from_name(forced, gpus)

    names = " ".join(gpu.name.lower() for gpu in gpus)
    max_mb = max((gpu.memory_total_mb for gpu in gpus), default=0)
    if "gb10" in names or "dgx spark" in names:
        return _profile_from_name("dgx_spark", gpus)
    if "blackwell" in names or "rtx pro 6000" in names or "b200" in names or "b300" in names:
        return _profile_from_name("blackwell_dedicated", gpus)
    if max_mb >= 70_000:
        return _profile_from_name("cuda_large_vram", gpus)
    if gpus:
        return _profile_from_name("cuda_small_vram", gpus)
    return _profile_from_name("cpu", gpus)


def recommended_vllm_env(env: Mapping[str, str] | None = None) -> dict[str, str]:
    env = env or os.environ
    profile = detect_hardware_profile(env)
    recommendations = dict(profile.vllm_env)
    for key in (
        "GPU_MEMORY_UTILIZATION",
        "MAX_MODEL_LEN",
        "MAX_NUM_BATCHED_TOKENS",
        "VLLM_CPU_OFFLOAD_GB",
        "VLLM_SWAP_SPACE",
        "VLLM_KV_CACHE_DTYPE",
        "VLLM_ENABLE_CHUNKED_PREFILL",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_MATMUL_PRECISION",
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
