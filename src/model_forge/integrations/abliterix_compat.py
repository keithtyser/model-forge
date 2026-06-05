from __future__ import annotations

import functools
from typing import Any


def reduce_harmfulness_pair(vectors: Any, *, reduction: str = "normalized_sum", harmfulness_weight: float = 1.0) -> Any:
    """Reduce Abliterix (refusal, harmfulness) directions to layer-aligned vectors.

    Abliterix harmfulness mode returns shape ``(2, layers + 1, hidden_dim)``.
    Its LoRA steering path expects ``(layers + 1, hidden_dim)``.  Model Forge
    keeps the dual-direction signal by default via a normalized per-layer sum,
    while preserving exact refusal-only/harmfulness-only options for diagnostics.
    """
    if not (getattr(vectors, "ndim", None) == 3 and vectors.shape[0] == 2):
        return vectors

    import torch
    import torch.nn.functional as F

    refusal = vectors[0].float()
    harmfulness = vectors[1].float()
    mode = reduction.replace("-", "_").lower()

    if mode == "refusal_only":
        combined = refusal
    elif mode == "harmfulness_only":
        combined = harmfulness
    elif mode in {"normalized_sum", "sum"}:
        combined = refusal + float(harmfulness_weight) * harmfulness
    else:
        raise ValueError(
            "unsupported Abliterix harmfulness pair reduction "
            f"{reduction!r}; expected normalized_sum, refusal_only, or harmfulness_only"
        )

    combined = torch.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
    norms = torch.linalg.vector_norm(combined, dim=1, keepdim=True)
    fallback = refusal if mode != "harmfulness_only" else harmfulness
    combined = torch.where(norms > 1e-6, combined, fallback)
    combined = F.normalize(combined.float(), p=2, dim=1)
    return combined.to(vectors.dtype)


def apply_abliterix_compat_patches(
    *,
    reduction: str = "normalized_sum",
    harmfulness_weight: float = 1.0,
) -> None:
    """Patch Abliterix harmfulness-pair vectors into layer-aligned steering.

    This is intentionally narrow: it only touches tensors shaped like the
    Abliterix harmfulness/refusal pair.  Normal single-direction and unrelated
    multi-direction paths are left alone.
    """
    import abliterix.cli as ax_cli
    import abliterix.core.steering as ax_steering
    import abliterix.harmfulness as ax_harmfulness
    import abliterix.optimizer as ax_optimizer
    import abliterix.vectors as ax_vectors

    if getattr(ax_vectors.compute_steering_vectors, "_model_forge_pair_patch", False):
        return

    original_compute = ax_vectors.compute_steering_vectors
    original_extract = ax_harmfulness.extract_harm_refusal_pair
    original_apply = ax_steering.apply_steering

    @functools.wraps(original_compute)
    def compute_steering_vectors_patched(*args: Any, **kwargs: Any) -> Any:
        result = original_compute(*args, **kwargs)
        if kwargs.get("ablate_harmfulness_direction"):
            return reduce_harmfulness_pair(
                result,
                reduction=reduction,
                harmfulness_weight=harmfulness_weight,
            )
        return result

    @functools.wraps(original_extract)
    def extract_harm_refusal_pair_patched(*args: Any, **kwargs: Any) -> Any:
        return reduce_harmfulness_pair(
            original_extract(*args, **kwargs),
            reduction=reduction,
            harmfulness_weight=harmfulness_weight,
        )

    @functools.wraps(original_apply)
    def apply_steering_patched(
        engine: Any,
        steering_vectors: Any,
        vector_index: float | None,
        profiles: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return original_apply(
            engine,
            reduce_harmfulness_pair(
                steering_vectors,
                reduction=reduction,
                harmfulness_weight=harmfulness_weight,
            ),
            vector_index,
            profiles,
            *args,
            **kwargs,
        )

    compute_steering_vectors_patched._model_forge_pair_patch = True  # type: ignore[attr-defined]
    ax_vectors.compute_steering_vectors = compute_steering_vectors_patched
    ax_cli.compute_steering_vectors = compute_steering_vectors_patched
    ax_harmfulness.extract_harm_refusal_pair = extract_harm_refusal_pair_patched
    ax_steering.apply_steering = apply_steering_patched
    ax_optimizer.apply_steering = apply_steering_patched

