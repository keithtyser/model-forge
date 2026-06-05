from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DECODER_KEY_HINTS = (
    "decoder",
    "w_dec",
    "wdec",
    "dec_weight",
    "dictionary",
)

ENCODER_KEY_HINTS = (
    "encoder",
    "w_enc",
    "wenc",
    "enc_weight",
)


def _is_probably_decoder_key(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in DECODER_KEY_HINTS) and not any(
        hint in lowered for hint in ENCODER_KEY_HINTS
    )


def _safe_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except TypeError:
        return None


def discover_decoder_tensor(state: dict[str, Any], *, hidden_size: int) -> tuple[str, Any]:
    """Return a decoder dictionary as rows shaped [features, hidden_size]."""
    import torch

    candidates: list[tuple[int, str, Any]] = []
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        shape = _safe_shape(value)
        if shape is None or len(shape) != 2 or hidden_size not in shape:
            continue
        rows_are_features = shape[1] == hidden_size
        tensor = value if rows_are_features else value.T
        score = 0
        if _is_probably_decoder_key(str(key)):
            score += 100
        score += min(int(tensor.shape[0]), 10_000) // 1000
        candidates.append((score, str(key), tensor.detach().float().cpu()))
    if not candidates:
        raise SystemExit(f"could not find a 2D SAE decoder tensor with hidden size {hidden_size}")
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, key, tensor = candidates[0]
    return key, tensor.contiguous()


def _load_state_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise SystemExit("safetensors is required to load SAE safetensors files") from exc
        return dict(load_file(str(path), device="cpu"))
    if suffix in {".pt", ".pth", ".bin"}:
        import torch

        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            return loaded
        raise SystemExit(f"SAE state file {path} did not contain a tensor dictionary")
    raise SystemExit(f"unsupported SAE state file suffix: {path}")


def resolve_sae_source(source: str | Path, *, local_files_only: bool = False) -> Path:
    path = Path(str(source)).expanduser()
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            f"SAE source {source!r} is not a local path and huggingface_hub is not installed"
        ) from exc
    return Path(snapshot_download(str(source), local_files_only=local_files_only))


def load_decoder_dictionary(
    source: str | Path,
    *,
    hidden_size: int,
    file_name: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    root = resolve_sae_source(source, local_files_only=local_files_only)
    files: list[Path]
    if root.is_file():
        files = [root]
    elif file_name:
        selected = root / file_name
        if not selected.exists():
            raise SystemExit(f"configured SAE file does not exist: {selected}")
        files = [selected]
    else:
        files = [
            *sorted(root.glob("*.safetensors")),
            *sorted(root.glob("*.pt")),
            *sorted(root.glob("*.pth")),
            *sorted(root.glob("*.bin")),
        ]
    if not files:
        raise SystemExit(f"no SAE tensor files found in {root}")

    errors: list[str] = []
    for path in files:
        try:
            key, decoder = discover_decoder_tensor(_load_state_file(path), hidden_size=hidden_size)
        except SystemExit as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        return {
            "source": str(source),
            "resolved_path": str(root),
            "file": str(path),
            "decoder_key": key,
            "decoder": decoder,
            "feature_count": int(decoder.shape[0]),
            "hidden_size": int(decoder.shape[1]),
        }
    raise SystemExit("could not load an SAE decoder dictionary; " + "; ".join(errors[:5]))


def normalize_basis(direction: Any) -> Any:
    import torch

    direction = direction.float()
    if direction.ndim == 1:
        return direction / torch.linalg.vector_norm(direction).clamp_min(1e-6)
    if direction.ndim != 2:
        raise SystemExit(f"direction tensor must be 1D or 2D, got shape {tuple(direction.shape)}")
    norms = torch.linalg.vector_norm(direction, dim=1)
    direction = direction[norms > 1e-6]
    if direction.numel() == 0:
        raise SystemExit("direction basis is empty after dropping zero-norm components")
    q, _ = torch.linalg.qr(direction.T, mode="reduced")
    return q.T.contiguous()


def constrain_direction_to_decoder(
    direction: Any,
    decoder: Any,
    *,
    top_k: int = 8,
    min_abs_cosine: float = 0.0,
) -> tuple[Any, list[dict[str, Any]]]:
    import torch

    basis = normalize_basis(direction)
    if basis.ndim == 1:
        basis = basis.unsqueeze(0)
    decoder = decoder.float()
    decoder = decoder / torch.linalg.vector_norm(decoder, dim=1, keepdim=True).clamp_min(1e-6)
    selected_vectors = []
    selected: list[dict[str, Any]] = []
    for component_index, component in enumerate(basis):
        scores = decoder @ component.float()
        k = min(max(1, int(top_k)), int(scores.numel()))
        values, indices = torch.topk(scores.abs(), k=k)
        for rank, (value, feature_index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
            cosine = float(scores[int(feature_index)].item())
            if abs(cosine) < float(min_abs_cosine):
                continue
            selected_vectors.append(decoder[int(feature_index)] * (1.0 if cosine >= 0 else -1.0))
            selected.append({
                "component": component_index,
                "rank": rank,
                "feature_index": int(feature_index),
                "cosine": cosine,
                "abs_cosine": float(value),
            })
    if not selected_vectors:
        raise SystemExit(
            f"no SAE decoder features met min_abs_cosine={float(min_abs_cosine):.4f}; lower the gate or choose another SAE"
        )
    constrained = normalize_basis(torch.stack(selected_vectors))
    return constrained, selected


def constrain_direction_artifact(
    artifact: dict[str, Any],
    *,
    decoder: Any,
    top_k: int,
    min_abs_cosine: float = 0.0,
    layer_filter: set[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directions = artifact.get("refusal_directions") or artifact.get("directions") or {}
    if not isinstance(directions, dict):
        raise SystemExit("direction artifact does not contain a layer-indexed direction dictionary")
    output = dict(artifact)
    constrained: dict[int, Any] = {}
    layer_reports: dict[str, Any] = {}
    for raw_layer, direction in directions.items():
        layer = int(raw_layer)
        if layer_filter is not None and layer not in layer_filter:
            constrained[layer] = direction
            continue
        constrained_direction, selected = constrain_direction_to_decoder(
            direction,
            decoder,
            top_k=top_k,
            min_abs_cosine=min_abs_cosine,
        )
        constrained[layer] = constrained_direction
        layer_reports[str(layer)] = {
            "input_shape": list(direction.shape),
            "output_shape": list(constrained_direction.shape),
            "selected_features": selected,
        }
    output["refusal_directions"] = constrained
    output["format"] = "direction_artifact_v1"
    output["sae_dictionary_constraint"] = {
        "top_k": int(top_k),
        "min_abs_cosine": float(min_abs_cosine),
        "constrained_layers": sorted(int(item) for item in layer_reports),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report = {
        "schema_version": "model_forge.sae_dictionary_constraint.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_k": int(top_k),
        "min_abs_cosine": float(min_abs_cosine),
        "layer_count": len(layer_reports),
        "layers": layer_reports,
    }
    return output, report


def constrain_direction_artifact_with_decoder_loader(
    artifact: dict[str, Any],
    *,
    decoder_for_layer: Any,
    top_k: int,
    min_abs_cosine: float = 0.0,
    layer_filter: set[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    directions = artifact.get("refusal_directions") or artifact.get("directions") or {}
    if not isinstance(directions, dict):
        raise SystemExit("direction artifact does not contain a layer-indexed direction dictionary")
    output = dict(artifact)
    constrained: dict[int, Any] = {}
    layer_reports: dict[str, Any] = {}
    for raw_layer, direction in directions.items():
        layer = int(raw_layer)
        if layer_filter is not None and layer not in layer_filter:
            constrained[layer] = direction
            continue
        decoder_info = decoder_for_layer(layer, direction)
        constrained_direction, selected = constrain_direction_to_decoder(
            direction,
            decoder_info["decoder"],
            top_k=top_k,
            min_abs_cosine=min_abs_cosine,
        )
        constrained[layer] = constrained_direction
        layer_reports[str(layer)] = {
            "input_shape": list(direction.shape),
            "output_shape": list(constrained_direction.shape),
            "sae_file": decoder_info.get("file"),
            "decoder_key": decoder_info.get("decoder_key"),
            "feature_count": decoder_info.get("feature_count"),
            "selected_features": selected,
        }
    output["refusal_directions"] = constrained
    output["format"] = "direction_artifact_v1"
    output["sae_dictionary_constraint"] = {
        "top_k": int(top_k),
        "min_abs_cosine": float(min_abs_cosine),
        "constrained_layers": sorted(int(item) for item in layer_reports),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    report = {
        "schema_version": "model_forge.sae_dictionary_constraint.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "top_k": int(top_k),
        "min_abs_cosine": float(min_abs_cosine),
        "layer_count": len(layer_reports),
        "layers": layer_reports,
    }
    return output, report


def rewrite_direction_artifact_with_sae(
    *,
    input_path: Path,
    output_path: Path,
    sae_source: str,
    hidden_size: int,
    top_k: int,
    min_abs_cosine: float = 0.0,
    sae_file: str | None = None,
    sae_file_pattern: str | None = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch

    artifact = torch.load(input_path, map_location="cpu")
    decoder_cache: dict[str, dict[str, Any]] = {}
    loaded_decoders: list[dict[str, Any]] = []

    def decoder_for_layer(layer: int, direction: Any) -> dict[str, Any]:
        feature_file = sae_file
        if sae_file_pattern:
            feature_file = sae_file_pattern.format(layer=layer)
            dictionary = load_decoder_dictionary(
                sae_source,
                hidden_size=hidden_size,
                file_name=feature_file,
                local_files_only=local_files_only,
            )
            loaded_decoders.append({key: value for key, value in dictionary.items() if key != "decoder"})
            return dictionary
        cache_key = feature_file or "<auto>"
        if cache_key not in decoder_cache:
            dictionary = load_decoder_dictionary(
                sae_source,
                hidden_size=hidden_size,
                file_name=feature_file,
                local_files_only=local_files_only,
            )
            decoder_cache[cache_key] = dictionary
            loaded_decoders.append({key: value for key, value in dictionary.items() if key != "decoder"})
        return decoder_cache[cache_key]

    constrained, report = constrain_direction_artifact_with_decoder_loader(
        artifact,
        decoder_for_layer=decoder_for_layer,
        top_k=top_k,
        min_abs_cosine=min_abs_cosine,
    )
    report["sae"] = {
        "source": sae_source,
        "file": sae_file,
        "file_pattern": sae_file_pattern,
        "hidden_size": int(hidden_size),
        "loaded_decoder_count": len(loaded_decoders),
        "loaded_decoders": loaded_decoders,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(constrained, output_path)
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["output_path"] = str(output_path)
    report["report_path"] = str(report_path)
    return report
