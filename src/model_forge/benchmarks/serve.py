from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.runs.manifest import (
    REPO_DIR,
    build_canonical_manifest,
    display_path,
    redact_value,
    sanitize_run_id,
)


DEFAULT_CONFIG = REPO_DIR / "configs" / "serving" / "serve_bench_smoke.yaml"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
SCHEMA_VERSION = "model_forge.serve_benchmark.v1"

console = Console(stderr=True)


@dataclass(frozen=True)
class ServeRequest:
    request_id: str
    category: str
    messages: list[dict[str, str]]
    sampling: dict[str, Any]
    extra_body: dict[str, Any]


@dataclass(frozen=True)
class ServeBenchConfig:
    name: str
    description: str
    family: str | None
    variant: str | None
    model: str
    base_url: str
    api_key: str | None
    timeout_seconds: int
    stream: bool
    stream_include_usage: bool
    sampling: dict[str, Any]
    repetitions: int
    concurrency: int
    output_root: Path
    requests: list[ServeRequest]
    workload_sources: list[Path]
    raw_config: dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return REPO_DIR / path


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {path}")
    return data


def env_value(env: Mapping[str, str], key: str | None) -> str | None:
    if not key:
        return None
    value = env.get(key)
    return value if value not in {None, ""} else None


def backend_value(backend: Mapping[str, Any], key: str, env: Mapping[str, str]) -> Any:
    env_key = backend.get(f"{key}_env")
    if env_key:
        value = env_value(env, str(env_key))
        if value is not None:
            return value
    return backend.get(key)


def load_family_variant_model(family: str | None, variant: str | None) -> str | None:
    if not family or not variant:
        return None
    path = REPO_DIR / "configs" / "model_families" / f"{family}.yaml"
    if not path.exists():
        return None
    data = load_yaml(path)
    raw_variant = (data.get("variants") or {}).get(variant)
    if not isinstance(raw_variant, dict):
        return None
    model = raw_variant.get("served_model_name") or raw_variant.get("repo_id")
    return str(model) if model else None


def request_messages(raw: Mapping[str, Any], default_system_prompt: str | None) -> list[dict[str, str]]:
    if isinstance(raw.get("messages"), list):
        messages = []
        for item in raw["messages"]:
            if not isinstance(item, dict) or not item.get("role") or item.get("content") is None:
                raise ValueError(f"invalid messages item in request {raw.get('id')!r}")
            messages.append({"role": str(item["role"]), "content": str(item["content"])})
        return messages

    prompt = raw.get("prompt")
    if prompt is None:
        raise ValueError(f"request {raw.get('id')!r} must define prompt or messages")
    system_prompt = raw.get("system_prompt", default_system_prompt)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})
    messages.append({"role": "user", "content": str(prompt)})
    return messages


def parse_workload_request(
    raw: Mapping[str, Any],
    *,
    request_id: str,
    category: str,
    sampling: Mapping[str, Any],
    default_system_prompt: str | None,
) -> ServeRequest:
    merged_sampling = dict(sampling)
    merged_sampling.update(raw.get("sampling") or {})
    return ServeRequest(
        request_id=request_id,
        category=category,
        messages=request_messages(raw, default_system_prompt),
        sampling=merged_sampling,
        extra_body=dict(raw.get("extra_body") or {}),
    )


def workload_file_paths(raw_config: Mapping[str, Any]) -> list[Path]:
    workload = raw_config.get("workload") or {}
    raw_paths = workload.get("files") or raw_config.get("workload_files") or []
    if isinstance(raw_paths, (str, Path)):
        raw_paths = [raw_paths]
    if not isinstance(raw_paths, list):
        raise ValueError("workload.files must be a list of paths")
    return [resolve_repo_path(path) for path in raw_paths]


def load_workload_file(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    if data.get("schema_version") != "model_forge.serving_workload.v1":
        raise ValueError(f"{display_path(path)} must use schema_version model_forge.serving_workload.v1")
    if not data.get("id"):
        raise ValueError(f"{display_path(path)} must define id")
    if not isinstance(data.get("requests"), list) or not data["requests"]:
        raise ValueError(f"{display_path(path)} must define a non-empty requests list")
    return data


def parse_requests(raw_config: Mapping[str, Any], sampling: Mapping[str, Any]) -> tuple[int, int, list[ServeRequest], list[Path]]:
    workload = raw_config.get("workload") or {}
    repetitions = int(workload.get("repetitions", 1))
    concurrency = int(workload.get("concurrency", 1))
    if repetitions < 1:
        raise ValueError("workload.repetitions must be >= 1")
    if concurrency < 1:
        raise ValueError("workload.concurrency must be >= 1")

    default_system_prompt = raw_config.get("system_prompt")
    requests: list[ServeRequest] = []
    workload_sources: list[Path] = []
    seen_ids: set[str] = set()

    for path in workload_file_paths(raw_config):
        workload_data = load_workload_file(path)
        workload_sources.append(path)
        workload_id = str(workload_data["id"])
        workload_category = str(workload_data.get("category") or workload_id)
        workload_system_prompt = workload_data.get("system_prompt", default_system_prompt)
        workload_sampling = dict(sampling)
        workload_sampling.update(workload_data.get("default_sampling") or workload_data.get("sampling") or {})
        for index, raw in enumerate(workload_data["requests"], start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"workload requests in {display_path(path)} must be mappings")
            raw_request_id = str(raw.get("id") or f"request_{index}")
            request_id = f"{workload_id}:{raw_request_id}"
            if request_id in seen_ids:
                raise ValueError(f"duplicate request id {request_id!r}")
            seen_ids.add(request_id)
            requests.append(
                parse_workload_request(
                    raw,
                    request_id=request_id,
                    category=str(raw.get("category", workload_category)),
                    sampling=workload_sampling,
                    default_system_prompt=str(raw.get("system_prompt", workload_system_prompt) or ""),
                )
            )

    raw_requests = workload.get("requests") or []
    if raw_requests and not isinstance(raw_requests, list):
        raise ValueError("workload.requests must be a list")
    for index, raw in enumerate(raw_requests, start=1):
        if not isinstance(raw, dict):
            raise ValueError("workload request entries must be mappings")
        request_id = str(raw.get("id") or f"request_{index}")
        if request_id in seen_ids:
            raise ValueError(f"duplicate request id {request_id!r}")
        seen_ids.add(request_id)
        requests.append(
            parse_workload_request(
                raw,
                request_id=request_id,
                category=str(raw.get("category", "generic")),
                sampling=sampling,
                default_system_prompt=str(raw.get("system_prompt", default_system_prompt) or ""),
            )
        )
    if not requests:
        raise ValueError("workload must define requests or workload.files")
    return repetitions, concurrency, requests, workload_sources


def load_config(
    path: Path,
    *,
    family: str | None = None,
    variant: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    output_dir: str | Path | None = None,
    repetitions: int | None = None,
    limit: int | None = None,
    stream: bool | None = None,
    env: Mapping[str, str] | None = None,
) -> ServeBenchConfig:
    env = env or os.environ
    raw = load_yaml(path)
    backend = raw.get("backend") or {}
    sampling = dict(raw.get("sampling") or {})

    resolved_family = family or (raw.get("model") or {}).get("family")
    resolved_variant = variant or (raw.get("model") or {}).get("variant")
    resolved_model = (
        model
        or env_value(env, "MODEL_FORGE_MODEL")
        or backend_value(backend, "model", env)
        or load_family_variant_model(resolved_family, resolved_variant)
    )
    if not resolved_model:
        raise ValueError("serving benchmark needs --model, MODEL_FORGE_MODEL, backend.model, or --family/--variant")

    resolved_base_url = (
        base_url
        or env_value(env, "MODEL_FORGE_BASE_URL")
        or backend_value(backend, "base_url", env)
        or DEFAULT_BASE_URL
    )
    api_key = env_value(env, "MODEL_FORGE_API_KEY") or backend_value(backend, "api_key", env)
    api_key_env = backend.get("api_key_env")
    if not api_key and api_key_env:
        api_key = env_value(env, str(api_key_env))

    if env_value(env, "MODEL_FORGE_TEMPERATURE"):
        sampling["temperature"] = float(str(env["MODEL_FORGE_TEMPERATURE"]))
    if env_value(env, "MODEL_FORGE_MAX_TOKENS"):
        sampling["max_tokens"] = int(str(env["MODEL_FORGE_MAX_TOKENS"]))
    if env_value(env, "MODEL_FORGE_TOP_P"):
        sampling["top_p"] = float(str(env["MODEL_FORGE_TOP_P"]))

    parsed_repetitions, concurrency, requests, workload_sources = parse_requests(raw, sampling)
    if repetitions is not None:
        parsed_repetitions = repetitions
    if limit is not None:
        requests = requests[:limit]
    if not requests:
        raise ValueError("serving benchmark request list is empty after applying --limit")
    if concurrency != 1:
        raise ValueError("serve bench MVP supports serial workloads only; set workload.concurrency: 1")

    output_root = resolve_repo_path(output_dir or raw.get("output_dir") or "reports/generated/serve_bench")
    return ServeBenchConfig(
        name=str(raw.get("name", "serve_bench")),
        description=str(raw.get("description", "")),
        family=str(resolved_family) if resolved_family else None,
        variant=str(resolved_variant) if resolved_variant else None,
        model=str(resolved_model),
        base_url=str(resolved_base_url).rstrip("/"),
        api_key=str(api_key) if api_key else None,
        timeout_seconds=int(env_value(env, "MODEL_FORGE_TIMEOUT_SECONDS") or backend.get("timeout_seconds", 120)),
        stream=bool(backend.get("stream", True) if stream is None else stream),
        stream_include_usage=bool(backend.get("stream_include_usage", True)),
        sampling=sampling,
        repetitions=parsed_repetitions,
        concurrency=concurrency,
        output_root=output_root,
        requests=requests,
        workload_sources=workload_sources,
        raw_config=raw,
    )


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def request_body(config: ServeBenchConfig, request: ServeRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": config.model,
        "messages": request.messages,
        "stream": config.stream,
    }
    for key in ("temperature", "top_p", "max_tokens", "min_tokens", "stop", "presence_penalty", "frequency_penalty"):
        if key in request.sampling:
            body[key] = request.sampling[key]
    extra_body = dict((config.raw_config.get("backend") or {}).get("extra_body") or {})
    extra_body.update(request.extra_body)
    body.update(extra_body)
    if config.stream and config.stream_include_usage:
        body.setdefault("stream_options", {"include_usage": True})
    return body


def request_headers(config: ServeBenchConfig, stream: bool) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def parse_streaming_chunks(
    lines: Iterable[bytes],
    *,
    started_at: float,
    clock: Any = time.perf_counter,
) -> dict[str, Any]:
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    first_chunk_seconds: float | None = None
    first_token_seconds: float | None = None
    chunk_count = 0
    token_event_count = 0
    finish_reason: str | None = None
    raw_chunk_count = 0

    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if payload_text == "[DONE]":
            break
        raw_chunk_count += 1
        if first_chunk_seconds is None:
            first_chunk_seconds = clock() - started_at
        payload = json.loads(payload_text)
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        choices = payload.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        finish_reason = choice.get("finish_reason") or finish_reason
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is None:
            content = choice.get("text")
        if content:
            chunk_count += 1
            token_event_count += 1
            if first_token_seconds is None:
                first_token_seconds = clock() - started_at
            text_parts.append(str(content))

    return {
        "text": "".join(text_parts),
        "usage": usage,
        "first_chunk_seconds": first_chunk_seconds,
        "first_token_seconds": first_token_seconds,
        "stream_chunk_count": chunk_count,
        "stream_token_event_count": token_event_count,
        "raw_stream_chunk_count": raw_chunk_count,
        "finish_reason": finish_reason,
    }


def completion_token_estimate(usage: Mapping[str, Any], stream_token_event_count: int) -> tuple[int | None, str | None]:
    for key in ("completion_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            return value, f"usage.{key}"
    if stream_token_event_count > 0:
        return stream_token_event_count, "stream_event_count_estimate"
    return None, None


def build_metrics(
    *,
    total_latency_seconds: float,
    usage: Mapping[str, Any],
    first_token_seconds: float | None,
    stream_token_event_count: int,
    first_chunk_seconds: float | None = None,
) -> dict[str, Any]:
    completion_tokens, completion_token_source = completion_token_estimate(usage, stream_token_event_count)
    prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
    total_tokens = usage.get("total_tokens")
    metrics: dict[str, Any] = {
        "total_latency_seconds": round(total_latency_seconds, 6),
        "time_to_first_chunk_seconds": round(first_chunk_seconds, 6) if first_chunk_seconds is not None else None,
        "time_to_first_token_seconds": round(first_token_seconds, 6) if first_token_seconds is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "completion_token_source": completion_token_source,
        "total_tokens": total_tokens,
    }
    if completion_tokens and total_latency_seconds > 0:
        metrics["output_tokens_per_second"] = round(float(completion_tokens) / total_latency_seconds, 6)
    if first_token_seconds is not None and completion_tokens and completion_tokens > 1:
        decode_seconds = max(0.0, total_latency_seconds - first_token_seconds)
        metrics["decode_latency_seconds"] = round(decode_seconds, 6)
        if decode_seconds > 0:
            metrics["decode_tokens_per_second"] = round(float(completion_tokens - 1) / decode_seconds, 6)
            metrics["inter_token_latency_seconds"] = round(decode_seconds / float(completion_tokens - 1), 6)
    if isinstance(total_tokens, int) and total_latency_seconds > 0:
        metrics["total_tokens_per_second"] = round(float(total_tokens) / total_latency_seconds, 6)
    return metrics


def call_chat_completion(config: ServeBenchConfig, request: ServeRequest) -> dict[str, Any]:
    body = request_body(config, request)
    url = chat_completions_url(config.base_url)
    http_request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=request_headers(config, config.stream),
        method="POST",
    )
    started_at = time.perf_counter()
    try:
        with urllib.request.urlopen(http_request, timeout=config.timeout_seconds) as response:
            if config.stream:
                stream = parse_streaming_chunks(response, started_at=started_at)
                elapsed = time.perf_counter() - started_at
                usage = stream["usage"]
                metrics = build_metrics(
                    total_latency_seconds=elapsed,
                    usage=usage,
                    first_token_seconds=stream["first_token_seconds"],
                    stream_token_event_count=int(stream["stream_token_event_count"]),
                    first_chunk_seconds=stream["first_chunk_seconds"],
                )
                return {
                    "ok": True,
                    "http_status": getattr(response, "status", None),
                    "response_text": stream["text"],
                    "usage": usage,
                    "stream": {
                        "first_chunk_seconds": stream["first_chunk_seconds"],
                        "stream_chunk_count": stream["stream_chunk_count"],
                        "raw_stream_chunk_count": stream["raw_stream_chunk_count"],
                        "finish_reason": stream["finish_reason"],
                    },
                    "metrics": metrics,
                }

            status = getattr(response, "status", None)
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "http_status": exc.code, "error": f"backend HTTP {exc.code}: {detail}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "http_status": None, "error": f"backend URL error: {exc}"}

    elapsed = time.perf_counter() - started_at
    content = raw["choices"][0]["message"]["content"]
    usage = raw.get("usage", {})
    metrics = build_metrics(
        total_latency_seconds=elapsed,
        usage=usage,
        first_token_seconds=None,
        stream_token_event_count=0,
    )
    return {
        "ok": True,
        "http_status": status,
        "response_text": content,
        "usage": usage,
        "stream": None,
        "metrics": metrics,
        "raw_response_id": raw.get("id"),
    }


def prompt_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_benchmark(config: ServeBenchConfig) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(config.requests) * config.repetitions
    index = 0
    for repetition in range(1, config.repetitions + 1):
        for request in config.requests:
            index += 1
            console.print(f"[{index}/{total}] {request.request_id} rep={repetition}", style="dim")
            started = utc_now().isoformat()
            try:
                outcome = call_chat_completion(config, request)
            except Exception as exc:  # Keep the benchmark artifact useful even if one request fails.
                outcome = {"ok": False, "http_status": None, "error": str(exc)}
            response_text = str(outcome.get("response_text") or "")
            results.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": started,
                    "request_id": request.request_id,
                    "category": request.category,
                    "repetition": repetition,
                    "ok": bool(outcome.get("ok")),
                    "error": outcome.get("error"),
                    "http_status": outcome.get("http_status"),
                    "streaming": config.stream,
                    "prompt_sha256": prompt_hash(request.messages),
                    "prompt_chars": sum(len(message.get("content", "")) for message in request.messages),
                    "response_chars": len(response_text),
                    "usage": outcome.get("usage") or {},
                    "stream": outcome.get("stream"),
                    "metrics": outcome.get("metrics") or {},
                }
            )
    return results


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": round(min(values), 6),
        "mean": round(statistics.fmean(values), 6),
        "p50": round(percentile(values, 0.50) or 0.0, 6),
        "p95": round(percentile(values, 0.95) or 0.0, 6),
        "p99": round(percentile(values, 0.99) or 0.0, 6),
        "max": round(max(values), 6),
    }


def count_values(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value if value not in {None, ""} else "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def summarize_results(config: ServeBenchConfig, results: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    successful = [row for row in results if row.get("ok")]
    metric_names = sorted({metric for row in successful for metric, value in (row.get("metrics") or {}).items() if isinstance(value, (int, float))})
    metrics = {
        metric: metric_summary([float((row.get("metrics") or {}).get(metric)) for row in successful if isinstance((row.get("metrics") or {}).get(metric), (int, float))])
        for metric in metric_names
    }
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(row.get("category", "generic")) for row in results}):
        category_rows = [row for row in results if row.get("category") == category and row.get("ok")]
        all_category_rows = [row for row in results if row.get("category") == category]
        by_category[category] = {
            "request_count": len(all_category_rows),
            "successful_requests": len(category_rows),
            "failed_requests": len(all_category_rows) - len(category_rows),
            "metrics": {
                metric: metric_summary([
                    float((row.get("metrics") or {}).get(metric))
                    for row in category_rows
                    if isinstance((row.get("metrics") or {}).get(metric), (int, float))
                ])
                for metric in metric_names
            },
        }
    total_latency_values = [
        float((row.get("metrics") or {}).get("total_latency_seconds"))
        for row in successful
        if isinstance((row.get("metrics") or {}).get("total_latency_seconds"), (int, float))
    ]
    total_latency_sum = sum(total_latency_values)
    request_throughput = len(successful) / total_latency_sum if total_latency_sum > 0 else None
    finish_reasons = count_values([
        (row.get("stream") or {}).get("finish_reason")
        for row in successful
        if row.get("stream") is not None
    ])
    error_types = count_values([row.get("error") for row in results if not row.get("ok")])
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "name": config.name,
        "description": config.description,
        "family": config.family,
        "variant": config.variant,
        "model": config.model,
        "base_url": config.base_url,
        "streaming": config.stream,
        "repetitions": config.repetitions,
        "concurrency": config.concurrency,
        "workload_sources": [display_path(path) for path in config.workload_sources],
        "request_count": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "success_rate": round(len(successful) / len(results), 6) if results else 0.0,
        "request_throughput_per_second_serial_estimate": round(request_throughput, 6) if request_throughput else None,
        "finish_reasons": finish_reasons,
        "error_types": error_types,
        "metrics": metrics,
        "by_category": by_category,
        "output_dir": display_path(output_dir),
        "notes": [
            "This benchmark measures an already-running OpenAI-compatible serving endpoint.",
            "It is not a quality or behavior evaluation by itself.",
            "Run sampled evals under the same serving config before publishing serving claims.",
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(redact_value(row), sort_keys=True) + "\n")


def table_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def metric_values(metrics: Mapping[str, Any], metric: str) -> Mapping[str, Any]:
    raw = metrics.get(metric) or {}
    return raw if isinstance(raw, Mapping) else {}


def metric_triplet(metrics: Mapping[str, Any], metric: str) -> str:
    values = metric_values(metrics, metric)
    return f"p50={table_cell(values.get('p50'))}, p95={table_cell(values.get('p95'))}, p99={table_cell(values.get('p99'))}"


def serving_card_status(summary: Mapping[str, Any]) -> str:
    if summary.get("failed_requests"):
        return "needs_investigation"
    if not summary.get("request_count"):
        return "empty"
    return "serving_metrics_recorded"


def write_serving_card(path: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    by_category = summary.get("by_category") or {}
    identity = manifest.get("identity") or {}
    hardware = manifest.get("hardware") or {}
    outputs = manifest.get("outputs") or {}
    artifacts = outputs.get("artifacts") or {}
    command = (manifest.get("command") or {}).get("display") or "n/a"
    workload_sources = summary.get("workload_sources") or []
    config_paths = [
        item.get("path")
        for item in manifest.get("configs", [])
        if isinstance(item, Mapping) and item.get("path")
    ]

    category_rows = []
    for category, data in sorted(by_category.items()):
        cat_metrics = data.get("metrics") or {}
        category_rows.append(
            "| "
            + " | ".join(
                [
                    str(category),
                    table_cell(data.get("successful_requests")),
                    table_cell(data.get("failed_requests")),
                    metric_triplet(cat_metrics, "time_to_first_token_seconds"),
                    metric_triplet(cat_metrics, "inter_token_latency_seconds"),
                    metric_triplet(cat_metrics, "output_tokens_per_second"),
                    metric_triplet(cat_metrics, "total_latency_seconds"),
                ]
            )
            + " |"
        )
    if not category_rows:
        category_rows.append("| n/a | 0 | 0 | n/a | n/a | n/a | n/a |")

    lines = [
        f"# Serving Card: {summary.get('name')}",
        "",
        "## Identity",
        "",
        f"- Status: `{serving_card_status(summary)}`",
        f"- Model: `{summary.get('model')}`",
        f"- Family: `{summary.get('family') or identity.get('family') or ''}`",
        f"- Variant: `{summary.get('variant') or identity.get('variant') or ''}`",
        f"- Engine/API: `openai_compatible`",
        f"- Streaming: `{summary.get('streaming')}`",
        f"- Concurrency: `{summary.get('concurrency')}`",
        f"- Repetitions: `{summary.get('repetitions')}`",
        f"- Requests: `{summary.get('successful_requests')}/{summary.get('request_count')}` successful",
        f"- Success rate: `{table_cell(summary.get('success_rate'))}`",
        f"- Run manifest: `{manifest.get('run_id')}`",
        f"- Output directory: `{summary.get('output_dir')}`",
        "",
        "## Hardware And Config",
        "",
        f"- Hardware profile: `{hardware.get('profile') or ''}`",
        f"- Hardware label: `{hardware.get('label') or ''}`",
        f"- GPU count recorded: `{len(hardware.get('gpus') or [])}`",
        f"- Benchmark config: `{config_paths[0] if config_paths else ''}`",
        f"- Workload sources: `{', '.join(str(item) for item in workload_sources)}`",
        f"- Repro command: `{command}`",
        "",
        "## Overall Metrics",
        "",
        f"- TTFT seconds: `{metric_triplet(metrics, 'time_to_first_token_seconds')}`",
        f"- First chunk seconds: `{metric_triplet(metrics, 'time_to_first_chunk_seconds')}`",
        f"- ITL seconds: `{metric_triplet(metrics, 'inter_token_latency_seconds')}`",
        f"- Total latency seconds: `{metric_triplet(metrics, 'total_latency_seconds')}`",
        f"- Output tokens/sec: `{metric_triplet(metrics, 'output_tokens_per_second')}`",
        f"- Decode tokens/sec: `{metric_triplet(metrics, 'decode_tokens_per_second')}`",
        f"- Total tokens/sec: `{metric_triplet(metrics, 'total_tokens_per_second')}`",
        f"- Serial request throughput/sec: `{table_cell(summary.get('request_throughput_per_second_serial_estimate'))}`",
        f"- Finish reasons: `{json.dumps(summary.get('finish_reasons') or {}, sort_keys=True)}`",
        f"- Error types: `{json.dumps(summary.get('error_types') or {}, sort_keys=True)}`",
        "",
        "## Workload Metrics",
        "",
        "| Workload | Success | Failed | TTFT | ITL | Output tok/sec | Total latency |",
        "|---|---:|---:|---|---|---|---|",
        *category_rows,
        "",
        "## Artifacts",
        "",
        f"- Requests JSONL: `{artifacts.get('requests_jsonl', 'requests.jsonl')}`",
        f"- Summary JSON: `{artifacts.get('summary_json', 'summary.json')}`",
        f"- Serving Card: `{artifacts.get('serving_card_md', 'serving_card.md')}`",
        f"- Manifest JSON: `{artifacts.get('manifest_json', 'manifest.json')}`",
        "",
        "## Promotion Gates",
        "",
        "- Serving metrics are operational evidence only.",
        "- Run sampled quality and behavior evals under the same serving config before promotion.",
        "- Compare against a baseline with the same model, workload files, sampling, endpoint shape, and hardware profile.",
        "- Treat missing memory, cache-hit, truncation, quality, or behavior evidence as not yet measured.",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in summary.get("notes", [])],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_id_for(config: ServeBenchConfig, now: datetime | None = None) -> str:
    now = now or utc_now()
    parts = [config.name, config.family or "generic", config.variant or "model", now.strftime("%Y%m%dT%H%M%SZ")]
    return sanitize_run_id("_".join(parts))


def write_outputs(
    config: ServeBenchConfig,
    config_path: Path,
    results: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    command: list[str] | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    actual_run_id = run_id or run_id_for(config)
    output_dir = config.output_root / actual_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "requests.jsonl"
    summary_path = output_dir / "summary.json"
    card_path = output_dir / "serving_card.md"
    manifest_path = output_dir / "manifest.json"

    summary = summarize_results(config, results, output_dir)
    artifacts = {
        "requests_jsonl": "requests.jsonl",
        "summary_json": "summary.json",
        "serving_card_md": "serving_card.md",
        "manifest_json": "manifest.json",
    }
    manifest = build_canonical_manifest(
        run_type="serving",
        status="completed",
        family=config.family,
        variant=config.variant,
        command=command or sys.argv,
        config_paths=[config_path, *config.workload_sources],
        output_dir=output_dir,
        artifacts=artifacts,
        metrics=summary.get("metrics", {}),
        metadata={
            "schema_version": SCHEMA_VERSION,
            "name": config.name,
            "model": config.model,
            "base_url": config.base_url,
            "streaming": config.stream,
            "repetitions": config.repetitions,
            "concurrency": config.concurrency,
            "workload_sources": [display_path(path) for path in config.workload_sources],
            "request_count": len(results),
            "successful_requests": summary.get("successful_requests"),
        },
        notes=list(summary.get("notes") or []),
        run_id=actual_run_id,
    )

    write_jsonl(results_path, results)
    summary_path.write_text(json.dumps(redact_value(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_serving_card(card_path, summary, manifest)
    manifest_path.write_text(json.dumps(redact_value(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_dir, summary, manifest


def build_plan(config: ServeBenchConfig, config_path: Path) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "name": config.name,
        "config": display_path(config_path),
        "family": config.family,
        "variant": config.variant,
        "model": config.model,
        "base_url": config.base_url,
        "streaming": config.stream,
        "timeout_seconds": config.timeout_seconds,
        "repetitions": config.repetitions,
        "concurrency": config.concurrency,
        "request_count": len(config.requests) * config.repetitions,
        "workload_sources": [display_path(path) for path in config.workload_sources],
        "output_root": display_path(config.output_root),
        "dry_run_only": True,
        "notes": [
            "The benchmark expects a compatible server to already be running.",
            "The benchmark does not start vLLM, Docker, torchrun, Ray, or training jobs.",
        ],
    }


def render_plan(plan: Mapping[str, Any]) -> None:
    table = Table(title="Serve Benchmark Plan")
    table.add_column("Field")
    table.add_column("Value")
    for key in ("name", "family", "variant", "model", "base_url", "streaming", "request_count", "workload_sources", "output_root"):
        table.add_row(key, str(plan.get(key)))
    console.print(table)
    for note in plan.get("notes", []):
        console.print(f"- {note}")


def render_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    table = Table(title="Serve Benchmark Summary")
    table.add_column("Metric")
    table.add_column("p50")
    table.add_column("p95")
    table.add_column("mean")
    for metric, values in sorted((summary.get("metrics") or {}).items()):
        table.add_row(
            metric,
            str(values.get("p50", "")),
            str(values.get("p95", "")),
            str(values.get("mean", "")),
        )
    console.print(table)
    console.print(f"Output: {display_path(output_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an already-running OpenAI-compatible serving endpoint")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--family", help="Optional model family used to resolve served_model_name")
    parser.add_argument("--variant", default="base", help="Variant within --family")
    parser.add_argument("--model", help="Served model alias; overrides config/env/family lookup")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL; overrides config/env")
    parser.add_argument("--output-dir", type=Path, help="Output root for benchmark artifacts")
    parser.add_argument("--run-id", help="Stable run id/output subdirectory")
    parser.add_argument("--repetitions", type=int, help="Override workload repetitions")
    parser.add_argument("--limit", type=int, help="Use only the first N configured requests")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming and TTFT measurement")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved benchmark plan without sending requests")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    config_path = resolve_repo_path(args.config)
    config = load_config(
        config_path,
        family=args.family,
        variant=args.variant if args.family else None,
        model=args.model,
        base_url=args.base_url,
        output_dir=args.output_dir,
        repetitions=args.repetitions,
        limit=args.limit,
        stream=False if args.no_stream else None,
    )

    plan = build_plan(config, config_path)
    if args.dry_run:
        if args.json:
            print(json.dumps(redact_value(plan), indent=2, sort_keys=True) + "\n")
        else:
            render_plan(plan)
        return

    results = run_benchmark(config)
    output_dir, summary, _manifest = write_outputs(
        config,
        config_path,
        results,
        run_id=args.run_id,
        command=sys.argv,
    )
    if args.json:
        print(json.dumps(redact_value(summary), indent=2, sort_keys=True) + "\n")
    else:
        render_summary(summary, output_dir)

    if summary["failed_requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
