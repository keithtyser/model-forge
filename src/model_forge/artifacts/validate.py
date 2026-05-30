from __future__ import annotations

import argparse
import fnmatch
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from rich.console import Console
from rich.table import Table

from model_forge.evals.run_eval import validate_artifact
from model_forge.runs.manifest import REPO_DIR, display_path, redact_value, sanitize_run_id


SCHEMA_VERSION = "model_forge.artifact_execution_card.v1"
DEFAULT_OUTPUT_DIR = REPO_DIR / "reports" / "generated" / "artifact_validation"
SUPPORTED_EXTENSIONS = {
    ".html": "html",
    ".htm": "html",
    ".py": "python",
}

console = Console(stderr=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return REPO_DIR / candidate


def infer_artifact_type(path: Path) -> str | None:
    return SUPPORTED_EXTENSIONS.get(path.suffix.lower())


def discover_artifacts(paths: list[Path], *, recursive: bool, include_unsupported: bool) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_file():
            candidates = [path]
        elif path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            candidates = [candidate for candidate in iterator if candidate.is_file()]
        else:
            raise FileNotFoundError(f"artifact path does not exist: {display_path(path)}")

        for candidate in candidates:
            if not include_unsupported and infer_artifact_type(candidate) is None:
                continue
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(candidate)
    return sorted(discovered, key=lambda item: str(item))


def load_checks_config(path: Path | None) -> dict[str, Any]:
    if not path:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping in {display_path(path)}")
    return data


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def checks_for(path: Path, artifact_type: str | None, checks_config: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if artifact_type and isinstance(checks_config.get(artifact_type), Mapping):
        config = merge_dicts(config, checks_config[artifact_type])
    files = checks_config.get("files") or {}
    if isinstance(files, Mapping):
        rel = display_path(path)
        for pattern, value in files.items():
            if isinstance(value, Mapping) and (fnmatch.fnmatch(path.name, str(pattern)) or fnmatch.fnmatch(rel, str(pattern))):
                config = merge_dicts(config, value)
    return config


def enforce_browser_requirement(validation: dict[str, Any], *, require_browser: bool) -> dict[str, Any]:
    if not require_browser or validation.get("type") != "html":
        return validation
    browser = validation.get("browser") or {}
    if not browser.get("skipped"):
        return validation
    updated = dict(validation)
    errors = list(updated.get("errors") or [])
    errors.append("browser_validation_skipped")
    updated["ok"] = False
    updated["errors"] = errors
    return updated


def compact_validation(path: Path, artifact_type: str, validation: Mapping[str, Any]) -> dict[str, Any]:
    browser = validation.get("browser") if isinstance(validation.get("browser"), Mapping) else {}
    screenshot = browser.get("screenshot_path") if isinstance(browser, Mapping) else None
    screenshot_path = str(path.with_name(str(screenshot))) if screenshot else None
    return {
        "path": display_path(path),
        "type": artifact_type,
        "ok": bool(validation.get("ok")),
        "errors": list(validation.get("errors") or []),
        "checks": validation.get("checks") or {},
        "browser": browser,
        "python": {
            "fixture": validation.get("fixture"),
            "details": validation.get("details"),
        } if artifact_type == "python" else None,
        "screenshot_path": display_path(screenshot_path) if screenshot_path else None,
    }


def rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)


def metric_bool(checks: Mapping[str, Any], key: str) -> bool | None:
    value = checks.get(key)
    return bool(value) if isinstance(value, bool) else None


def summarize_artifacts(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(artifacts)
    passed = sum(1 for item in artifacts if item["ok"])
    failed = total - passed
    html_items = [item for item in artifacts if item["type"] == "html"]
    python_items = [item for item in artifacts if item["type"] == "python"]
    unsupported_items = [item for item in artifacts if item["type"] == "unsupported"]

    python_compiles = 0
    python_runs = 0
    for item in python_items:
        checks = item.get("checks") or {}
        fixture = ((item.get("python") or {}).get("fixture") or {})
        if checks.get("compiles"):
            python_compiles += 1
        fixture_ok = fixture.get("ok")
        if fixture.get("skipped"):
            fixture_ok = True
        if checks.get("help_exits_cleanly") and fixture_ok is not False:
            python_runs += 1

    browser_checked = []
    browser_skipped = 0
    screenshots = 0
    html_console_errors = 0
    nonblank_render = 0
    for item in html_items:
        browser = item.get("browser") or {}
        if browser.get("skipped"):
            browser_skipped += 1
            continue
        browser_checked.append(item)
        if item.get("screenshot_path"):
            screenshots += 1
        checks = browser.get("checks") or {}
        if checks and checks.get("console_error_free") is False:
            html_console_errors += 1
        visible_checks = [
            value
            for key, value in checks.items()
            if key.endswith("_dom_has_visible_text") or key.endswith("_canvas_nonblank")
        ]
        if visible_checks and all(bool(value) for value in visible_checks):
            nonblank_render += 1

    manual_review = failed + browser_skipped + len(unsupported_items)
    return {
        "artifact_count": total,
        "passed_count": passed,
        "failed_count": failed,
        "html_count": len(html_items),
        "python_count": len(python_items),
        "unsupported_count": len(unsupported_items),
        "browser_checked_count": len(browser_checked),
        "browser_skipped_count": browser_skipped,
        "screenshot_count": screenshots,
        "manual_review_required_count": manual_review,
        "metrics": {
            "artifact_execution_pass_rate": rate(passed, total),
            "artifact_compiles_rate": rate(python_compiles, len(python_items)),
            "artifact_runs_rate": rate(python_runs, len(python_items)),
            "html_console_error_rate": rate(html_console_errors, len(browser_checked)),
            "nonblank_render_rate": rate(nonblank_render, len(browser_checked)),
        },
    }


def write_card_markdown(path: Path, card: Mapping[str, Any]) -> None:
    summary = card.get("summary") or {}
    metrics = summary.get("metrics") or {}
    lines = [
        f"# Artifact Execution Card: {card.get('run_id')}",
        "",
        "## Summary",
        "",
        f"- Artifact count: `{summary.get('artifact_count')}`",
        f"- Passed: `{summary.get('passed_count')}`",
        f"- Failed: `{summary.get('failed_count')}`",
        f"- Browser checked: `{summary.get('browser_checked_count')}`",
        f"- Browser skipped: `{summary.get('browser_skipped_count')}`",
        f"- Manual review required: `{summary.get('manual_review_required_count')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "## Artifacts", "", "| Artifact | Type | Result | Errors | Screenshot |", "|---|---|---|---|---|"])
    for item in card.get("artifacts") or []:
        result = "pass" if item.get("ok") else "fail"
        errors = ", ".join(item.get("errors") or [])
        screenshot = item.get("screenshot_path") or ""
        lines.append(f"| `{item.get('path')}` | `{item.get('type')}` | {result} | {errors} | `{screenshot}` |")
    lines.extend(["", "## Notes", ""])
    for note in card.get("notes") or []:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_paths(
    paths: list[Path],
    *,
    output_dir: Path,
    run_id: str,
    checks_config: Mapping[str, Any] | None = None,
    recursive: bool = True,
    include_unsupported: bool = False,
    require_browser: bool = False,
) -> dict[str, Any]:
    checks_config = checks_config or {}
    artifacts = []
    for path in discover_artifacts(paths, recursive=recursive, include_unsupported=include_unsupported):
        artifact_type = infer_artifact_type(path) or "unsupported"
        if artifact_type == "unsupported":
            validation = {
                "ok": False,
                "type": "unsupported",
                "checks": {},
                "errors": ["unsupported_artifact_type"],
                "details": {"suffix": path.suffix},
            }
        else:
            validation = validate_artifact(path, artifact_type, checks_for(path, artifact_type, checks_config))
            validation = enforce_browser_requirement(validation, require_browser=require_browser)
        artifacts.append(compact_validation(path, artifact_type, validation))

    summary = summarize_artifacts(artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)
    card = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now().isoformat(),
        "run_id": run_id,
        "input_paths": [display_path(path) for path in paths],
        "output_dir": display_path(output_dir),
        "checks_config": redact_value(checks_config),
        "require_browser": require_browser,
        "recursive": recursive,
        "summary": summary,
        "artifacts": artifacts,
        "notes": [
            "This validates extracted artifact files; it does not prove the model generated them.",
            "Promotion claims should connect this card to source/candidate eval or serving manifests.",
            "Browser-skipped HTML validation is acceptable for local smoke checks only; require browser validation for artifact-generation claims.",
        ],
    }
    (output_dir / "artifact_validations.json").write_text(json.dumps(artifacts, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "artifact_execution_card.json").write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_card_markdown(output_dir / "artifact_execution_card.md", card)
    return card


def default_run_id(paths: list[Path]) -> str:
    label = paths[0].name if paths else "artifacts"
    return sanitize_run_id(f"{label}_artifact_validation_{utc_now().strftime('%Y%m%dT%H%M%SZ')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate generated HTML and Python artifacts")
    sub = parser.add_subparsers(dest="command")
    validate = sub.add_parser("validate", help="Validate one artifact file or an artifact directory")
    validate.add_argument("paths", nargs="+", type=Path)
    validate.add_argument("--output-dir", type=Path, default=None)
    validate.add_argument("--run-id", default=None)
    validate.add_argument("--checks-config", type=Path, default=None)
    validate.add_argument("--no-recursive", action="store_true")
    validate.add_argument("--include-unsupported", action="store_true")
    validate.add_argument("--require-browser", action="store_true")
    validate.add_argument("--strict", action="store_true", help="Exit nonzero if any artifact fails validation")
    validate.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "validate":
        parser.print_help()
        return 2

    paths = [resolve_path(path) for path in args.paths]
    run_id = args.run_id or default_run_id(paths)
    output_dir = resolve_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR / run_id
    checks_config = load_checks_config(resolve_path(args.checks_config) if args.checks_config else None)
    card = validate_paths(
        paths,
        output_dir=output_dir,
        run_id=run_id,
        checks_config=checks_config,
        recursive=not args.no_recursive,
        include_unsupported=args.include_unsupported,
        require_browser=args.require_browser,
    )

    if args.json:
        print(json.dumps(card, indent=2, sort_keys=True))
    else:
        summary = card["summary"]
        table = Table(title="Artifact Validation")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("artifacts", str(summary["artifact_count"]))
        table.add_row("passed", str(summary["passed_count"]))
        table.add_row("failed", str(summary["failed_count"]))
        table.add_row("browser checked", str(summary["browser_checked_count"]))
        table.add_row("browser skipped", str(summary["browser_skipped_count"]))
        table.add_row("card", display_path(output_dir / "artifact_execution_card.md"))
        console.print(table)

    if args.strict and card["summary"]["failed_count"]:
        return 1
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    raise SystemExit(main())
