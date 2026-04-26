from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TOOLS = {
    "lm-eval": {
        "commands": [["lm_eval"], [sys.executable, "-m", "lm_eval"]],
        "module": "lm_eval",
        "purpose": "broad static capability benchmarks",
    },
    "lighteval": {
        "commands": [["lighteval"], [sys.executable, "-m", "lighteval"]],
        "module": "lighteval",
        "purpose": "Hugging Face-oriented benchmark execution",
    },
    "inspect": {
        "commands": [["inspect"], [sys.executable, "-m", "inspect_ai"]],
        "module": "inspect_ai",
        "purpose": "agent/tool/sandbox evaluations",
    },
    "promptfoo": {
        "commands": [["promptfoo"]],
        "purpose": "prompt regression and red-team checks",
    },
}


def command_available(tool: str, command: list[str]) -> bool:
    module = TOOLS[tool].get("module")
    if command[0] == sys.executable and module:
        return importlib.util.find_spec(module) is not None
    if command[0] == sys.executable:
        proc = subprocess.run([*command, "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
        return proc.returncode in {0, 1, 2}
    return shutil.which(command[0]) is not None


def find_tool_command(tool: str) -> list[str] | None:
    for command in TOOLS[tool]["commands"]:
        if command_available(tool, command):
            return command
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge model-forge runs to external benchmark tools")
    parser.add_argument("tool", choices=sorted(TOOLS))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/external"))
    parser.add_argument("--dry-run", action="store_true", help="Only record availability and intended command")
    args, tool_args = parser.parse_known_args()

    if tool_args and tool_args[0] == "--":
        tool_args = tool_args[1:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = find_tool_command(args.tool)
    metadata = {
        "tool": args.tool,
        "purpose": TOOLS[args.tool]["purpose"],
        "available": command is not None,
        "command": command,
        "tool_args": tool_args,
        "dry_run": args.dry_run,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if command is None:
        metadata["error"] = f"{args.tool} is not installed or not on PATH"
        (args.output_dir / "external_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(json.dumps(metadata, indent=2))
        sys.exit(0 if args.dry_run else 2)

    if not args.dry_run:
        proc = subprocess.run([*command, *tool_args], text=True, capture_output=True, check=False)
        metadata["returncode"] = proc.returncode
        (args.output_dir / "stdout.txt").write_text(proc.stdout)
        (args.output_dir / "stderr.txt").write_text(proc.stderr)
        if proc.returncode != 0:
            metadata["error"] = f"{args.tool} exited with {proc.returncode}"
    (args.output_dir / "external_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    if metadata.get("returncode", 0) != 0:
        sys.exit(int(metadata["returncode"]))


if __name__ == "__main__":
    main()
