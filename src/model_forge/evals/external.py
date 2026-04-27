from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {sec:02d}s"
    if minutes:
        return f"{minutes}m {sec:02d}s"
    return f"{sec}s"


def color(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def cyan(text: str) -> str:
    return color("36", text)


def green(text: str) -> str:
    return color("32", text)


def run_with_tee(command: list[str], output_dir: Path) -> int:
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    started = time.perf_counter()
    print(f"{cyan('...')} external runner started: {' '.join(command[:2])}", flush=True)
    proc = subprocess.Popen(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )

    def pump(stream: Any, chunks: list[str], target: Any) -> None:
        try:
            for line in stream:
                chunks.append(line)
                target.write(line)
                target.flush()
        finally:
            stream.close()

    stdout_thread = threading.Thread(target=pump, args=(proc.stdout, stdout_chunks, sys.stdout), daemon=True)
    stderr_thread = threading.Thread(target=pump, args=(proc.stderr, stderr_chunks, sys.stderr), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    while proc.poll() is None:
        print(
            f"{cyan('...')} external runner still running | elapsed {format_duration(time.perf_counter() - started)}",
            flush=True,
        )
        time.sleep(60)
    stdout_thread.join()
    stderr_thread.join()
    (output_dir / "stdout.txt").write_text("".join(stdout_chunks))
    (output_dir / "stderr.txt").write_text("".join(stderr_chunks))
    print(
        f"{green('OK')} external runner finished in {format_duration(time.perf_counter() - started)} "
        f"with exit code {proc.returncode}",
        flush=True,
    )
    return int(proc.returncode or 0)


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
        returncode = run_with_tee([*command, *tool_args], args.output_dir)
        metadata["returncode"] = returncode
        if returncode != 0:
            metadata["error"] = f"{args.tool} exited with {returncode}"
    (args.output_dir / "external_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print()
    status = green("OK") if metadata.get("returncode", 0) == 0 and command is not None else "ERROR"
    print(f"{status} External benchmark complete")
    print(f"  tool:    {args.tool}")
    print(f"  output:  {args.output_dir}")
    print(f"  command: {' '.join(command or [])}")
    if metadata.get("returncode", 0) != 0:
        sys.exit(int(metadata["returncode"]))


if __name__ == "__main__":
    main()
