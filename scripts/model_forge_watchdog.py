#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import time

import psutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kill model-forge workloads when the host is unhealthy.")
    parser.add_argument("--pattern", default=r"train_trl_sft\.py|model_forge\.pipelines\.finetune")
    parser.add_argument("--cpu-percent", type=float, default=95.0)
    parser.add_argument("--mem-percent", type=float, default=96.0)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def matching_processes(pattern: re.Pattern[str]) -> list[psutil.Process]:
    matches = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        if proc.info["pid"] == current_pid:
            continue
        cmdline = " ".join(proc.info.get("cmdline") or [])
        if cmdline and pattern.search(cmdline):
            matches.append(proc)
    return matches


def terminate(processes: list[psutil.Process], dry_run: bool) -> None:
    for proc in processes:
        try:
            print(f"resource watchdog terminating pid={proc.pid} cmd={' '.join(proc.cmdline())}", flush=True)
            if not dry_run:
                proc.send_signal(signal.SIGTERM)
        except psutil.Error as exc:
            print(f"resource watchdog could not terminate pid={proc.pid}: {exc}", flush=True)
    if dry_run:
        return
    gone, alive = psutil.wait_procs(processes, timeout=15)
    for proc in alive:
        try:
            print(f"resource watchdog killing pid={proc.pid}", flush=True)
            proc.kill()
        except psutil.Error as exc:
            print(f"resource watchdog could not kill pid={proc.pid}: {exc}", flush=True)


def main() -> None:
    args = parse_args()
    pattern = re.compile(args.pattern)
    print(
        "model-forge watchdog active "
        f"pattern={args.pattern!r} cpu>{args.cpu_percent}% mem>{args.mem_percent}% interval={args.interval}s",
        flush=True,
    )
    while True:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        if cpu > args.cpu_percent and mem > args.mem_percent:
            procs = matching_processes(pattern)
            if procs:
                terminate(procs, args.dry_run)
            else:
                print("resource watchdog threshold hit, but no matching workload process found", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
