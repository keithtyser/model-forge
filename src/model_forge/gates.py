"""Shared gate primitives for promotion / release evidence.

Two gate shapes recur across the pipelines, each with its own rollup that was
copy-pasted at every call site:

1. *Required checks* (used by quantization and serving evidence reports): a list
   of ``{name, status, required, message}`` dicts where ``status`` is
   ``pass``/``fail``/``missing`` and the artifact is ready when every required
   check passes. ``status_check``/``evidence_status`` were byte-identical
   functions in two modules and ``all(check["status"] == "pass" ...)`` was
   written six times.

2. *Blocking gates* (used by the Hub publish planner): a list of
   ``Gate(name, status, message)`` where ``status`` is ``pass``/``fail``/``warn``
   and the plan is blocked when any gate failed. The ``[g for g in gates if
   g.status == "fail"]`` filter was duplicated for model and dataset plans.

This module owns both shapes so the "is this blocked / ready?" decision lives in
one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

__all__ = [
    "check",
    "all_required_pass",
    "Gate",
    "gate_status",
    "optional_gate",
    "failing_gates",
    "blocked_until",
]


# --- required-check gates (dict form) ---------------------------------------


def check(name: str, passed: bool, message: str, *, required: bool = True) -> dict[str, Any]:
    """A required-check gate. Unmet optional checks are ``missing``, not ``fail``."""
    return {
        "name": name,
        "status": "pass" if passed else ("fail" if required else "missing"),
        "required": required,
        "message": message,
    }


def all_required_pass(checks: Iterable[Mapping[str, Any]]) -> bool:
    """True when every required check passed (optional/missing checks ignored)."""
    return all(c["status"] == "pass" for c in checks if c["required"])


# --- blocking gates (Gate form) ---------------------------------------------


@dataclass(frozen=True)
class Gate:
    name: str
    status: str
    message: str


def gate_status(name: str, passed: bool, message: str) -> Gate:
    """A blocking gate: ``pass`` or ``fail``."""
    return Gate(name=name, status="pass" if passed else "fail", message=message)


def optional_gate(name: str, passed: bool, message: str) -> Gate:
    """A non-blocking gate: ``pass`` or ``warn``."""
    return Gate(name=name, status="pass" if passed else "warn", message=message)


def failing_gates(gates: Iterable[Gate]) -> list[Gate]:
    """The gates that block (``status == "fail"``)."""
    return [gate for gate in gates if gate.status == "fail"]


def blocked_until(gates: Iterable[Gate]) -> list[str]:
    """Human-readable ``"name: message"`` reasons for each blocking gate."""
    return [f"{gate.name}: {gate.message}" for gate in failing_gates(gates)]
