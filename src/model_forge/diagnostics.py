"""Shared diagnostics vocabulary.

Audits across the repo emit "findings" and then decide whether the findings
block (exit non-zero). The finding *shape* is sometimes domain-specific -- a
cluster finding names a node, an upstream finding names a candidate -- but the
severity vocabulary and the "do these findings block?" rule are the same
everywhere, and were duplicated as literal ``any(f.severity == "error" ...)``
expressions in a dozen call sites with the blocking threshold hardcoded.

This module owns that vocabulary: a canonical :class:`Finding` for the common
case, the severity ordering, and the rollup from a list of findings to a
blocking decision / exit code. Domain modules keep their own richer finding
types where they need extra fields; they all route the rollup through here so
the blocking threshold lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# Ordered from least to most severe. Anything at or above ``BLOCKING_SEVERITY``
# makes a finding set blocking.
SEVERITY_ORDER = {
    "info": 0,
    "notice": 1,
    "warning": 2,
    "error": 3,
    "critical": 4,
}
BLOCKING_SEVERITY = "error"

__all__ = [
    "SEVERITY_ORDER",
    "BLOCKING_SEVERITY",
    "Finding",
    "severity_rank",
    "has_errors",
    "severity_exit_code",
    "worst_severity",
    "count_by_severity",
]


@dataclass(frozen=True)
class Finding:
    """A diagnostic with a severity. ``context`` carries domain-specific extras."""

    severity: str
    check: str
    message: str
    path: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)


def severity_rank(severity: str | None) -> int:
    """Numeric rank for ``severity``; unknown severities rank below ``info``."""
    return SEVERITY_ORDER.get(str(severity or ""), -1)


def has_errors(findings: Iterable[Any], *, threshold: str = BLOCKING_SEVERITY) -> bool:
    """True if any finding's severity is at or above ``threshold``.

    Duck-typed on a ``severity`` attribute, so it works for any finding type.
    """
    cutoff = SEVERITY_ORDER.get(threshold, SEVERITY_ORDER[BLOCKING_SEVERITY])
    return any(severity_rank(getattr(f, "severity", None)) >= cutoff for f in findings)


def severity_exit_code(findings: Iterable[Any], *, threshold: str = BLOCKING_SEVERITY) -> int:
    """``1`` if the findings block at ``threshold``, else ``0``."""
    return 1 if has_errors(findings, threshold=threshold) else 0


def worst_severity(findings: Iterable[Any]) -> str | None:
    """The highest-ranked severity present, or ``None`` if there are no findings."""
    severities = [getattr(f, "severity", None) for f in findings]
    severities = [s for s in severities if s]
    if not severities:
        return None
    return max(severities, key=severity_rank)


def count_by_severity(findings: Iterable[Any]) -> dict[str, int]:
    """Count findings grouped by severity string."""
    counts: dict[str, int] = {}
    for f in findings:
        severity = str(getattr(f, "severity", "") or "")
        counts[severity] = counts.get(severity, 0) + 1
    return counts
