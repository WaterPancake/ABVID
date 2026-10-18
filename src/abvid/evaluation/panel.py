"""Per-dataset target panel: one row per dataset role, never a pooled score.

Each row carries its own support, exposure status and metrics with intervals.
Sealed, pending and planned rows must not carry metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

ROLES = {
    "source_within_domain": "grouped held-out source data (already exposed)",
    "exposed_development": "previously scored target; development evidence only",
    "sealed_confirmation": "admitted from metadata and sealed; not scored",
    "pending_audit": "requires overlap/provenance audit before any role",
    "planned_collection": "not yet recorded",
}
SCORED = {"source_within_domain", "exposed_development"}


@dataclass(frozen=True)
class Entry:
    dataset_id: str
    role: str
    task: str
    groups: int | None
    support: dict
    metrics: dict | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def validate(entries):
    seen = set()
    for e in entries:
        if e.role not in ROLES:
            raise ValueError(f"Unknown panel role: {e.role}")
        if any(token in e.dataset_id.lower() for token in ("+", "pooled", "combined")):
            raise ValueError("Panel rows are per dataset; pooled targets are not allowed")
        if (e.dataset_id, e.role, e.task) in seen:
            raise ValueError(f"Duplicate panel row: {e.dataset_id}/{e.role}/{e.task}")
        seen.add((e.dataset_id, e.role, e.task))
        if (e.metrics is not None) != (e.role in SCORED):
            raise ValueError(f"{e.dataset_id}: metrics are allowed only for scored roles")
    return entries


def _cell(value):
    if value is None:
        return "—"
    point, (lo, hi) = value["estimate"], value["ci95"]
    if lo is None:
        return f"{100 * point:.2f}"
    return f"{100 * point:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]"


def markdown(entries, columns):
    """Render rows; ``columns`` maps metric keys to headers (percent scale)."""
    validate(entries)
    head = ["Dataset", "Role", "Task", "Groups", "Support (neg/pos)", *columns.values()]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in entries:
        support = f"{e.support.get('negative', '—')}/{e.support.get('positive', '—')}"
        cells = [_cell((e.metrics or {}).get(k)) if e.metrics else "sealed" if e.role == "sealed_confirmation"
                 else "—" for k in columns]
        lines.append("| " + " | ".join([e.dataset_id, e.role, e.task, str(e.groups or "—"),
                                         support, *cells]) + " |")
    return "\n".join(lines)
