"""Shared source-metadata validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


OPERATING_CONDITIONS = frozenset(
    {
        "idle",
        "accelerating",
        "decelerating",
        "steady_speed",
        "startup",
        "mixed",
        "unknown",
    }
)


def validate_operating_condition(value: str) -> str:
    condition = value.strip()
    if condition not in OPERATING_CONDITIONS:
        raise ValueError(
            f"unsupported operating condition {condition!r}; "
            f"expected one of {sorted(OPERATING_CONDITIONS)}"
        )
    return condition


@dataclass(frozen=True)
class ConditionSegment:
    operating_condition: str
    start_seconds: float
    end_seconds: float
    notes: str | None = None

    def __post_init__(self) -> None:
        validate_operating_condition(self.operating_condition)
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("condition segment must have 0 <= start_seconds < end_seconds")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ConditionSegment":
        required = ("operating_condition", "start_seconds", "end_seconds")
        missing = [name for name in required if value.get(name) is None]
        if missing:
            raise ValueError(f"condition segment missing required fields: {missing}")
        return cls(
            operating_condition=validate_operating_condition(str(value["operating_condition"])),
            start_seconds=float(value["start_seconds"]),
            end_seconds=float(value["end_seconds"]),
            notes=None if value.get("notes") is None else str(value["notes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

