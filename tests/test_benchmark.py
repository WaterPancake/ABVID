from pathlib import Path

from vehicle_audio.benchmark import audit_benchmark


ROOT = Path(__file__).parents[1]


def test_benchmark_v0_1_roles_are_disjoint_and_development_count_gate_passes() -> None:
    result = audit_benchmark(ROOT / "configs" / "benchmark_v0_1.yaml")

    assert result["benchmark_id"] == "abvid-benchmark-v0.1"
    assert result["development"]["recording_session_counts"] == {
        "tracked": 7,
        "wheeled": 5,
    }
    assert result["development"]["data_gate_passed"] is True
    assert result["protected_roles"]["role_audit_passed"] is True
    assert result["ready_for_refreshed_development_evaluation"] is True
    assert result["protected_roles"]["classes_by_role"] == {
        "consumed_locked": ["tracked", "wheeled"],
        "future_confirmation": ["tracked", "wheeled"],
    }
