import collections
import json
from pathlib import Path

import numpy as np
import pytest

from abvid.cast.utility_arms import PER_CLASS, RU_SPACING_HZ, rc_schedule, ru_parameters
from abvid.learning.utility import background_pool, head_metrics, ra_copy, training_rows


def test_rc_schedule_is_seed_major_first_190_v15_joint_entries():
    schedule = rc_schedule("utility_s42_0")
    assert len(schedule) == 2 * PER_CLASS and all(e["arm"] == "v15_joint" for e in schedule)
    counts = collections.Counter((e["class"], e["seed"]) for e in schedule)
    for label in ("car", "truck"):
        assert [counts[(label, s)] for s in (42, 123, 456, 789, 1024)] == [50, 50, 50, 40, 0]
    # Streams are fold-independent; identities carry the partition.
    other = rc_schedule("utility_s123_4")
    assert [e["input_id"] for e in other] == [e["input_id"] for e in schedule]
    assert all(e["partition"] == "utility_s123_4" for e in other)


@pytest.mark.parametrize("label", ["car", "truck"])
def test_ru_parameters_are_deterministic_and_within_declared_ranges(label):
    a, b = ru_parameters(label, 42, 7), ru_parameters(label, 42, 7)
    assert a == b and a[0] == f"cast5a_ru_v1:{label}:42:0007"
    p = a[1]
    lo, hi = RU_SPACING_HZ[label]
    assert all(lo * np.exp(-0.1) - 1e-9 <= s <= hi * np.exp(0.1) + 1e-9 for s in p["spacing_hz"])
    assert len(p["harmonic_weights"]) == 8 and abs(sum(p["harmonic_weights"]) - 1) < 1e-12
    assert len(p["noise_weights"]) == 16 and abs(sum(p["noise_weights"]) - 1) < 1e-12
    assert len(p["envelope_knots"]) == 41 and 0.1 <= min(p["envelope_knots"]) <= max(p["envelope_knots"]) <= 3.0
    assert ru_parameters(label, 42, 8) != a and ru_parameters(label, 123, 7) != a


def test_backgrounds_come_only_from_training_groups(tmp_path):
    import soundfile as sf
    names = {"2019-11-12-07-00_SiteA_50Kmh_1_SE_CH34-BG.wav": "gA",
             "2019-11-13-07-00_SiteA_50Kmh_2_SE_CH34-BG.wav": "gB",
             "2019-11-14-07-00_SiteC_50Kmh_3_SE_CH34-BG.wav": None}
    for name in names:
        sf.write(tmp_path / name, np.random.default_rng(len(name)).standard_normal((96000, 2)) * 0.1, 48000)
    sources = [{"site_id": "SiteA", "calendar_date": "2019-11-12", "provenance_group_id": "gA"},
               {"site_id": "SiteA", "calendar_date": "2019-11-13", "provenance_group_id": "gB"}]
    pool = background_pool(tmp_path, sources)
    assert {g: [p.name for p in v] for g, v in pool.items()} == {
        "gA": ["2019-11-12-07-00_SiteA_50Kmh_1_SE_CH34-BG.wav"], "gB": ["2019-11-13-07-00_SiteA_50Kmh_2_SE_CH34-BG.wav"]}

    class Fixed:
        def get(self, path):
            return np.random.default_rng(1).standard_normal(32000), "digest"
    x = np.sin(np.arange(32000) / 10)
    for _ in range(5):
        wave, record = ra_copy("utility_s42_0", "file", x, {"gB": 3}, pool, Fixed())
        assert record["background"].startswith("2019-11-13") and wave.shape == (32000,)
    again, record_again = ra_copy("utility_s42_0", "file", x, {"gB": 3}, pool, Fixed())
    np.testing.assert_array_equal(wave, again)
    assert record == record_again
    with pytest.raises(ValueError, match="training groups"):
        ra_copy("utility_s42_0", "file", x, {"gZ": 1}, pool, Fixed())


def test_training_rows_enforce_protocol_budgets():
    def rows(n):
        return np.zeros((2 * n, 3)), np.repeat([0, 1], n)
    real, extra = rows(190), rows(190)
    for arm, n in {"R": 190, "RR": 380, "RA": 380, "RU": 380, "RC": 380, "U": 190, "C": 190}.items():
        x, y = training_rows(arm, real, extra, extra, extra)
        assert np.bincount(y).tolist() == [n, n] and len(x) == 2 * n
    with pytest.raises(ValueError):
        training_rows("RC", real, extra, extra, rows(150))
    with pytest.raises(ValueError):
        training_rows("X", real, extra, extra, extra)


def test_head_metrics_match_direct_computation():
    rng = np.random.default_rng(3)
    y = (rng.random(400) < 0.12).astype(int)
    p = np.clip(0.3 * y + rng.random(400) * 0.6, 0, 1)
    m = head_metrics(y, p)
    pred = p > 0.5
    assert m["positive_recall"] == pytest.approx(pred[y == 1].mean())
    assert m["negative_recall"] == pytest.approx((~pred[y == 0]).mean())
    assert sum(map(sum, m["confusion"])) == 400


@pytest.mark.archive
def test_frozen_design_reconstructs_and_counts():
    from abvid.cast.utility_design import design_payload, load_context
    payload = design_payload()
    design = json.loads(payload["design.json"])
    assert (design["union_parents"], design["from_cast4a_run"], design["new_fits"]) == (3245, 1749, 1496)
    context = load_context()
    assert len(context["roles"]) == 30
    for role in context["roles"].values():
        assert len(role.train_ids) == 380 and set(role.held_groups).isdisjoint(role.training_groups)
    assert Path(context["output"]).name == "cast5a_run_20261011_v1"


def synthetic_heads(effect=0.0, missing=None):
    from abvid.learning.utility_stats import SEEDS
    parts, heads = [], []
    for seed in SEEDS:
        for fold in range(6):
            pid = f"utility_s{seed}_{fold}"
            parts.append({"id": pid, "analysis": f"utility_s{seed}", "fold": fold})
            for rep in ("BEATs768", "MFCC26"):
                for k, arm in enumerate(("R", "RR", "RA", "RU", "RC", "U", "C")):
                    if (rep, arm, pid) == missing:
                        continue
                    base = 0.5 + 0.01 * fold + 0.002 * k
                    value = base + (effect if arm == "RC" else 0.0)
                    heads.append({"partition": pid, "representation": rep, "arm": arm,
                                  **{m: value for m in ("macro_f1", "balanced_accuracy", "auroc",
                                                        "average_precision", "positive_recall", "negative_recall")}})
    return parts, heads


FOLD_SITE = {0: "a", 1: "a", 2: "b", 3: "b", 4: "c", 5: "c"}


def test_statistics_point_estimates_and_decision_flags():
    from abvid.learning.utility_stats import statistics
    parts, heads = synthetic_heads(effect=0.10)
    s = statistics(heads, parts, FOLD_SITE, draws=500)
    rc_ra = s["contrasts"]["BEATs768"]["RC-RA"]["macro_f1"]
    assert rc_ra["estimate"] == pytest.approx(0.10 + 0.004)
    assert s["flags"]["primary_supports_practical_gain"] and not s["flags"]["primary_inconclusive"]
    assert s["arms"]["BEATs768"]["R"]["macro_f1"]["estimate"] == pytest.approx(0.525)
    parts, heads = synthetic_heads(effect=-0.004)
    s = statistics(heads, parts, FOLD_SITE, draws=500)
    assert s["flags"]["primary_rules_out_practical_gain"] and not s["flags"]["any_gain_lower_bound_above_zero"]


def test_statistics_refuse_subset_intervals_when_a_head_is_missing():
    from abvid.learning.utility_stats import statistics
    parts, heads = synthetic_heads(effect=0.1, missing=("BEATs768", "RC", "utility_s42_0"))
    s = statistics(heads, parts, FOLD_SITE, draws=200)
    assert s["arms"]["BEATs768"]["RC"]["macro_f1"] is None
    assert s["contrasts"]["BEATs768"]["RC-RA"]["macro_f1"] is None
    assert s["flags"] == {"primary": "incomplete; no subset-only interval"}


def test_cast4a_context_is_built_once_per_5a_context(monkeypatch):
    """Regression: dict.setdefault evaluated the pinned 4A loader for every parent."""
    from abvid.cast import grouped_parents, utility_design
    calls = []
    monkeypatch.setattr(utility_design, "_cast4a_context", lambda: calls.append(1) or {"roles": "4a"})
    monkeypatch.setattr(grouped_parents, "_witness", lambda ctx, fid: ("role", ctx["roles"]))
    monkeypatch.setattr(grouped_parents, "load_parent", lambda ctx, fid, role: (fid, role))
    context = {"parents": {"from_cast4a_run": ["a", "b", "c"]}, "roles": "5a"}
    rows = [utility_design.load_parent_rows(context, fid) for fid in ("a", "b", "x", "c")]
    assert len(calls) == 1
    assert rows == [("a", ("role", "4a")), ("b", ("role", "4a")), ("x", ("role", "5a")), ("c", ("role", "4a"))]
