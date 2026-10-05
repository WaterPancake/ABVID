"""Meaningful fail-closed tests for precommitted pairing and split roles."""
import copy
import unittest
from metadata import HERE, ROOT, build, derived_seed, read, rows, validate


class DesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read(HERE/"config.json")
        cls.tables = build(cls.cfg)
        hp = ROOT/cls.cfg["target"]["h1_lock"]
        admitted = {r["file_id"]: r for r in rows(hp.parent/"admitted.jsonl")}
        cls.target = [{**admitted[i], "h2_role": "exposed_development_evaluation_only"} for i in read(hp)["target_ids"]]

    def test_budget_roles_and_full_pairing(self):
        self.assertTrue(validate(self.cfg, self.tables, self.target)["passed"])

    def test_independent_streams_do_not_depend_on_s1_draw_count(self):
        before = derived_seed(self.cfg, 42, "train", "geometry", None, 0, 0)
        cfg = copy.deepcopy(self.cfg); cfg["source"]["S1"]["unused_extra_dimension"] = [0, 1]
        self.assertEqual(before, derived_seed(cfg, 42, "train", "geometry", None, 0, 0))
        self.assertNotEqual(before, derived_seed(cfg, 42, "validation", "geometry", None, 0, 0))

    def test_cross_role_template_leak_fails(self):
        table = copy.deepcopy(self.tables)
        train = next(t for t in table["templates"] if t["role"] == "train")
        val = next(t for t in table["templates"] if t["role"] == "validation")
        val["source_parent_group"] = train["source_parent_group"]
        with self.assertRaisesRegex(ValueError, "crosses roles"): validate(self.cfg, table, self.target)

    def test_class_specific_path_substitution_fails(self):
        table = copy.deepcopy(self.tables)
        table["render_plan"][0]["geometry_id"] = table["geometries"][1]["geometry_id"]
        with self.assertRaisesRegex(ValueError, "event pairing"): validate(self.cfg, table, self.target)

    def test_cell_specific_dry_source_substitution_fails(self):
        table = copy.deepcopy(self.tables)
        table["render_plan"][0]["dry_source_id"] = table["dry_source_plan"][1]["dry_source_id"]
        with self.assertRaisesRegex(ValueError, "Dry source pairing"): validate(self.cfg, table, self.target)

    def test_target_population_change_fails(self):
        with self.assertRaisesRegex(ValueError, "target IDs/count"): validate(self.cfg, self.tables, self.target[:-1])

    def test_per_cell_tuned_head_fails(self):
        table = copy.deepcopy(self.tables)
        table["fit_plan"][0]["classifier"] = {**self.cfg["classifier"], "C": .01}
        with self.assertRaisesRegex(ValueError, "Common head"): validate(self.cfg, table, self.target)

    def test_fake_audio_or_execution_completion_fails(self):
        table = copy.deepcopy(self.tables); table["render_plan"][0]["observation_sha256"] = "fake"
        with self.assertRaisesRegex(ValueError, "False waveform"): validate(self.cfg, table, self.target)
        cfg = copy.deepcopy(self.cfg); cfg["implementation_status"]["ready_for_target_evaluation"] = True
        with self.assertRaisesRegex(ValueError, "not executable"): validate(cfg, self.tables, self.target)


if __name__ == "__main__": unittest.main()
