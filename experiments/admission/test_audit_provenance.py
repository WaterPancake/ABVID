import unittest
import numpy as np
from scipy.signal import resample_poly
from audit_provenance import Union, normalized_center, parent_proxy, overlapping_components, verified_pairs


class ProvenanceTests(unittest.TestCase):
    def test_normalization_gain_and_redundant_channels(self):
        x = np.random.default_rng(3).normal(size=32000)
        a = normalized_center(x, 16000)
        b = normalized_center(np.stack([x, x], axis=1) * 3, 16000)
        np.testing.assert_allclose(a, b, atol=1e-7)
        c = normalized_center(resample_poly(x, 3, 1), 48000)
        self.assertGreater(float(np.dot(a, c)), .9999)
        self.assertIsNone(normalized_center(np.zeros(32000), 16000))

    def test_unknown_location_aliases_overmerge_by_day(self):
        a = dict(dataset_id='MELAUDIS', calendar_date='2024-02-09', site_id='Swanston1')
        b = dict(a, site_id='Swanston10')
        self.assertEqual(parent_proxy(a), parent_proxy(b))

    def test_original_recording_overlap_is_connected(self):
        rows = [dict(dataset_id='IDMT', sample_position_center=t * 48000,
                     native_sample_rate_hz=48000, candidate_recording_id='same',file_id=str(t))
                for t in (1, 2, 3, 9)]
        cs = overlapping_components(rows)
        self.assertEqual(cs, [dict(parent_time_axis='same',members=['1','2','3'])])

    def test_duplicate_can_bridge_excluded_class_and_dates(self):
        u = Union(['a','b','c','d'])
        u.join('a','b'); u.join('c','d'); u.join('b','c')
        self.assertEqual(u.find('a'), u.find('d'))

    def test_cross_dataset_polarity_reuse_is_found(self):
        waves = np.random.default_rng(4).normal(size=(20, 4000)).astype(np.float32)
        waves /= np.linalg.norm(waves, axis=1, keepdims=True)
        waves[11] = -waves[0]
        pairs, _ = verified_pairs(waves, ['A'] * 10 + ['B'] * 10, list(range(20)))
        self.assertIn((0,11), [(a,b) for a,b,_ in pairs])
        self.assertEqual(len(pairs), 1)


if __name__ == '__main__':
    unittest.main()
