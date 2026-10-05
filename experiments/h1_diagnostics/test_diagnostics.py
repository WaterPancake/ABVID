"""Protocol boundary tests, using metadata and artificial scores only."""
import unittest
from pathlib import Path

import numpy as np
from common import HERE, read_lock, source_path, assert_separation
from run_checks import score, choose_C, choose_threshold


class DiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock, cls.cfg, cls.rows = read_lock(HERE/'frozen/source_diagnostics_20261003_v1/lock.json')
        cls.index = {r['file_id']: r for r in cls.rows}

    def test_reject_target_and_escaped_path(self):
        row = dict(self.rows[0])
        for change in [dict(dataset_id='MELAUDIS'), dict(class_id=2),
                       dict(source_path='dataset/IDMT_Traffic/audio/../../MELAUDIS/a.wav')]:
            with self.assertRaises(ValueError): source_path(dict(row, **change))

    def test_nested_selection_never_sees_outer_group(self):
        for fold in self.lock['folds']:
            for inner in fold['inner']:
                assert_separation(inner['validation_ids'], fold['test_ids'], self.index)
                for ids in inner['train_ids'].values():
                    assert_separation(ids, inner['validation_ids'], self.index)
                    assert_separation(ids, fold['test_ids'], self.index)
                    self.assertEqual([sum(self.index[i]['class_id'] == c for i in ids) for c in (0, 1)], [190, 190])

    def test_learning_samples_are_classwise_prefixes(self):
        for fold in self.lock['folds']:
            for seed in self.cfg['seeds']:
                full = fold['train_ids']['359'][str(seed)]
                for n in self.cfg['learning_curve_examples_per_class']:
                    ids = fold['train_ids'][str(n)][str(seed)]
                    for c in (0, 1):
                        self.assertEqual([i for i in ids if self.index[i]['class_id'] == c],
                                         [i for i in full if self.index[i]['class_id'] == c][:n])
                    assert_separation(ids, fold['test_ids'], self.index)

    def test_whole_location_is_excluded(self):
        for site in self.lock['locations']:
            self.assertEqual({self.index[i]['site_id'] for i in site['test_ids']}, {site['site']})
            for ids in site['train_ids'].values():
                self.assertNotIn(site['site'], {self.index[i]['site_id'] for i in ids})
                assert_separation(ids, site['test_ids'], self.index)

    def test_selection_tie_rules(self):
        inner = [dict(y=np.array([0, 1]), p=np.array([0., 1.]))]
        C, _ = choose_C({str(c): inner for c in [.01, .1, 1.]}, [.01, .1, 1.])
        self.assertEqual(C, .01)
        t, _ = choose_threshold(inner, [.1, .4, .5, .6, .9]); self.assertEqual(t, .5)
        t, _ = choose_threshold(inner, [.4, .6]); self.assertEqual(t, .4)

    def test_selection_is_mean_of_groups_not_pooled_clips(self):
        one = dict(y=np.array([0, 1]), p=np.array([0., 1.]))
        many_bad = dict(y=np.tile([0, 1], 30), p=np.tile([1., 0.], 30))
        _, values = choose_C({'1.0': [one, many_bad]}, [1.])
        self.assertEqual(values['1.0'], .5)

    def test_threshold_changes_only_decisions(self):
        y = np.array([0, 0, 1, 1]); p = np.array([.1, .6, .5, .9])
        a, b = score(y, p, .5), score(y, p, .7)
        for key in ['roc_auc', 'average_precision', 'brier_truck', 'log_loss', 'ece10_truck']:
            self.assertEqual(a[key], b[key])
        self.assertEqual(a['confusion_matrix'], [[1, 1], [1, 1]])
        self.assertNotEqual(a['macro_f1'], b['macro_f1'])

    def test_duplicate_group_fails_closed(self):
        i = self.rows[0]['file_id']
        with self.assertRaises(ValueError): assert_separation([i], [i], self.index)


if __name__ == '__main__': unittest.main(verbosity=2)
