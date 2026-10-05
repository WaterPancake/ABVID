"""Pre-inference tests: provenance boundaries, locked choices and paired estimators."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from common import HERE, ROOT, load_lock, read_json, save, sha, score, cm_scores, assert_separation
from evaluate_transfer import bootstrap


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder=HERE/'frozen/regularization_transfer_20261004_v1'
        cls.lock,cls.cfg,cls.index=load_lock(cls.folder/'lock.json')
        cls.refs=read_json(cls.folder/'model_references.json')

    def test_all_120_roles_are_frozen_and_target_disjoint(self):
        self.assertEqual(len(self.refs),120);self.assertEqual(len({r['fit_id'] for r in self.refs}),107)
        for r in self.refs:
            self.assertTrue(all(self.index[i]['dataset_id']=='IDMT' for i in r['train_ids']))
            self.assertEqual([sum(self.index[i]['class_id']==c for i in r['train_ids']) for c in (0,1)],[190,190])
            assert_separation(r['train_ids'],self.lock['target_ids'],self.index)
            assert_separation(r['train_ids'],r['source_test_ids'],self.index)

    def test_choices_are_exact_source_nested_choices(self):
        choices={(r['representation'],r['fold'],r['seed']):r for r in read_json(self.folder/'source_selections.json')}
        dl=read_json(ROOT/self.cfg['diagnostic_lock'])
        for r in self.refs:
            expected=1. if r['arm']=='baseline' else choices[r['representation'],r['fold'],r['seed']]['chosen_C']
            self.assertEqual(r['C'],expected)
            for inner in dl['folds'][r['fold']]['inner']:
                assert_separation(inner['validation_ids'],self.lock['target_ids'],self.index)
                assert_separation(inner['train_ids'][str(r['seed'])],self.lock['target_ids'],self.index)

    def test_probability_tie_is_car_and_metrics_use_natural_prior(self):
        y=np.array([0,0,0,1]);p=np.array([.5,.1,.8,.9]);m=score(y,p)
        self.assertEqual(m['confusion_matrix'],[[2,1],[0,1]])
        self.assertEqual(m['truck_prevalence'],.25)
        self.assertAlmostEqual(m['macro_f1'],f1_score(y,p>.5,average='macro'))

    def test_aggregate_is_pooled_target_not_mean_group_f1(self):
        a=np.array([[90,10],[9,1]]);b=np.array([[1,0],[0,1]])
        pooled=cm_scores(a+b)['macro_f1'];equal=np.mean([cm_scores(a)['macro_f1'],cm_scores(b)['macro_f1']])
        self.assertNotAlmostEqual(pooled,equal)

    def test_paired_identical_models_give_zero_gain_intervals(self):
        sc=np.tile(np.array([[4,1],[2,3]]),(2,2,1,1))
        tc=np.tile(sc[:,:,None],(1,1,2,1,1))
        cfg=dict(representations=['toy'],arms=['baseline','nested_C'],uncertainty=dict(draws=64,seed=314159))
        s={('toy',a):sc for a in cfg['arms']};t={('toy',a):tc for a in cfg['arms']}
        result,raw,indices=bootstrap(s,t,cfg)
        for domain in ['source','target']:
            for metric,value in result['paired_gains']['toy'][domain].items():
                self.assertEqual(value,dict(estimate=0.,ci95=[0.,0.]))
        self.assertEqual(indices['target_group'].shape,(64,2))

    def test_class_missing_resamples_are_rejected(self):
        sc=np.tile(np.array([[4,1],[2,3]]),(2,2,1,1))
        tc=np.tile(np.array([[[5,0],[0,0]],[[0,0],[0,5]]]),(2,2,1,1,1))
        cfg=dict(representations=['toy'],arms=['baseline','nested_C'],uncertainty=dict(draws=64,seed=314159))
        result,_,_=bootstrap({('toy',a):sc for a in cfg['arms']},{('toy',a):tc for a in cfg['arms']},cfg)
        self.assertGreater(result['class_missing_draws_rejected'],0)

    def test_mutated_frozen_artifact_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'artifact').write_text('before')
            save(p/'lock.json',dict(artifacts_sha256={'artifact':sha(p/'artifact')}))
            (p/'artifact').write_text('after')
            with self.assertRaises(ValueError):load_lock(p/'lock.json')


if __name__=='__main__':unittest.main(verbosity=2)
