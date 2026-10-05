import unittest
import numpy as np
from freeze import select,assert_separation
from extract import observation,mfcc_features
from evaluate import f1,cm_metrics,fixed_fit,bootstrap


class H1Tests(unittest.TestCase):
    def test_round_robin_and_determinism(self):
        rows=[dict(file_id=f'{g}-{c}-{i}',provenance_group_id=g,class_id=c)
              for g,n in [('large',100),('small',3)] for c in (0,1) for i in range(n)]
        a=select(rows,8,42,'test')
        self.assertEqual(a,select(rows[::-1],8,42,'test'))
        self.assertEqual(len(a),len(set(a)))
        self.assertEqual(sum(s.startswith('small') for s in a),6)

    def test_group_leakage_fails_even_for_distinct_files(self):
        index={'a':dict(provenance_group_id='same',source_file_sha256='a',duplicate_group_id='a'),
               'b':dict(provenance_group_id='same',source_file_sha256='b',duplicate_group_id='b')}
        with self.assertRaisesRegex(ValueError,'provenance_group_id'):assert_separation(['a'],['b'],index)

    def test_resampling_channel_selection_and_no_padding(self):
        config=dict(intermediate_rate_hz=8000,sample_rate_hz=16000,samples=32000,window_beta=5.)
        t=np.arange(48000*3)/48000
        x=.1*np.sin(2*np.pi*220*t)
        a=observation(np.c_[x,x],48000,config); b=observation(x,48000,config)
        self.assertEqual(a.shape,(32000,));np.testing.assert_array_equal(a,b)
        self.assertEqual(a.dtype,np.float32)
        with self.assertRaisesRegex(ValueError,'Short'):observation(x[:48000],48000,config)
        with self.assertRaises(ValueError):observation(np.full(96000,np.nan),48000,config)

    def test_mfcc_finite_fixed_width(self):
        import json
        from pathlib import Path
        cfg=json.loads((Path(__file__).parent/'config.json').read_text())['mfcc']
        x=.1*np.sin(2*np.pi*220*np.arange(32000)/16000).astype(np.float32)
        f=mfcc_features(x,cfg)
        self.assertEqual(f.shape,(26,));self.assertTrue(np.isfinite(f).all())

    def test_f1_and_missing_support(self):
        self.assertEqual(float(f1([[10,0],[0,10]])),1.)
        self.assertEqual(float(f1([[0,10],[10,0]])),0.)
        self.assertEqual(cm_metrics([[2,0],[0,0]])['per_class']['truck']['recall'],None)
        self.assertEqual(cm_metrics([[2,0],[0,0]])['balanced_accuracy'],None)

    def test_scaler_uses_training_only_and_prediction_is_read_only(self):
        x=np.array([[0,2],[1,3],[10,5],[11,6]],float);y=np.array([0,0,1,1])
        cfg=dict(C=1.,solver='lbfgs',tol=1e-6,max_iter=5000,fit_intercept=True,class_weight=None)
        model=fixed_fit(x,y,cfg);before=model[0].mean_.copy()
        model.predict_proba(np.full((2,2),10000.))
        np.testing.assert_array_equal(model[0].mean_,before)
        np.testing.assert_allclose(before,x.mean(axis=0))

    def test_bootstrap_pairing_and_synthetic_replication(self):
        source=np.broadcast_to([[9,1],[2,8]],(5,6,2,2)).copy()
        real=np.broadcast_to([[9,1],[2,8]],(5,6,4,2,2)).copy()
        targets={'real':real,'procedural':real[:,:1], 'audioldm':real[:,:1],
                 'real_repeated':real,'real_plus_procedural':real,'real_plus_audioldm':real}
        cfg=dict(draws=30,seed=314159)
        estimates,samples,rejected=bootstrap(source,targets,cfg)
        self.assertEqual(rejected,0)
        for name in ['delta_domain','delta_sim2real_procedural','gain_real_plus_audioldm']:
            self.assertAlmostEqual(estimates[name]['estimate'],0.)
            np.testing.assert_allclose(estimates[name]['ci95'],[0.,0.],atol=1e-15)
        _,again,_=bootstrap(source,targets,cfg)
        np.testing.assert_array_equal(samples['real'],again['real'])


if __name__=='__main__':unittest.main()
