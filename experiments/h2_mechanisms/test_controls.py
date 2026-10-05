"""Population, intervention and independent uncertainty checks before execution."""
import unittest
from collections import Counter,defaultdict
from sklearn.metrics import f1_score
from common import *
from features import normalize_target,acoustic
from effect_statistics import effects,calculate


class Controls(unittest.TestCase):
    def test_exact_factorial_population_and_parent_roles(self):
        observations=rows(FREEZE/'observations.jsonl');index={r['job_id']:r for r in observations}
        self.assertEqual(len(index),19200);self.assertEqual(sum(not r['reused'] for r in observations),9600)
        roles=defaultdict(set);geometry=defaultdict(set)
        for r in observations:
            roles[r['source_parent_group']].add(r['role'])
            geometry[r['geometry_id']].add((r['class_id'],r['source_level'],r['path_level']))
        self.assertTrue(all(len(v)==1 for v in roles.values()))
        self.assertTrue(all(len(v)==16 for v in geometry.values()));self.assertEqual(len(geometry),1200)
        fits=rows(FREEZE/'fits.jsonl');self.assertEqual(len(fits),80)
        self.assertEqual(sum(not r['reused'] for r in fits),40)
        for fit in fits:
            train=fit['train_job_ids'];val=fit['validation_job_ids']
            self.assertEqual(Counter(index[i]['class_id'] for i in train),{0:190,1:190})
            self.assertEqual(Counter(index[i]['class_id'] for i in val),{0:50,1:50})
            self.assertFalse({index[i]['source_parent_group'] for i in train}&{index[i]['source_parent_group'] for i in val})

    def test_known_ground_and_air_effects_are_distinct(self):
        e=effects(np.array([.2,.3,.5,.6]))
        self.assertAlmostEqual(e['ground'],.3);self.assertAlmostEqual(e['air'],.1)
        self.assertAlmostEqual(e['interaction'],0.);self.assertAlmostEqual(e['ground_minus_air'],.2)

    def test_rms_intervention_preserves_shape_and_does_not_clip(self):
        x=np.zeros(32000,np.float32);x[100]=.9
        y,gain=normalize_target(x)
        self.assertGreater(abs(y).max(),1.)
        self.assertAlmostEqual(np.sqrt(np.mean(y.astype(float)**2)),10**(-26/20),places=8)
        np.testing.assert_array_equal(y,(x.astype(float)*gain).astype(np.float32))
        with self.assertRaises(ValueError):normalize_target(np.zeros(32000,np.float32))

    def test_spectral_ratios_do_not_change_under_scalar_gain(self):
        t=np.arange(32000)/16000;x=.1*np.sin(2*np.pi*150*t)+.02*np.sin(2*np.pi*1100*t)
        a=acoustic(x);b=acoustic(x*10**(12/20))
        self.assertAlmostEqual(b[0]-a[0],12.,places=10)
        np.testing.assert_allclose(a[3:],b[3:],rtol=0,atol=1e-9)
        self.assertAlmostEqual(a[5:].sum(),1.,places=12)

    def test_paired_bootstrap_matches_literal_repeated_groups(self):
        y=np.tile([0,1],4);group=np.repeat(np.arange(4),2);rng=np.random.default_rng(82)
        probabilities={variant:{rep:rng.uniform(.01,.99,(2,4,5,8)) for rep in REPS} for variant in ['native','matched']}
        result,raw,indices=calculate(y,group,probabilities)
        for draw in [0,17,9999]:
            events=np.concatenate([np.flatnonzero(group==g) for g in indices['target_groups'][draw]])
            q=np.empty((2,4))
            for si in range(2):
                for pi in range(4):
                    q[si,pi]=np.mean([f1_score(y[events],probabilities['native']['BEATs768'][si,pi,b,events]>.5,
                        labels=[0,1],average='macro',zero_division=0) for b in indices['banks'][draw]])
            self.assertAlmostEqual(raw['native__BEATs768__mean_S__ground_minus_air__macro_f1'][draw],np.mean(q[:,2]-q[:,1]))
        self.assertFalse(result['decision']['independent_confirmation'])


if __name__=='__main__':unittest.main()
