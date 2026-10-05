"""Scientific-control failures and independent metric calculations for H2."""
import unittest
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from metrics import score,contrasts,bootstrap,CELLS
from deterministic_filters import butterworth
from scipy.signal import butter, freqz


class ScientificControls(unittest.TestCase):
    def test_threshold_ties_are_car_and_undefined_metrics_are_null(self):
        s=score(np.array([0,0]),np.array([.5,.6]))
        self.assertEqual(s['confusion_matrix'],[[1.,1.],[0.,0.]])
        self.assertIsNone(s['truck_recall']);self.assertIsNone(s['roc_auc'])
        self.assertIsNone(s['average_precision']);self.assertIsNone(s['balanced_accuracy'])

    def test_weighted_scores_equal_explicit_group_replication(self):
        y=np.array([0,1,0,1,0,1,0,1]);p=np.array([.1,.3,.9,.8,.5,.7,.2,.1])
        w=np.repeat([2,0,1,1],2);indices=np.repeat(np.arange(8),w)
        weighted=score(y,p,w); duplicated=score(y[indices],p[indices])
        for key,value in weighted.items():
            if key=='count': continue
            np.testing.assert_allclose(value,duplicated[key],rtol=0,atol=1e-14)
        self.assertAlmostEqual(weighted['macro_f1'],f1_score(y[indices],p[indices]>.5,average='macro'))

    def test_factorial_identity_and_interaction_are_not_pooled(self):
        q=np.array([[.4,.5],[.45,.55],[.6,.65],[.5,.6]])
        c=contrasts(q)
        np.testing.assert_allclose(c['D'],c['delta_S']-c['delta_P'])
        np.testing.assert_allclose(c['D'],q[2]-q[1])
        np.testing.assert_allclose(c['interaction'],q[3]-q[2]-q[1]+q[0])

    def test_bootstrap_draw_matches_literal_group_and_bank_repetition(self):
        y=np.tile([0,1],4);g=np.repeat(np.arange(4),2)
        p=np.random.default_rng(7).uniform(.01,.99,(4,5,8))
        result,raw,indices=bootstrap(y,g,{'BEATs768':p},draws=30,seed=99)
        for draw in [0,1,29]:
            ii=np.concatenate([np.flatnonzero(g==group) for group in indices['target_groups'][draw]])
            expected=[]
            for ci,cell in enumerate(CELLS):
                scores=[score(y[ii],p[ci,bank,ii]) for bank in indices['banks'][draw]]
                for key in ['macro_f1','roc_auc','ece10_truck','log_loss']:
                    self.assertAlmostEqual(raw[f'BEATs768__{cell}__{key}'][draw],np.mean([s[key] for s in scores]))
                expected.append(np.mean([s['macro_f1'] for s in scores]))
            self.assertAlmostEqual(raw['BEATs768__D__macro_f1'][draw],expected[2]-expected[1])

    def test_identical_cells_have_zero_contrast_uncertainty(self):
        y=np.tile([0,1],4);g=np.repeat(np.arange(4),2)
        p=np.broadcast_to(np.random.default_rng(4).uniform(.01,.99,(5,8)),(4,5,8)).copy()
        result,raw,_=bootstrap(y,g,{'BEATs768':p},draws=20)
        np.testing.assert_array_equal(raw['BEATs768__D__macro_f1'],np.zeros(20))
        self.assertFalse(result['decision']['practical_source_dominance_supported'])

    def test_fixed_butterworth_matches_scipy_response_and_replays(self):
        for order,cutoff,band in [(4,[.075,.425],True),(4,[.15,.3],True),(3,.5,False)]:
            b,a=butterworth(order,cutoff,band)
            reference=butter(order,cutoff,btype='band' if band else 'low')
            _,actual=freqz(b,a,worN=1025);_,expected=freqz(*reference,worN=1025)
            np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-9)
            for _ in range(20):
                bb,aa=butterworth(order,cutoff,band)
                np.testing.assert_array_equal(b,bb);np.testing.assert_array_equal(a,aa)


if __name__=='__main__': unittest.main()
