"""Population, factor-label and uncertainty controls before target evaluation."""
from collections import Counter,defaultdict
import importlib.util
import unittest
from phase_common import *
from phase_statistics import calculate,effects,NAMES
from metrics import interval


class Controls(unittest.TestCase):
    def test_complete_population_and_source_roles(self):
        obs=rows(FREEZE/'observations.jsonl');fits=rows(FREEZE/'fits.jsonl')
        self.assertEqual(len(obs),19200);self.assertEqual(len(fits),80)
        self.assertEqual(sum(r['reused'] for r in obs),9600)
        self.assertEqual(sum(r['reused'] for r in fits),40)
        self.assertEqual(Counter(r['role'] for r in obs),{'train':15200,'validation':4000})
        byid={r['job_id']:r for r in obs};roles=defaultdict(set);geometries=defaultdict(set)
        for r in obs:
            roles[r['source_parent_group']].add(r['role'])
            geometries[r['geometry_id']].add((r['source_level'],r['cell'],r['class_id']))
            self.assertTrue(r['ground'])
            self.assertEqual(r['amplitude'],r['cell'][1]=='1')
            self.assertEqual(r['latency'],r['cell'][3]=='1')
        self.assertEqual(len(roles),240);self.assertEqual(len(geometries),1200)
        self.assertTrue(all(len(v)==1 for v in roles.values()))
        self.assertTrue(all(len(v)==16 for v in geometries.values()))
        for f in fits:
            train=f['train_job_ids'];val=f['validation_job_ids']
            self.assertFalse(set(train)&set(val))
            self.assertEqual(Counter(byid[j]['class_id'] for j in train),{0:190,1:190})
            self.assertEqual(Counter(byid[j]['class_id'] for j in val),{0:50,1:50})
            self.assertTrue(all(byid[j]['role']=='train' for j in train))
            self.assertTrue(all(byid[j]['role']=='validation' for j in val))
            self.assertFalse({byid[j]['source_parent_group'] for j in train}&{byid[j]['source_parent_group'] for j in val})
            for j in train+val:
                self.assertEqual(byid[j]['cell'],f['cell'])
                self.assertEqual(byid[j]['source_level'],f['source_level'])
                self.assertEqual(byid[j]['replicate_seed'],f['replicate_seed'])

    def test_axis_meanings(self):
        q=np.array([.1,.2,.5,.8])
        e=effects(q)
        self.assertAlmostEqual(e['attenuation'],.2)
        self.assertAlmostEqual(e['latency'],.5)
        self.assertAlmostEqual(e['interaction'],.2)
        self.assertAlmostEqual(e['latency_minus_attenuation'],.3)
        self.assertAlmostEqual(e['attenuation_at_L1'],.3)

    def test_paired_intervals_and_translation_against_integer_counts(self):
        rng=np.random.default_rng(904)
        y=np.tile([0,1],4);g=np.repeat(np.arange(4),2)
        p={v:{r:rng.uniform(.01,.99,(2,4,5,8)) for r in REPS} for v in ['native','matched']}
        stats,raw,indices=calculate(y,g,p)
        inverse={v:k for k,v in NAMES.items()}
        def translate(value):
            return {inverse.get(k,k):translate(v) for k,v in value.items()} if isinstance(value,dict) else value
        prior_stats={k:translate(stats[k]) for k in ['estimates','matched_minus_native']}
        prior_raw={'__'.join(inverse.get(k,k) for k in key.split('__')):value
            for key,value in raw.items() if not key.startswith('S1_minus_S0')}
        spec=importlib.util.spec_from_file_location('verified_ground_air_counts',PARENT_HERE/'verify_results.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        count=module.independent_intervals(y,g,p,prior_stats,prior_raw,indices)
        self.assertEqual(count,504)
        for variant in p:
            for rep in REPS:
                for name in ['latency','attenuation','interaction','latency_minus_attenuation']:
                    for metric,result in stats['source_interactions'][variant][rep][name].items():
                        point=stats['estimates'][variant][rep]['S1'][name][metric]['estimate']-stats['estimates'][variant][rep]['S0'][name][metric]['estimate']
                        sample=raw[f'{variant}__{rep}__S1__{name}__{metric}']-raw[f'{variant}__{rep}__S0__{name}__{metric}']
                        module.equal(result,interval(point,sample))
        primary=stats['estimates']['native']['BEATs768']['mean_S']
        q=np.array([primary[c]['macro_f1']['estimate'] for c in CELLS])
        for key,value in effects(q).items():self.assertAlmostEqual(value,primary[key]['macro_f1']['estimate'])
        ci=primary['attenuation_at_L1']['macro_f1']['ci95']
        self.assertEqual(stats['decision']['latency_only_equivalent_to_joint_within_three_points'],ci[0]>=-.03 and ci[1]<=.03)


if __name__=='__main__':unittest.main()
