"""Use the verified paired bootstrap with explicit attenuation/latency axes.

Array order is [none, amplitude-only, latency-only, both]. The parent's numerical
first factor is therefore latency, and its second factor is attenuation. Names
are translated; no historical probabilities or estimates enter the computation.
"""
from phase_common import *
from effect_statistics import calculate as paired_calculate,full_score
from metrics import interval

NAMES=dict(G0A0='A0L0',G0A1='A1L0',G1A0='A0L1',G1A1='A1L1',
    ground_at_A0='latency_at_A0',ground_at_A1='latency_at_A1',
    air_at_G0='attenuation_at_L0',air_at_G1='attenuation_at_L1',
    ground='latency',air='attenuation',interaction='interaction',ground_minus_air='latency_minus_attenuation')


def rename(value):
    if isinstance(value,dict):return {NAMES.get(k,k):rename(v) for k,v in value.items()}
    return value


def effects(q):
    none,amplitude,latency,both=q
    return dict(latency_at_A0=latency-none,latency_at_A1=both-amplitude,
        attenuation_at_L0=amplitude-none,attenuation_at_L1=both-latency,
        latency=(latency-none+both-amplitude)/2,attenuation=(amplitude-none+both-latency)/2,
        interaction=both-latency-amplitude+none,latency_minus_attenuation=latency-amplitude)


def calculate(y,groups,probabilities):
    original,old_raw,indices=paired_calculate(y,groups,probabilities)
    stats={k:rename(original[k]) for k in ['estimates','matched_minus_native']}
    raw={'__'.join(NAMES.get(k,k) for k in key.split('__')):value for key,value in old_raw.items()}
    stats['source_interactions']={}
    for variant in probabilities:
        stats['source_interactions'][variant]={}
        for rep in REPS:
            stats['source_interactions'][variant][rep]={}
            for effect in ['latency','attenuation','interaction','latency_minus_attenuation']:
                stats['source_interactions'][variant][rep][effect]={}
                values=stats['estimates'][variant][rep]
                for metric in values['S0'][effect]:
                    point=values['S1'][effect][metric]['estimate']-values['S0'][effect][metric]['estimate']
                    sample=raw[f'{variant}__{rep}__S1__{effect}__{metric}']-raw[f'{variant}__{rep}__S0__{effect}__{metric}']
                    stats['source_interactions'][variant][rep][effect][metric]=interval(point,sample)
                    raw[f'S1_minus_S0__{variant}__{rep}__{effect}__{metric}']=sample
    primary=stats['estimates']['native']['BEATs768']['mean_S']
    d=primary['latency_minus_attenuation']['macro_f1']['ci95']
    l=primary['latency']['macro_f1']['ci95'];a=primary['attenuation']['macro_f1']['ci95']
    extra=primary['attenuation_at_L1']['macro_f1']
    stats['decision']=dict(primary=CONFIG['primary'],latency_larger=d[0]>0,attenuation_larger=d[1]<0,
        practical_latency_dominance=d[0]>=.03 and l[0]>0,
        practical_attenuation_dominance=d[1]<=-.03 and a[0]>0,
        latency_only_equivalent_to_joint_within_three_points=extra['ci95'][0]>=-.03 and extra['ci95'][1]<=.03,
        equivalence_diagnostic=extra,independent_confirmation=False)
    return stats,raw,indices
