"""Post-result, read-only implementation inspection; no audio/model changes.

The frozen protocol already names finite-FIR latency as part of the air factor.
This records its exact algebra and a static response witness, without assigning
the observed classification benefit to delay or choosing any classifier setting.
"""
from common import *
from renderer import reference_manager, air_filters, build_plan


def main():
    check_execution()
    out=RESULTS/'air_filter_phase_inspection.json'
    if out.exists():
        raise ValueError('Inspection already exists; preserve it')
    obj=reference_manager()
    filters=air_filters(np.array([0.,5.,20.,50.]),obj)
    np.testing.assert_array_equal(filters,filters[:,::-1])
    identity=np.zeros(11);identity[5]=1.
    np.testing.assert_allclose(filters[0],identity,rtol=0,atol=1e-14)
    records=[]
    for horizontal_range in [5.,20.,50.]:
        plan=build_plan(np.array([[horizontal_range,0.,.5]]),np.array([0.,0.,1.2]))
        d,a,b=[float(plan[k][0]) for k in ['d','a','b']]
        for frequency in [125.,500.,1500.,3000.]:
            omega=2*np.pi*frequency/8000
            response=lambda h:np.dot(h,np.exp(-1j*omega*np.arange(len(h))))
            h=[response(v[0]) for v in plan['air']]
            centered=[z*np.exp(1j*omega*5) for z in h]
            assert max(abs(z.imag) for z in centered)<1e-12
            assert all(z.real>0 for z in centered)
            direct=np.exp(-2j*np.pi*frequency*d/obj.c)/d
            reflected=response(plan['ground'][0])*np.exp(-2j*np.pi*frequency*(a+b)/obj.c)/(a+b)
            combined=direct*h[0]+reflected*h[1]*h[2]
            amplitude_only=direct*abs(h[0])+reflected*abs(h[1])*abs(h[2])
            delay_only=direct*np.exp(-1j*omega*5)+reflected*np.exp(-1j*omega*10)
            records.append(dict(horizontal_range_m=horizontal_range,frequency_hz=frequency,
                direct_air_magnitude_db=float(20*np.log10(abs(h[0]))),
                reflected_air_magnitude_db=float(20*np.log10(abs(h[1]*h[2]))),
                ground_no_air_magnitude=float(abs(direct+reflected)),
                joint_implemented_magnitude=float(abs(combined)),
                ground_with_amplitude_only_air_magnitude=float(abs(amplitude_only)),
                ground_with_delay_only_air_magnitude=float(abs(delay_only)),
                joint_vs_amplitude_only_db=float(20*np.log10(abs(combined)/abs(amplitude_only))),
                joint_vs_delay_only_db=float(20*np.log10(abs(combined)/abs(delay_only)))))
    report=dict(passed=True,analysis_role='post_result_read_only_implementation_inspection',
        new_target_predictions=0,new_fits=0,changes_to_frozen_renderer=0,
        filter_taps=11,symmetry_exact=True,static_fir_delay_samples=5,sample_rate_hz=8000,
        static_direct_air_delay_samples=5,static_reflected_air_delay_samples=10,
        static_extra_relative_reflected_delay_samples=5,static_extra_relative_delay_ms=.625,
        zero_distance_filter=filters[0].tolist(),zero_distance_identity_delay_error=float(abs(filters[0]-identity).max()),
        static_witnesses=records,
        interpretation='Air toggle includes differential finite-FIR latency. Static witnesses show interference can change even when attenuation is removed. No classification experiment isolates latency from absorption; their contributions to the observed F1 interaction remain unknown.',
        renderer_sha256=sha(H2/'renderer.py'),execution_lock_sha256=sha(EXECUTION/'lock.json'),
        script_sha256=sha(Path(__file__)))
    save(out,report)
    print({k:v for k,v in report.items() if k not in ['static_witnesses','zero_distance_filter']},flush=True)


if __name__=='__main__':main()
