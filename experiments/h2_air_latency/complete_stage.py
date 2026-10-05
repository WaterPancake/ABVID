"""Close the requested mechanism/collapse investigation from verified artifacts.

Checks reports and evidence without changing models, predictions or upstream
artifacts. A completed development investigation is not independent confirmation.
"""
from datetime import datetime, timezone
import re

from phase_common import *


def main():
    out = RESULTS / 'stage_completion.json'
    if out.exists():
        raise ValueError('Preserve existing completion record')
    check_execution()
    audit = read(RESULTS / 'verification.json')
    old = read(parent.RESULTS / 'verification.json')
    summary = read(RESULTS / 'summary.json')
    stats = read(RESULTS / 'statistics.json')
    generated = read(RESULTS / 'report_generation.json')
    assert audit['passed'] and old['passed'] and summary['passed']
    assert audit['result_summary_sha256'] == sha(RESULTS / 'summary.json')
    assert old['result_summary_sha256'] == sha(parent.RESULTS / 'summary.json')
    assert audit['verifier_sha256'] == sha(HERE / 'verify_control.py')
    assert generated['generator_sha256'] == sha(HERE / 'report_control.py')
    assert generated['verification_sha256'] == sha(RESULTS / 'verification.json')
    assert not summary['selection_or_tuning'] and summary['target_role'] == 'previously_exposed_development'
    assert not stats['decision']['independent_confirmation']
    assert audit['new_complete_waveform_replays'] == audit['new_exact_observation_replays'] == 9600
    assert audit['heads'] == 80 and audit['target_evaluations_replayed'] == 160
    assert audit['cross_cell_evaluations_replayed'] == 320 and audit['max_prediction_replay_error'] == 0
    assert audit['independently_reconstructed_integer_count_intervals'] == 504
    assert audit['additional_source_interaction_intervals_verified'] == 240
    assert audit['ground_air_parent_files'] == 19877 and audit['parent_files'] == 22316 and audit['historical_files'] == 8257
    assert old['target_crop_and_gain_replays'] == 8066 and old['gain_evaluations_replayed'] == 240
    for name, digest in generated['artifacts_sha256'].items():
        assert sha(ROOT / name) == digest, name

    report = ROOT / 'reports/H2_air_latency_control.md'
    text = report.read_text()
    assert 'VERIFIED_RESULTS_AND_ARTIFACTS' not in text and '**Status:' not in text
    for phrase in ['Findings for each check', 'What changes downstream', 'Complete-target results',
                   'Individual source levels', 'Factor effects and source interactions',
                   'Group-specific results', 'Measured runtime', 'Reproducibility and artifacts']:
        assert phrase in text, phrase
    tables = (RESULTS / 'report_tables.md').read_text()
    assert tables in text, 'Report tables must be the exact audited generated tables'
    links = []
    for target in re.findall(r'\]\(([^)]+)\)', text):
        target = target.split('#', 1)[0]
        if not target or '://' in target:
            continue
        resolved = (report.parent / target).resolve()
        assert resolved.exists(), str(resolved)
        links.append(str(resolved.relative_to(ROOT)))
    figures = sorted((ROOT / 'reports/H2_air_latency_control_figures').glob('*'))
    assert len(figures) == 6
    for path in figures:
        assert path.stat().st_size > 1000
    visual = read(RESULTS / 'figure_review.json')
    assert visual['passed']
    for path in figures:
        assert visual['artifacts_sha256'][str(path.relative_to(ROOT))] == sha(path)
    for path in [ROOT / 'ABVID_ROADMAP.md', ROOT / 'reports/proposed_experiments.md', HERE / 'README.md']:
        assert 'H2_air_latency_control.md' in path.read_text(), str(path)
    relevant = [report, ROOT / 'ABVID_ROADMAP.md', ROOT / 'reports/proposed_experiments.md',
                HERE / 'README.md', RESULTS / 'verification.json', RESULTS / 'report_generation.json',
                RESULTS / 'figure_review.json', parent.RESULTS / 'verification.json']
    result = dict(
        passed=True, completed_utc=datetime.now(timezone.utc).isoformat(),
        objective='Separate implemented ground/air effects, resolve the air-latency confound, and investigate direct-path truck collapse',
        scope='Completed controlled development investigation; no independent confirmation or further research arms',
        ground_air_factorial_verified=True, attenuation_latency_factorial_verified=True,
        collapse_software_gain_ranking_feature_and_waveform_checks_verified=True,
        physical_source_background_sensor_allocation='unknown',
        physical_renderer_field_calibration='unknown',
        target_role='previously_exposed_development', target_tuning=False,
        historical_outputs_preserved=True, primary_decision=stats['decision'],
        report_links_checked=len(links), figures_checked=6,
        reports_and_evidence_sha256={str(p.relative_to(ROOT)): sha(p) for p in relevant},
        completion_script_sha256=sha(Path(__file__)))
    save(out, result)
    print('Mechanism/collapse investigation complete:', result['scope'])
    print('Report links checked:', len(links), '; figures:', len(figures))


if __name__ == '__main__':
    main()
