import copy
import numpy as np
import pytest

from cast.synthetic import fixtures
from cast.renderer import validate_controls
from cast_improvement.engine import model_config
from cast_improvement.evaluate import transformed, inverse, temper


def bank():
    rows=[]
    for c in ('car','truck'):
        for g in range(3):
            p=copy.deepcopy(fixtures()['mixture']);p['spacing_hz']=[100+g*20]*3;p['envelope_knots']=[.4+g*.1,.8,1.6,.9,.4]
            rows.append({'class':c,'group':str(g),'file_id':f'{c}_{g}','parameters':p})
    return rows


def test_unit_temperature_exact_identity_and_parent_retention():
    b=bank();assert temper(b,1,model_config())==b
    for t in (1.1,1.25,1.5):
        changed=temper(b,t,model_config())
        assert {r['file_id'] for r in changed}=={r['file_id'] for r in b}
        for row in changed:assert validate_controls(row['parameters'],model_config())


def test_transform_inverse_preserves_effective_controls_modulo_envelope_scale():
    p=fixtures()['mixture'];q=inverse(transformed(p),model_config())
    for key in ['spacing_hz','harmonic_weights','noise_weights','harmonic_fraction']:
        np.testing.assert_allclose(q[key],p[key],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(np.array(q['envelope_knots'])/q['envelope_knots'][0],np.array(p['envelope_knots'])/p['envelope_knots'][0])


def test_undeclared_temperatures_rejected():
    with pytest.raises(ValueError):temper(bank(),2.,model_config())


def test_class_center_does_not_use_other_class():
    original=bank(); altered=copy.deepcopy(original)
    for row in altered:
        if row['class']=='truck':row['parameters']['spacing_hz']=[390]*3
    a=temper(original,1.5,model_config());b=temper(altered,1.5,model_config())
    assert [r for r in a if r['class']=='car']==[r for r in b if r['class']=='car']
