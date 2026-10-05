import json
import pytest

from cast.cli import validate_gate
from cast.config import configuration,digest


def test_failed_synthetic_blocks_real_access(tmp_path):
    c=configuration();(tmp_path/"synthetic").mkdir()
    (tmp_path/"tests.json").write_text(json.dumps({"passed":True,"config_sha256":digest(c)}))
    (tmp_path/"synthetic/summary.json").write_text(json.dumps({"passed":False,"config_sha256":digest(c)}))
    with pytest.raises(RuntimeError,match="access denied"):validate_gate(tmp_path,c)


def test_changed_config_evidence_blocks_real_access(tmp_path):
    c=configuration()
    (tmp_path/"tests.json").write_text(json.dumps({"passed":True,"config_sha256":"0"*64}))
    with pytest.raises(RuntimeError,match="access denied"):validate_gate(tmp_path,c)
