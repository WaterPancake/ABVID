from copy import deepcopy
from pathlib import Path
import pytest

from cast.config import configuration
from cast.provenance import checked_path, select, validate_row


def test_exact_parent_selection_group_exclusion_and_counts():
    c=configuration(); rows,inventory=select(c)
    assert len(rows)==50
    assert {r["dataset_id"] for r in rows}=={"IDMT"}
    assert len({r["provenance_group_id"] for r in rows})==5
    assert c["data"]["excluded_group"] not in {r["provenance_group_id"] for r in rows}
    assert len(inventory)==10 and all(b["selected"]==5 and b["shortage"]==0 for b in inventory)
    for group in {r["provenance_group_id"] for r in rows}:
        for label in ("car","truck"):
            ids=[r["file_id"] for r in rows if r["provenance_group_id"]==group and r["canonical_class"]==label]
            assert ids==sorted(ids)


@pytest.mark.parametrize("key,value",[("dataset_id","MELAUDIS"),("dataset_id","AI4TEN_pyroadacoustics"),
    ("provenance_group_id","connected_4001f06f57cfeee7"),("canonical_class","motorcycle"),
    ("source_path","dataset/MELAUDIS/target.wav"),("source_path","/tmp/target.wav"),
    ("source_path","dataset/IDMT_Traffic/audio/../../target_SE_CH34.wav"),
    ("original_channel_ids",[1,2]),("admitted_for_training",False),("file_id","0"*64),
    ("split_role","test"),("source_file_sha256",None)])
def test_disallowed_metadata_rejected_before_audio_hash(monkeypatch,key,value):
    c=configuration(); rows,_=select(c); bad=deepcopy(rows[0]);bad[key]=value
    def forbidden(*args):raise AssertionError("Audio bytes touched before rejection")
    monkeypatch.setattr("cast.provenance.sha",forbidden)
    with pytest.raises((ValueError,TypeError)):checked_path(bad,c,rows)


def test_symlink_and_mutated_ancestry_rejected(tmp_path,monkeypatch):
    c=configuration();rows,_=select(c); row=deepcopy(rows[0]);base=tmp_path/c["data"]["allowed_audio_root"];base.mkdir(parents=True)
    p=tmp_path/row["source_path"];p.symlink_to(tmp_path/"excluded.wav")
    with pytest.raises(ValueError,match="symlink"):validate_row(row,c,{row["file_id"]},tmp_path)
    p.unlink();p.write_bytes(b"fake")
    with pytest.raises(ValueError,match="hash"):checked_path(row,c,rows,tmp_path)
    row["recording_session"]="invented"
    with pytest.raises(ValueError,match="Ancestry"):checked_path(row,c,rows,tmp_path)
