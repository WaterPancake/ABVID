import hashlib
import io
import json
import zipfile

import pytest

from abvid.data import dcase_t10
from abvid.data.remote_zip import Volumes, central_directory, read_member
from abvid.data.sealing import SealedDataError, authorize, verify_seal

HEADER = ",".join(dcase_t10.FIELDS)


def table(split, rows):
    lines = [HEADER] + [f"{d},{h},{m},{a},{b},{c},{e},{split}/{i:05d}.flac,{split}"
                        for i, (d, h, m, a, b, c, e) in enumerate(rows)]
    return ("\n".join(lines) + "\n").encode()


def archive(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return stream.getvalue()


def local(blobs):
    def fetcher(urls):
        def fetch(volume, start, length):
            return blobs[urls[volume]][start:start + length]
        return fetch
    return fetcher


def site_members(site, train_rows, val_rows):
    members = {f"{site}/meta.json": b'{"geometry": {}, "traffic": {}}',
               f"{site}/train.csv": table("train", train_rows),
               f"{site}/val.csv": table("val", val_rows)}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for i in range(len(rows)):
            members[f"{site}/{split}/{i:05d}.flac"] = b"fLaC" + bytes(64)
    return members


def test_reader_never_requests_audio_members():
    blob = archive(site_members("loc1", [(0, 1, 2, 1, 0, 0, 0)], [(0, 1, 3, 0, 0, 0, 0)]))
    volumes = Volumes([len(blob)], lambda v, s, n: blob[s:s + n], max_bytes=10_000)
    members, _ = central_directory(volumes)
    requested = len(volumes.requests)
    audio = next(m for m in members if m.is_audio)
    with pytest.raises(PermissionError):
        read_member(volumes, audio, allow=r".*")
    assert len(volumes.requests) == requested
    meta = next(m for m in members if m.name == "loc1/meta.json")
    assert json.loads(read_member(volumes, meta, allow=dcase_t10.METADATA)) == {"geometry": {}, "traffic": {}}


def test_volumes_read_across_boundaries_and_enforce_budget():
    data = bytes(range(100))
    volumes = Volumes([40, 35, 25], lambda v, s, n: data[[0, 40, 75][v] + s:[0, 40, 75][v] + s + n], max_bytes=60)
    assert volumes.read(0, 30, 20) == data[30:50]
    assert volumes.read(1, 30, 10) == data[70:80]
    assert volumes.transferred == [10, 15, 5]
    with pytest.raises(RuntimeError, match="budget"):
        volumes.read(0, 0, 40)


def test_label_parsing_validation_and_single_vehicle_fields():
    rows = dcase_t10.parse_labels(table("val", [(3, 9, 1, 0, 1, 0, 0), (3, 9, 2, 0, 0, 0, 1),
                                                (3, 9, 3, 2, 0, 0, 0)]), "loc4", "val")
    assert [r["single_class"] for r in rows] == ["car", "cv", None]
    assert rows[0]["single_direction"] == "right" and rows[1]["single_direction"] == "right"
    assert rows[0]["recording_session"] == "unknown" and rows[0]["provenance_group_id"] == "DCASE2024T10|loc4"
    with pytest.raises(ValueError, match="columns"):
        dcase_t10.parse_labels(b"dow,hour\n1,2\n", "loc4", "val")
    with pytest.raises(ValueError, match="Negative"):
        dcase_t10.parse_labels(table("val", [(3, 9, 1, -1, 0, 0, 0)]), "loc4", "val")
    with pytest.raises(ValueError, match="split"):
        dcase_t10.parse_labels(table("train", [(3, 9, 1, 1, 0, 0, 0)]), "loc4", "val")


def test_isolation_requires_listed_empty_unambiguous_neighbours():
    rows = dcase_t10.parse_labels(table("train", [(1, 0, 0, 0, 0, 0, 0), (1, 0, 1, 1, 0, 0, 0),
                                                  (1, 0, 2, 0, 0, 0, 0), (1, 0, 4, 0, 0, 1, 0),
                                                  (2, 0, 7, 1, 0, 0, 0), (2, 0, 7, 0, 0, 0, 0)]), "loc5", "train")
    dcase_t10.mark_isolation(rows)
    assert [r["isolated_single_vehicle"] for r in rows] == [False, True, False, False, False, False]
    assert rows[4]["weekday_clock_repeated"] and not rows[0]["weekday_clock_repeated"]


def test_member_inventory_must_match_labels():
    members = site_members("loc2", [(0, 1, 2, 1, 0, 0, 0)], [(0, 1, 3, 0, 0, 0, 0)])
    members["loc2/val/00009.flac"] = b"fLaC"
    blob = archive(members)
    volumes = Volumes([len(blob)], lambda v, s, n: blob[s:s + n], max_bytes=10_000)
    listed, _ = central_directory(volumes)
    rows = dcase_t10.parse_labels(members["loc2/train.csv"], "loc2", "train") + \
        dcase_t10.parse_labels(members["loc2/val.csv"], "loc2", "val")
    with pytest.raises(ValueError, match="mismatch"):
        dcase_t10.check_members(rows, listed, "loc2")


def test_admission_end_to_end_transfers_no_audio(tmp_path):
    blobs, dev, ev = {}, [], []
    for site in dcase_t10.SITES:
        blobs[f"dev/{site}.zip"] = archive(site_members(site, [(0, 1, 2, 1, 0, 0, 0), (0, 1, 3, 0, 0, 1, 0)],
                                                        [(0, 1, 9, 2, 1, 0, 0)]))
        blobs[f"eval/{site}.zip"] = archive({f"{site}/test/00000.flac": b"fLaC"})
    blobs["dev/simulation.zip"] = archive({"simulation/loc1/car/left/0.flac": b"fLaC"})
    for key, blob in blobs.items():
        entry = {"key": key.split("/")[1], "size": len(blob), "links": {"self": key},
                 "checksum": "md5:" + hashlib.md5(blob).hexdigest()}
        (dev if key.startswith("dev") else ev).append(entry)
    records = {dcase_t10.DEVELOPMENT_RECORD: {"files": dev}, dcase_t10.EVALUATION_RECORD: {"files": ev}}
    summary, receipt = dcase_t10.admit(tmp_path / "out", fetcher=local(blobs), records=records)
    assert summary["segments"] == 18 and summary["sites"]["loc1|all"]["single_car"] == 1
    assert summary["sites"]["loc1|all"]["single_cv"] == 1 and summary["simulation_audio_members"] == 1
    assert receipt["audio_bytes_requested"] == 0
    hashes = json.loads((tmp_path / "out/hashes.json").read_text())
    assert all(hashlib.sha256((tmp_path / "out" / k).read_bytes()).hexdigest() == v for k, v in hashes.items())
    assert not any(k.endswith(".flac") for k in hashes)
    with pytest.raises(FileExistsError):
        dcase_t10.admit(tmp_path / "out", fetcher=local(blobs), records=records)


def write_json(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seal_detects_changes_and_requires_matching_authorization(tmp_path, monkeypatch):
    monkeypatch.setenv("ABVID_DATA_ROOT", str(tmp_path))
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "labels.csv").write_text("a\n")
    write_json(meta / "hashes.json", {"labels.csv": hashlib.sha256(b"a\n").hexdigest()})
    (tmp_path / "protocol.md").write_text("frozen\n")
    (tmp_path / "audit.json").write_text("{}\n")
    seal = {"schema_version": 1, "dataset_id": "X", "role": "T_confirm_candidate", "status": "sealed",
            "protocol": "data:protocol.md", "protocol_sha256": hashlib.sha256(b"frozen\n").hexdigest(),
            "metadata": "data:meta", "metadata_hashes_sha256": hashlib.sha256((meta / "hashes.json").read_bytes()).hexdigest(),
            "audio_downloaded": False, "exposure_history": [], "consumed_by": None, "created": "2026-10-11"}
    seal_sha = write_json(tmp_path / "seal.json", seal)
    assert verify_seal(tmp_path / "seal.json")["status"] == "sealed"
    with pytest.raises(SealedDataError, match="No authorization"):
        authorize(tmp_path / "seal.json", tmp_path / "missing.json")
    auth = {"schema_version": 1, "seal": "data:seal.json", "seal_sha256": seal_sha,
            "protocol": "data:protocol.md", "protocol_sha256": seal["protocol_sha256"],
            "overlap_audit": "data:audit.json", "overlap_audit_sha256": hashlib.sha256(b"{}\n").hexdigest(),
            "authorized_by": "reviewer", "date": "2026-10-12", "scoring_passes": 1}
    write_json(tmp_path / "auth.json", auth)
    assert authorize(tmp_path / "seal.json", tmp_path / "auth.json")["scoring_passes"] == 1
    write_json(tmp_path / "auth.json", {**auth, "scoring_passes": 2})
    with pytest.raises(SealedDataError, match="exactly one"):
        authorize(tmp_path / "seal.json", tmp_path / "auth.json")
    (meta / "labels.csv").write_text("b\n")
    with pytest.raises(ValueError, match="changed"):
        verify_seal(tmp_path / "seal.json")
    (meta / "labels.csv").write_text("a\n")
    write_json(tmp_path / "seal.json", {**seal, "status": "consumed", "consumed_by": "study"})
    with pytest.raises(SealedDataError, match="consumed"):
        authorize(tmp_path / "seal.json", tmp_path / "auth.json")


def test_audit_authorization_is_separate_and_purpose_bound(tmp_path, monkeypatch):
    from abvid.data.sealing import authorize_audit
    monkeypatch.setenv("ABVID_DATA_ROOT", str(tmp_path))
    meta = tmp_path / "meta"
    meta.mkdir()
    write_json(meta / "hashes.json", {})
    (tmp_path / "protocol.md").write_text("frozen\n")
    seal = {"schema_version": 1, "dataset_id": "X", "role": "T_confirm_candidate", "status": "sealed",
            "protocol": "data:protocol.md", "protocol_sha256": hashlib.sha256(b"frozen\n").hexdigest(),
            "metadata": "data:meta", "metadata_hashes_sha256": hashlib.sha256((meta / "hashes.json").read_bytes()).hexdigest(),
            "audio_downloaded": False, "exposure_history": [], "consumed_by": None, "created": "2026-10-11"}
    seal_sha = write_json(tmp_path / "seal.json", seal)
    auth = {"schema_version": 1, "seal": "data:seal.json", "seal_sha256": seal_sha,
            "protocol": "data:protocol.md", "protocol_sha256": seal["protocol_sha256"],
            "purpose": "overlap_audit", "authorized_by": "reviewer", "date": "2026-10-12"}
    write_json(tmp_path / "audit.json", auth)
    assert authorize_audit(tmp_path / "seal.json", tmp_path / "audit.json")["purpose"] == "overlap_audit"
    write_json(tmp_path / "audit.json", {**auth, "purpose": "scoring"})
    with pytest.raises(SealedDataError, match="purpose"):
        authorize_audit(tmp_path / "seal.json", tmp_path / "audit.json")
    with pytest.raises(SealedDataError, match="No audit authorization"):
        authorize_audit(tmp_path / "seal.json", tmp_path / "missing.json")
    with pytest.raises(ValueError, match="Unexpected fields"):
        authorize(tmp_path / "seal.json", tmp_path / "audit.json")
