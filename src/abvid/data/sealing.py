"""Seal records for candidate confirmation data.

A seal binds admitted metadata hashes to a protocol. Opening sealed audio needs a
human-committed authorization naming the exact seal and frozen comparison
protocol. These checks guard procedure; they do not prove that nobody listened.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..paths import resolve
from ..provenance.hashing import sha256

SEAL_FIELDS = {"schema_version", "dataset_id", "role", "status", "protocol", "protocol_sha256",
               "metadata", "metadata_hashes_sha256", "audio_downloaded", "exposure_history",
               "consumed_by", "created"}
AUTH_FIELDS = {"schema_version", "seal", "seal_sha256", "protocol", "protocol_sha256",
               "overlap_audit", "overlap_audit_sha256", "authorized_by", "date", "scoring_passes"}
AUDIT_FIELDS = {"schema_version", "seal", "seal_sha256", "protocol", "protocol_sha256",
                "purpose", "authorized_by", "date"}


class SealedDataError(PermissionError):
    """Raised when sealed confirmation data would be opened without authorization."""


def _load(path, fields):
    value = json.loads(Path(path).read_text())
    if set(value) != fields:
        raise ValueError(f"Unexpected fields in {path}")
    return value


def verify_seal(seal_path):
    """Check that the sealed protocol and admitted metadata are unchanged."""
    seal = _load(seal_path, SEAL_FIELDS)
    if seal["status"] not in {"sealed", "consumed"}:
        raise ValueError("Unknown seal status")
    if sha256(resolve(seal["protocol"])) != seal["protocol_sha256"]:
        raise ValueError("Sealed protocol changed")
    metadata = resolve(seal["metadata"])
    hashes_path = metadata / "hashes.json"
    if sha256(hashes_path) != seal["metadata_hashes_sha256"]:
        raise ValueError("Sealed metadata hash list changed")
    for name, expected in json.loads(hashes_path.read_text()).items():
        if sha256(metadata / name) != expected:
            raise ValueError(f"Sealed metadata changed: {name}")
    return seal


def authorize_audit(seal_path, authorization_path):
    """Permit audio access for the label-blind overlap audit only (step A1).

    The named comparison protocol must already be frozen: its hash is bound
    here, so the audit cannot inform later design choices without detection.
    """
    seal = verify_seal(seal_path)
    if seal["status"] != "sealed":
        raise SealedDataError("This confirmation set has already been consumed")
    if not Path(authorization_path).is_file():
        raise SealedDataError("No audit authorization record; the set remains sealed")
    auth = _load(authorization_path, AUDIT_FIELDS)
    if auth["purpose"] != "overlap_audit":
        raise SealedDataError("Audit authorization must state purpose 'overlap_audit'")
    if auth["seal_sha256"] != sha256(seal_path):
        raise SealedDataError("Authorization names a different seal")
    if sha256(resolve(auth["protocol"])) != auth["protocol_sha256"]:
        raise SealedDataError("Frozen comparison protocol changed after authorization")
    if not str(auth["authorized_by"]).strip():
        raise SealedDataError("Authorization must name a person")
    return auth


def authorize(seal_path, authorization_path):
    """Return the authorization if it matches this seal; otherwise refuse access."""
    seal = verify_seal(seal_path)
    if seal["status"] != "sealed":
        raise SealedDataError("This confirmation set has already been consumed")
    if not Path(authorization_path).is_file():
        raise SealedDataError("No authorization record; the set remains sealed")
    auth = _load(authorization_path, AUTH_FIELDS)
    if auth["seal_sha256"] != sha256(seal_path):
        raise SealedDataError("Authorization names a different seal")
    for key in ("protocol", "overlap_audit"):
        if sha256(resolve(auth[key])) != auth[key + "_sha256"]:
            raise SealedDataError(f"Authorized {key} changed after authorization")
    if not str(auth["authorized_by"]).strip() or auth["scoring_passes"] != 1:
        raise SealedDataError("Authorization must name a person and exactly one scoring pass")
    return auth
