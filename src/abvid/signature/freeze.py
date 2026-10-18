"""Read-only hash verification of a reviewable protocol draft."""
from __future__ import annotations

import json

from abvid.paths import resolve
from abvid.provenance.hashing import sha256


def verify(reference="repo:experiments/military_signature/p0_freeze.json"):
    ledger = json.loads(resolve(reference).read_text())
    if ledger["status"] != "frozen_draft_not_collection_authorization":
        raise ValueError("Unexpected freeze status")
    failures = []
    for file_ref, expected in ledger["file_sha256"].items():
        path = resolve(file_ref, must_exist=False)
        if not path.is_file() or sha256(path) != expected:
            failures.append(file_ref)
    if failures:
        raise ValueError("P0 freeze mismatch: " + ", ".join(failures))
    return {"valid": True, "status": ledger["status"], "files": len(ledger["file_sha256"]),
            "protocol_id": ledger["protocol_id"]}
