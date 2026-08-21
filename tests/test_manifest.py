from __future__ import annotations

import pytest

from vehicle_audio.manifest import REQUIRED_MANIFEST_FIELDS, validate_manifest_record


def test_manifest_validator_requires_reproduction_fields() -> None:
    record = {field: "value" for field in REQUIRED_MANIFEST_FIELDS}
    record["augmentation_seed"] = 123
    record["sample_rate"] = 16_000
    record["vehicle_model"] = None
    record["source_domain"] = "synthetic_toy"
    record["snr_db"] = 5.0
    record["gain_db"] = -2.0
    validate_manifest_record(record)

    del record["recording_session"]
    with pytest.raises(ValueError, match="recording_session"):
        validate_manifest_record(record)
