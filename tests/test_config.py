from __future__ import annotations

import pytest

from vehicle_audio.config import GenerationConfig, load_config


def test_default_yaml_loads() -> None:
    config = load_config("configs/default.yaml")
    assert config.audio.sample_rate == 16_000
    assert config.audio.num_samples == 64_000
    assert config.audio.background_channel == 0
    assert config.temporal_crop.probability == 1.0
    assert config.sampling.target_strategy == "class_condition_balanced"
    assert -10.0 in config.noise.snr_db_choices


def test_unknown_configuration_section_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown configuration"):
        GenerationConfig.from_mapping({"not_an_augmentation": {}})


def test_unknown_target_sampling_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="sampling.target_strategy"):
        GenerationConfig.from_mapping({"sampling": {"target_strategy": "mystery"}})


def test_session_condition_balanced_sampling_strategy_is_supported() -> None:
    config = GenerationConfig.from_mapping(
        {"sampling": {"target_strategy": "class_session_condition_balanced"}}
    )

    assert config.sampling.target_strategy == "class_session_condition_balanced"
