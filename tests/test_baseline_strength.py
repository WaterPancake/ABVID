import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from abvid.learning.grouped_search import (CS, ENCODERS, REFERENCE, fold_bootstrap, grid,  # noqa: E402
                                           metrics, nested_select)


def test_grid_is_complete_and_contains_the_h1_reference():
    configs = grid()
    assert len(configs) == 96 == len(set(configs)) and REFERENCE in configs
    assert {c[0] for c in configs} == set(ENCODERS) and {c[3] for c in configs} == set(CS)


def test_nested_selection_uses_inner_scores_and_breaks_ties_by_grid_order():
    configs = [("a",), ("b",), ("c",)]
    assert nested_select([0.6, 0.8, 0.7], configs) == ("b",)
    assert nested_select([0.8, 0.8, 0.7], configs) == ("a",)
    assert nested_select([-np.inf, 0.5, 0.5], configs) == ("b",)


def test_fold_bootstrap_point_and_site_blocks():
    delta = np.array([0.02, 0.04, 0.01, 0.03, 0.05, 0.0])
    result = fold_bootstrap(delta, ["a", "a", "b", "c", "b", "c"], draws=500)
    assert result["estimate"] == pytest.approx(delta.mean())
    assert result["ci95"][0] <= delta.mean() <= result["ci95"][1]
    constant = fold_bootstrap(np.full(6, 0.03), ["a", "a", "b", "b", "c", "c"], draws=200)
    assert constant["ci95"] == pytest.approx([0.03, 0.03]) and constant["site_ci95"] == pytest.approx([0.03, 0.03])


def test_metrics_report_threshold_free_and_fixed_rule_scores():
    y = np.array([0, 0, 0, 1, 1])
    m = metrics(y, np.array([0.1, 0.2, 0.7, 0.6, 0.9]))
    assert m["auroc"] == pytest.approx(5 / 6) and m["balanced_accuracy"] == pytest.approx((2 / 3 + 1) / 2)


def test_cnn14_front_end_matches_librosa_and_forward_shapes():
    librosa = pytest.importorskip("librosa")
    from abvid.representations.cnn14 import HOP, MELS, N_FFT, Cnn14
    model = Cnn14().eval()
    n = np.arange(N_FFT)
    window = librosa.filters.get_window("hann", N_FFT, fftbins=True)
    k = np.arange(N_FFT // 2 + 1)[:, None]
    real = (np.cos(2 * np.pi * k * n / N_FFT) * window)[:, None, :]
    imag = (-np.sin(2 * np.pi * k * n / N_FFT) * window)[:, None, :]
    with torch.no_grad():
        model.spectrogram_extractor.stft.conv_real.weight.copy_(torch.from_numpy(real).float())
        model.spectrogram_extractor.stft.conv_imag.weight.copy_(torch.from_numpy(imag).float())
        model.logmel_extractor.melW.copy_(torch.from_numpy(
            librosa.filters.mel(sr=32000, n_fft=N_FFT, n_mels=MELS, fmin=50, fmax=14000).T))
    x = np.random.default_rng(0).standard_normal(32000).astype(np.float32)
    power = model.spectrogram_extractor(torch.from_numpy(x)[None])[0, 0].detach().numpy()
    reference = np.abs(librosa.stft(x, n_fft=N_FFT, hop_length=HOP, window="hann", center=True, pad_mode="reflect")) ** 2
    np.testing.assert_allclose(power, reference.T, rtol=1e-3, atol=1e-2)
    embedding, probabilities = model(torch.from_numpy(np.stack([x, x])))
    assert embedding.shape == (2, 2048) and probabilities.shape == (2, 527)


@pytest.mark.archive
def test_official_cnn14_checkpoint_loads_strictly():
    from abvid.paths import resolve
    from abvid.representations.cnn14 import load
    model = load(resolve("artifact:models/panns/Cnn14_mAP=0.431.pth"),
                 "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31")
    assert not any(p.requires_grad for p in model.parameters()) and not model.training


@pytest.mark.archive
def test_beats_layer_embeddings_end_with_the_frozen_embedding():
    from abvid.paths import resolve
    from abvid.representations.beats import extract_embeddings, extract_layer_embeddings, load_encoder
    cfg = json.loads(resolve("archive:experiments/h1/frozen/h1_common_budget_v1.3/config.json").read_text())
    encoder, _ = load_encoder(dict(cfg["encoder"], checkpoint=str(resolve("artifact:models/beats/BEATs_iter3_plus_AS2M.pt"))))
    x = torch.from_numpy(np.random.default_rng(1).standard_normal((2, 32000)).astype(np.float32) * 0.1)
    layers = extract_layer_embeddings(encoder, x)
    assert layers.shape == (2, 12, 768) and torch.equal(layers[:, -1], extract_embeddings(encoder, x))
