import pytest
import torch

from abvid.representations.beats import extract_embeddings, extract_layer_embeddings, masked_mean


def test_masked_pooling_and_invalid_masks():
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [900.0, 900.0]]])
    mask = torch.tensor([[False, False, True]])
    assert torch.equal(masked_mean(x, mask), torch.tensor([[2.0, 3.0]]))
    assert torch.equal(masked_mean(x), x.mean(1))
    for bad in (
        torch.ones(1, 3, dtype=torch.bool),
        torch.zeros(1, 2, dtype=torch.bool),
        torch.zeros(1, 3),
    ):
        with pytest.raises(ValueError):
            masked_mean(x, bad)
    with pytest.raises(ValueError):
        masked_mean(torch.full((1, 3, 2), float("nan")))


class ToyEncoder(torch.nn.Module):
    def extract_features(self, x, padding_mask):
        return torch.ones(len(x), 3, 768), torch.zeros(len(x), 3, dtype=torch.bool)


def test_frozen_adapter_contract():
    encoder = ToyEncoder()
    with pytest.raises(ValueError):
        extract_embeddings(encoder, torch.zeros(1, 32000))
    encoder.eval()
    assert extract_embeddings(encoder, torch.zeros(2, 32000)).shape == (2, 768)
    for x in (
        torch.zeros(1, 16000),
        torch.zeros(32000),
        torch.full((1, 32000), float("inf")),
    ):
        with pytest.raises(ValueError):
            extract_embeddings(encoder, x)


class ScaleLayer(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale, None


class LayeredEncoder(torch.nn.Module):
    """Upstream layout: layers see time x batch x features; tokens come back batch-first."""

    def __init__(self, scales=(1.0, 2.0, 3.0), offset=0.0, skip_last=False):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layers = torch.nn.ModuleList(ScaleLayer(s) for s in scales)
        self.offset, self.skip_last = offset, skip_last

    def extract_features(self, x, padding_mask):
        tokens = x.reshape(len(x), 50, 640)[..., :1].expand(-1, -1, 768).transpose(0, 1)
        layers = self.encoder.layers[:-1] if self.skip_last else self.encoder.layers
        for layer in layers:
            tokens, _ = layer(tokens)
        tokens = tokens.transpose(0, 1) + self.offset
        return tokens, torch.zeros(tokens.shape[:2], dtype=torch.bool)


def test_layer_embeddings_capture_every_layer_and_remove_hooks():
    encoder = LayeredEncoder().eval()
    x = torch.linspace(-1, 1, 2 * 32000).reshape(2, 32000)
    layers = extract_layer_embeddings(encoder, x)
    assert layers.shape == (2, 3, 768)
    base = x.reshape(2, 50, 640)[..., 0].mean(dim=1, keepdim=True).expand(-1, 768)
    for i, cumulative in enumerate((1.0, 2.0, 6.0)):
        assert torch.allclose(layers[:, i], base * cumulative)
    assert torch.allclose(layers[:, -1], extract_embeddings(encoder, x))
    assert all(not layer._forward_hooks for layer in encoder.encoder.layers)


def test_layer_embeddings_reject_inconsistent_capture():
    x = torch.linspace(-1, 1, 32000).reshape(1, 32000)
    for encoder in (LayeredEncoder(offset=1.0), LayeredEncoder(skip_last=True)):
        with pytest.raises(ValueError):
            extract_layer_embeddings(encoder.eval(), x)
        assert all(not layer._forward_hooks for layer in encoder.encoder.layers)
