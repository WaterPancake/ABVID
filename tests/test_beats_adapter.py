import pytest
import torch

from vehicle_audio.beats_adapter import extract_embeddings, masked_mean


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
