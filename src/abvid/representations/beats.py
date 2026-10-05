"""Hash-verified, frozen official BEATs inference; no remote code auto-execution."""

import importlib
import hashlib
import json
from pathlib import Path
import sys

import torch

def sha256(path):
    """Keep frozen inference independent of benchmark plotting/training imports."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def masked_mean(tokens, padding_mask=None):
    if tokens.ndim != 3 or not torch.isfinite(tokens).all():
        raise ValueError("expected finite batch x tokens x features")
    if padding_mask is None:
        return tokens.mean(dim=1)
    if padding_mask.shape != tokens.shape[:2] or padding_mask.dtype != torch.bool:
        raise ValueError("invalid token padding mask")
    valid = ~padding_mask
    if not valid.any(dim=1).all():
        raise ValueError("all-padded sample")
    return (tokens * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)


def load_encoder(config):
    checkpoint = Path(config["checkpoint"])
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("BEATs checkpoint hash mismatch")
    root = checkpoint.parent / "upstream"
    provenance = json.loads((checkpoint.parent / "provenance.json").read_text())
    if provenance["upstream_commit"] != config["upstream_commit"]:
        raise ValueError("BEATs upstream revision mismatch")
    for name, entry in provenance["code"].items():
        if sha256(root / name) != entry["sha256"]:
            raise ValueError("BEATs upstream code hash mismatch")
    for name in ("BEATs", "backbone", "modules"):
        if (
            name in sys.modules
            and Path(sys.modules[name].__file__).parent != root.resolve()
        ):
            raise ValueError(f"upstream import name collision: {name}")
    sys.path.insert(0, str(root.resolve()))
    try:
        upstream = importlib.import_module("BEATs")
    finally:
        sys.path.pop(0)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = upstream.BEATsConfig(checkpoint_data["cfg"])
    if cfg.finetuned_model or cfg.encoder_embed_dim != 768:
        raise ValueError("expected pretrained 768D checkpoint, not class predictions")
    encoder = upstream.BEATs(cfg)
    encoder.load_state_dict(checkpoint_data["model"], strict=True)
    encoder.eval().requires_grad_(False)
    return encoder, provenance


@torch.inference_mode()
def extract_embeddings(encoder, waveforms):
    if (
        waveforms.ndim != 2
        or waveforms.shape[1] != 32000
        or not torch.isfinite(waveforms).all()
    ):
        raise ValueError("expected finite 2s/16kHz mono windows")
    if encoder.training or any(p.requires_grad for p in encoder.parameters()):
        raise ValueError("encoder must be frozen and in evaluation mode")
    tokens, mask = encoder.extract_features(
        waveforms, padding_mask=torch.zeros_like(waveforms, dtype=torch.bool)
    )
    embeddings = masked_mean(tokens, mask)
    if (
        embeddings.shape != (len(waveforms), 768)
        or not torch.isfinite(embeddings).all()
    ):
        raise ValueError("invalid BEATs embeddings")
    return embeddings.float()
