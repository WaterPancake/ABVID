"""Fetch pinned official BEATs code and hash-pinned mirror weights, without execution."""

from pathlib import Path
from urllib.request import Request, urlopen
import shutil

import yaml

from vehicle_audio.benchmark_followup import sha256, write_json


def fetch(url, path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(f"Inspect incomplete download before retry: {partial}")
    with (
        urlopen(
            Request(url, headers={"User-Agent": "ABVID-research/0.1"}), timeout=60
        ) as response,
        partial.open("xb") as output,
    ):
        shutil.copyfileobj(response, output)
    partial.rename(path)


def main():
    config = yaml.safe_load(Path("configs/benchmark_beats_v1.yaml").read_text())
    root = Path(config["checkpoint"]).parent
    commit = config["upstream_commit"]
    origins = {}
    for name in ("BEATs.py", "backbone.py", "modules.py", "README.md", "LICENSE"):
        relative = name if name == "LICENSE" else f"beats/{name}"
        url = f"https://raw.githubusercontent.com/microsoft/unilm/{commit}/{relative}"
        target = root / "upstream" / name
        fetch(url, target)
        origins[name] = {"url": url, "sha256": sha256(target)}
    checkpoint = Path(config["checkpoint"])
    print("Downloading 361.5 MB mirror checkpoint", flush=True)
    fetch(config["mirror_url"], checkpoint)
    if sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint does not match pinned mirror LFS SHA256")
    write_json(
        root / "provenance.json",
        {
            "upstream_commit": commit,
            "code": origins,
            "checkpoint_url": config["mirror_url"],
            "checkpoint_sha256": sha256(checkpoint),
            "official_checksum_identity_verified": False,
            "limitation": "Official download returned HTTP 403; mirror integrity only.",
        },
    )
    print(f"Verified mirror hash: {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
