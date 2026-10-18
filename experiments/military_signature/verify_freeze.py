"""Thin read-only verification of the P0 draft freeze."""
from abvid.signature.freeze import verify


if __name__ == "__main__":
    import json
    print(json.dumps(verify(), indent=2))
