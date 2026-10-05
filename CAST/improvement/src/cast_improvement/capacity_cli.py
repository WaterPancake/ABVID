"""Frozen numerical and fitting entry point for the capacity-v3 revision."""
import argparse
import re
from .capacity_data import HERE, prepare, synthetic, fit_bank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "synthetic", "fit"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--scope", choices=["pilot", "full"], default="pilot")
    p.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", a.run_id):
        p.error("Use a simple run ID")
    out = HERE/"runs"/a.run_id
    if a.command == "prepare":
        prepare(out)
    elif a.command == "synthetic":
        synthetic(out, a.workers)
    else:
        fit_bank(out, a.scope, a.workers)


if __name__ == "__main__":
    main()
