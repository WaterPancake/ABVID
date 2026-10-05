import argparse
import re
import torch
from .pipeline import EXT, verify, workflow


def main():
    parser = argparse.ArgumentParser(description="Frozen CAST held-group acoustic coverage, without classifier training")
    parser.add_argument("command", choices=["workflow", "verify"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", args.run_id):
        parser.error("Use a simple alphanumeric run ID")
    out = EXT/"runs"/args.run_id
    if args.command == "workflow":
        if out.exists():
            parser.error("Run already exists; choose a new run ID")
        workflow(out)
    else:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        print(verify(out))


if __name__ == "__main__":
    main()
