"""python -m abvid.physics run/replay --help"""
import argparse
from .workflow import DEFAULT_CONFIG, replay, run


def main():
    parser = argparse.ArgumentParser(description="Bounded RPM dataset audit and physics/CAST comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("run")
    start.add_argument("--config", default=DEFAULT_CONFIG)
    start.add_argument("--output", required=True, help="New artifact:physics_engine/... directory; never overwritten")
    start.add_argument("--download", action="store_true", help="Download only pinned missing metadata and 16 selected WAVs")
    check = sub.add_parser("replay")
    check.add_argument("--output", required=True)
    check.add_argument("--refit-first", action="store_true", help="Also repeat all three complete fits on the first selected clip")
    args = parser.parse_args()
    if args.command == "run":
        if not run(args.config, args.output, args.download):
            raise SystemExit(1)
    else:
        replay(args.output, refit_first=args.refit_first)


if __name__ == "__main__":
    main()
