"""Experiment discovery and explicit frozen-command replay."""
import argparse
import json
import os
import shlex
import subprocess

from .paths import repository_root, roots
from .protocol import command, load


def main(argv=None):
    parser = argparse.ArgumentParser(prog="abvid")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the active research phases")
    commands.add_parser("paths", help="Show local storage locations")
    validate = commands.add_parser("validate", help="Validate one canonical experiment configuration")
    validate.add_argument("config")
    validate.add_argument("--check-files", action="store_true", help="Also verify local contract/script hashes")
    replay = commands.add_parser("replay", help="Print or explicitly execute an existing frozen command")
    replay.add_argument("config")
    replay.add_argument("--execute", action="store_true", help="Execute the displayed historical command")
    args = parser.parse_args(argv)
    try:
        if args.command == "paths":
            print(json.dumps({k: str(v) for k, v in roots().items()}, indent=2))
        elif args.command == "list":
            for path in sorted((repository_root() / "configs/experiments").glob("*.json")):
                cfg = load(path)
                print(f"{cfg['id']:24s} {cfg['phase']:16s} {cfg['description']}")
        elif args.command == "validate":
            cfg = load(args.config, check_files=args.check_files)
            print(json.dumps({"id": cfg["id"], "valid": True, "file_hashes_checked": args.check_files}))
        else:
            cfg = load(args.config, check_files=True)
            cmd, cwd = command(cfg)
            cast_path = os.pathsep.join(str(cwd / p) for p in (
                "CAST/src", "CAST/generalization/src", "CAST/improvement/src"))
            prefix = f"PYTHONPATH={shlex.quote(cast_path)} " if cfg["execution"]["environment"] == "cast" else ""
            print(f"cd {shlex.quote(str(cwd))}\n{prefix}{shlex.join(cmd)}", flush=True)
            if args.execute:
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
                       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
                # No shell, no inherited package path pointing at refactored code.
                env.pop("PYTHONPATH", None)
                if cfg["execution"]["environment"] == "cast":
                    env["PYTHONPATH"] = cast_path
                raise SystemExit(subprocess.run(cmd, cwd=cwd, env=env, check=False).returncode)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
