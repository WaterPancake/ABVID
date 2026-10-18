"""P0 CLI: validate, make fixtures, blind review, trim, and audit reuse."""
from __future__ import annotations

import argparse
import json

from abvid.paths import resolve

from .config import DEFAULT_CONFIG, external_path, load
from .dedup import audit_manifest, with_reuse_links
from .fixtures import create_bundle
from .manifest import readiness, validate
from .trim import _write_json_new, export_segments, import_annotations, make_packet


def _read(reference):
    return json.loads(resolve(reference).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Synthetic-only military-signature P0 preparation")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("fixtures", help="Generate artificial chirp/noise fixtures outside Git")
    p.add_argument("--output", required=True)
    p = subs.add_parser("validate-p0", help="Run an artificial end-to-end preparation check")
    p.add_argument("--fixtures", required=True)
    p.add_argument("--output", required=True)
    for name in ["validate", "readiness", "dedup", "review", "import-annotations", "export"]:
        p = subs.add_parser(name)
        p.add_argument("--manifest", required=True)
        if name in {"dedup", "review", "import-annotations", "export"}:
            p.add_argument("--output", required=True)
        if name == "review":
            p.add_argument("--private-mapping", required=True)
            p.add_argument("--seed", type=int, default=17)
        if name == "import-annotations":
            p.add_argument("--batch", required=True)
            p.add_argument("--private-mapping", required=True)
            p.add_argument("--annotations", required=True)
            p.add_argument("--role", choices=["development", "validation", "evaluation"], default="development")
    p = subs.add_parser("group")
    p.add_argument("--manifest", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = load(args.config)
    if args.command == "validate-p0":
        from .validation import run
        result = run(args.fixtures, args.output, config_reference=args.config)
    elif args.command == "fixtures":
        result = create_bundle(args.output, config)
        result = {"manifest_ref": args.output + "/manifest.json", "assets": len(result["assets"])}
    else:
        manifest = _read(args.manifest)
        validate(manifest, config)
        if args.command == "validate":
            result = {"valid": True, "scope": "synthetic_fixture_only", "assets": len(manifest["assets"]),
                      "segments": len(manifest["segments"])}
        elif args.command == "readiness":
            result = readiness(manifest, config)
        elif args.command == "review":
            result = make_packet(manifest, config, args.output, args.private_mapping, seed=args.seed)
        elif args.command == "export":
            result = export_segments(manifest, config, args.output)
            result = {"output_ref": args.output, "clips": len(result["clips"]), "rejections": len(result["rejections"])}
        else:
            if args.command == "dedup":
                result = audit_manifest(manifest, config)
            elif args.command == "group":
                result = with_reuse_links(manifest, _read(args.audit), config)
            else:
                result = import_annotations(manifest, config, _read(args.batch), _read(args.private_mapping),
                                            _read(args.annotations), role=args.role)
            _write_json_new(external_path(args.output), result)
            result = {"output_ref": args.output, "command": args.command}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
