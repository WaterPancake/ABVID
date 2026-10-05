"""Score an actual participant export against the private answer key."""

import argparse
import json
from pathlib import Path

from vehicle_audio.benchmark_followup import sha256, write_json
from vehicle_audio.listening import score_responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument(
        "--key",
        type=Path,
        default=Path(".artifacts/blind_listening_v1/private/answer_key.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    key, response = (json.loads(p.read_text()) for p in (args.key, args.responses))
    write_json(
        args.output,
        {
            "domain": "human native-real development listening pilot",
            "metrics": score_responses(key, response),
            "key_sha256": sha256(args.key),
            "responses_sha256": sha256(args.responses),
            "familiarity": response.get("familiarity"),
            "warning": "Longer excerpts repeat short-block events; one listener is not population evidence.",
        },
    )


if __name__ == "__main__":
    main()
