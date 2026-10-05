"""Continue an existing full fit through audit and frozen source selection.

This never starts or restarts fitting. All experiment destinations must be new.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-eval-id", required=True)
    parser.add_argument("--wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,69}", x) for x in (args.run_id, args.source_eval_id)):
        parser.error("Use simple IDs of at most 70 characters")
    if not 0 <= args.wait_seconds <= 7200:
        parser.error("Waiting limit must be between 0 and 7200 seconds")
    run = HERE/"runs"/args.run_id
    if not (run/"lock.json").is_file():
        parser.error("Existing frozen run required; this command never starts fitting")
    source = HERE/"evaluations"/args.source_eval_id
    if source.exists():
        parser.error("Source evaluation already exists; preserve its artifacts")
    receipt_path = run/(args.source_eval_id+".continuation.json")
    if receipt_path.exists():
        parser.error("Continuation receipt already exists; inspect the original process")
    receipt = {"run_id": args.run_id, "source_eval_id": args.source_eval_id,
               "started_utc": datetime.now(timezone.utc).isoformat(),
               "starts_fitting": False, "steps": [], "state": "waiting_for_full_summary"}

    def checkpoint():
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2)+"\n")
        temporary.replace(receipt_path)

    checkpoint()
    print("Waiting for the existing full-fit summary; fitting will not be restarted.", flush=True)
    deadline = time.monotonic()+args.wait_seconds
    stage = run/"fit_full_summary.json"
    while True:
        try:
            summary = json.loads(stage.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= deadline:
                receipt["state"] = "wait_limit_reached_without_complete_summary"
                checkpoint()
                raise SystemExit("Full-fit summary not available; inspect the existing fit process before taking action")
            time.sleep(min(5, max(0, deadline-time.monotonic())))
            continue
        if not summary["passed"] or summary["scope"] != "full" or summary["selected"] != 380 or summary["failures"]:
            receipt["state"] = "full_fit_failed"
            checkpoint()
            raise SystemExit("Full fitting did not pass; all artifacts retained")
        break
    commands = [
        ["audit-fits", "--run-id", args.run_id, "--scope", "full"],
        ["source", "--run-id", args.run_id, "--eval-id", args.source_eval_id,
         "--scope", "full", "--temperatures", "1", "1.1", "1.25", "1.5"],
        ["audit-source", "--eval-id", args.source_eval_id],
        ["diagnostics", "source", "--id", args.source_eval_id, "--output-id", args.source_eval_id+"_figures"],
        ["diagnostics", "fits", "--id", args.run_id, "--scope", "full", "--output-id", args.source_eval_id+"_residuals"],
        ["gallery", "--run-id", args.run_id, "--scope", "full", "--output-id", args.source_eval_id+"_audio"],
    ]

    def invoke(arguments):
        command = ["bash", str(HERE/"analyze.sh"), *arguments]
        receipt["state"] = "running_"+arguments[0]
        step = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
        receipt["steps"].append(step)
        checkpoint()
        print("Running: "+" ".join(arguments), flush=True)
        completed = subprocess.run(command, cwd=ROOT)
        step.update(returncode=completed.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
        checkpoint()
        if completed.returncode:
            receipt["state"] = "stage_failed"
            checkpoint()
            raise SystemExit(completed.returncode)

    for command in commands:
        invoke(command)
    selected = json.loads((source/"summary.json").read_text())
    if not selected["candidates"][selected["selected"]]["passes"]:
        receipt.update(state="source_scientific_failure", outer_held_access=False)
        checkpoint()
        print("Full source selection fails scientific criteria. No outer comparison was run.", flush=True)
        raise SystemExit(3)
    outer_id = args.source_eval_id+"_outer"
    invoke(["outer", "run", "--source-eval-id", args.source_eval_id, "--eval-id", outer_id])
    invoke(["outer", "verify", "--eval-id", outer_id])
    verified = json.loads((HERE/"evaluations"/outer_id/"verification.json").read_text())
    receipt.update(state="outer_verified", technical_passed=verified["passed"], scientific_goal_passed=verified["goal_passed"])
    checkpoint()
    print(json.dumps({"technical_passed": verified["passed"], "scientific_goal_passed": verified["goal_passed"]}), flush=True)
    raise SystemExit(0 if verified["passed"] and verified["goal_passed"] else 3)


if __name__ == "__main__":
    main()
