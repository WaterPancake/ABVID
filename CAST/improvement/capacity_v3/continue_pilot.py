"""Finish the analysis of an already running pilot, never starting a fitter."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--eval-id", required=True)
    a = p.parse_args()
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,75}", v) for v in (a.run_id, a.eval_id)):
        p.error("Use simple IDs of at most 76 characters")
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    run = here/"runs"/a.run_id
    if not (run/"lock.json").is_file():
        p.error("Require an existing prepared run")
    receipt = run/(a.eval_id+".continuation.json")
    if receipt.exists():
        p.error("Preserve the previous continuation receipt")
    state = {"run_id": a.run_id, "eval_id": a.eval_id, "state": "waiting_for_existing_pilot",
             "started_utc": datetime.now(timezone.utc).isoformat(), "steps": [], "starts_fitting": False,
             "outer_held_access": False}
    def save():
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2)+"\n"); tmp.replace(receipt)
    save()
    deadline = time.monotonic()+1800
    stage = run/"fit_pilot_summary.json"
    while not stage.exists():
        if time.monotonic() >= deadline:
            state["state"] = "wait_expired_original_fitter_may_still_be_running"; save(); return 4
        time.sleep(5)
    summary = json.loads(stage.read_text())
    if not summary["passed"] or summary["selected"] != 50 or summary["failures"]:
        state["state"] = "pilot_fit_failure"; save(); return 3
    commands = [
        ["audit-fits", "--run-id", a.run_id, "--scope", "pilot"],
        ["source", "--run-id", a.run_id, "--scope", "pilot", "--eval-id", a.eval_id],
        ["audit-source", "--eval-id", a.eval_id],
        ["diagnostics", "paired", "--id", a.run_id, "--scope", "pilot", "--output-id", a.eval_id+"_paired"],
        ["diagnostics", "fits", "--id", a.run_id, "--scope", "pilot", "--output-id", a.eval_id+"_residuals"],
        ["diagnostics", "source", "--id", a.eval_id, "--output-id", a.eval_id+"_figures"],
        ["diagnostics", "gallery", "--id", a.run_id, "--scope", "pilot", "--output-id", a.eval_id+"_audio"],
    ]
    for index, args in enumerate(commands):
        cmd = ["sh", str(here/"run.sh"), *args]
        entry = {"command": cmd, "start_utc": datetime.now(timezone.utc).isoformat(),
                 "log": a.eval_id+f".step{index+1}.log"}
        state["steps"].append(entry); state["state"] = "running_"+args[0]; save()
        print(json.dumps({"step": index+1, "command": args}), flush=True)
        started = time.perf_counter()
        with (run/entry["log"]).open("xb") as log:
            result = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, seconds=time.perf_counter()-started,
                     end_utc=datetime.now(timezone.utc).isoformat()); save()
        if result.returncode:
            state["state"] = "implementation_or_verification_failure"; save(); return result.returncode
    source = json.loads((here/"evaluations"/a.eval_id/"summary.json").read_text())
    passed = source["candidates"][source["selected"]]["passes"]
    state["state"] = "source_passed_review_for_expansion" if passed else "source_scientific_failure"
    state["completed_utc"] = datetime.now(timezone.utc).isoformat(); save()
    return 0 if passed else 3


if __name__ == "__main__":
    sys.exit(main())
