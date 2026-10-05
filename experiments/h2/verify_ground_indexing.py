"""Record the bounded indexing repair; not a vehicle experiment or full physics admission."""
import argparse
import ast
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

sys.dont_write_bytecode = True
os.environ.setdefault("MPLBACKEND", "Agg")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+"\n")


def method_nodes(path):
    tree = ast.parse(Path(path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SimulatorManager")
    return {n.name: ast.dump(n, include_attributes=False) for n in cls.body if isinstance(n, ast.FunctionDef)}


def run_suite(suite, output):
    stream = io.StringIO()
    with redirect_stdout(stream), redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    Path(output).write_text(stream.getvalue())
    return {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "passed": result.wasSuccessful()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reference = HERE/"evidence/pyroadacoustics"
    # Audit the retained two-method repair, not later unfinished propagation work.
    backend = HERE/"backend_revisions/indexing_20261004_v1"
    working_backend = HERE/"backend"
    name = "pyroadacoustics/simulatorManager.py"
    upstream = json.loads((backend/"UPSTREAM.json").read_text())
    changed = []
    for file, digest in upstream["upstream_sha256"].items():
        if sha(reference/file) != digest: raise ValueError(f"Reference changed: {file}")
        if sha(backend/file) != digest: changed.append(file)
    if changed != [name]: raise ValueError(f"Unexpected modified backend files: {changed}")
    original = method_nodes(reference/name); repaired = method_nodes(backend/name)
    changed_methods = sorted(k for k in original if original[k] != repaired.get(k))
    expected = ["_get_asphalt_reflection_filter", "_precompute_complex_angle_reflection_filter_table"]
    if changed_methods != expected or original.keys() != repaired.keys():
        raise ValueError(f"Unexpected changes outside indexing: {changed_methods}")
    diff = difflib.unified_diff((reference/name).read_text().splitlines(keepends=True),
                    (backend/name).read_text().splitlines(keepends=True), fromfile="a/"+name, tofile="b/"+name)
    # The upstream file has no trailing newline. difflib does not emit the Git
    # marker, so preserve that distinction explicitly to make the patch usable.
    patch = "".join(line if line.endswith("\n") else line+"\n\\ No newline at end of file\n" for line in diff)
    (out/"ground_indexing.patch").write_text(patch)
    with tempfile.TemporaryDirectory(prefix="abvid-ground-patch-") as temporary:
        shutil.copytree(reference/"pyroadacoustics", Path(temporary)/"pyroadacoustics",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        applied = subprocess.run(["git", "apply", str(out/"ground_indexing.patch")],
                                 cwd=temporary, text=True, capture_output=True)
        if applied.returncode or sha(Path(temporary)/name) != sha(backend/name):
            raise RuntimeError("Patch cannot reproduce backend bytes: "+applied.stderr)
    import test_ground_indexing as tests
    if tests.INDEXING_BACKEND.resolve() != backend.resolve():
        raise ValueError("Regression tests selected the wrong indexing revision")
    regressions = run_suite(unittest.defaultTestLoader.loadTestsFromTestCase(tests.GroundIndexingTests), out/"regression_tests.txt")
    working_regressions = run_suite(
        unittest.defaultTestLoader.loadTestsFromTestCase(tests.WorkingGroundIndexingTests),
        out/"working_indexing_tests.txt")
    # Run the unchanged upstream manager suite against the patched package, with
    # an explicit import-path check so its sys.path append cannot select evidence.
    sys.path.insert(0, str(backend))
    source = reference/"tests/test_simulator_manager.py"
    spec = importlib.util.spec_from_file_location("_abvid_upstream_manager_tests", source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    loaded = Path(sys.modules["pyroadacoustics.simulatorManager"].__file__).resolve()
    if loaded != (backend/name).resolve(): raise ValueError("Upstream tests loaded wrong backend")
    upstream_tests = run_suite(unittest.defaultTestLoader.loadTestsFromModule(module), out/"upstream_manager_tests.txt")
    # The existing verifier checks immutable H2 manifests and 8,257 protected
    # H1/diagnostic/encoder artifacts. It does not access new target predictions.
    check = subprocess.run([sys.executable, str(HERE/"verify_design.py"), "--freeze",
                 str(HERE/"frozen/source_path_20261004_v1"), "--output", str(out/"design_verification.json")],
                 cwd=ROOT, text=True, capture_output=True)
    (out/"design_verification_log.txt").write_text(check.stdout+check.stderr)
    if check.returncode: raise RuntimeError("Design/protected verification failed; see saved log")
    fixed = tests.manager(tests.PATCHED)
    import numpy as np
    cases = []
    for angle in [-89, -60, -30, 0, 30, 60, 85, 89]:
        for distance in [5., 20., 50., 135.]:
            actual = fixed._get_asphalt_reflection_filter(angle, distance)
            expected_filter = tests.analytical_filter(fixed, angle, distance)
            cases.append({"angle_deg": angle, "distance_m": distance,
                          "max_abs_filter_error": float(np.max(np.abs(actual-expected_filter)))})
    save(out/"filter_equation_checks.json", cases)
    gate = json.loads((HERE/"frozen/source_path_20261004_v1/execution_gate.json").read_text())
    remaining = [g for g in gate["unresolved"] if g != "ground_angle_table_indexing"]
    passed = regressions["passed"] and working_regressions["passed"] and upstream_tests["passed"]
    dependencies = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    (out/"environment.freeze.txt").write_text("".join(f"{k}=={v}\n" for k,v in sorted(dependencies.items(), key=lambda v:v[0].lower())))
    for file in [HERE/"test_ground_indexing.py", Path(__file__), HERE/"backend-check-requirements.txt"]:
        (out/("executed_"+file.name)).write_bytes(file.read_bytes())
    report = {"repair_id": "ground_indexing_20261004_v1", "verification_id": out.name,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed, "scope": "ground_angle_indexing_only", "upstream_commit": upstream["upstream_commit"],
        "indexing_backend": str(backend.relative_to(ROOT)),
        "working_backend": str(working_backend.relative_to(ROOT)),
        "working_backend_scope": "indexing contract only; further propagation edits are not admitted by this check",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "design_lock_sha256": sha(HERE/"frozen/source_path_20261004_v1/lock.json"),
        "modified_backend_files": changed, "modified_methods": changed_methods,
        "patch_roundtrip_matches_backend": True,
        "reference_sha256": sha(reference/name), "patched_sha256": sha(backend/name),
        "backend_files_sha256": {file:sha(backend/file) for file in upstream["upstream_sha256"]},
        "working_backend_files_sha256": {file:sha(working_backend/file) for file in upstream["upstream_sha256"]},
        "regression_tests": regressions, "upstream_manager_tests": upstream_tests,
        "working_indexing_tests": working_regressions,
        "filter_equation_checks": len(cases), "max_abs_filter_equation_error": max(r["max_abs_filter_error"] for r in cases),
        "checked_angles_all_table_rows": 179, "frequency_bins": 512,
        "floating_point_note": "Equivalent recomputed filters initially differed by up to 1.39e-17; equivalence checks allow rtol=1e-13, atol=1e-14. Exact waveform hash determinism remains a separate outstanding full-render gate.",
        "physical_spreading_changed": False,
        "physical_spreading_changed_scope": "retained indexing-only backend; the working backend has subsequent unvalidated repairs",
        "full_physics_validated": False,
        "ready_for_target_evaluation": False, "remaining_execution_gates": remaining,
        "artificial_test_sequences_only": True, "vehicle_corpus_generated": False,
        "model_fitting": False, "target_inference": False,
        "python": platform.python_version(), "platform": platform.platform(), "dependencies": dependencies,
        "runtime_seconds": time.perf_counter()-started,
        "artifacts_sha256": {p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    save(out/"verification.json", report)
    print(json.dumps({k:report[k] for k in ["passed", "regression_tests", "working_indexing_tests", "upstream_manager_tests", "max_abs_filter_equation_error", "runtime_seconds", "ready_for_target_evaluation"]}))
    if not passed: raise SystemExit(1)


if __name__ == "__main__": main()
