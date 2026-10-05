# Numerical environments

Use Python 3.11.13 for matching the recorded runs. The project lock has mutually
exclusive `cast` and `baseline` extras. Use separate `.venv` / `.venv-baseline`
environments; do not substitute one for the other in a historical replay.

- `cast-requirements.lock`: recorded CAST environment (NumPy 2.4.6, Torch 2.13.0).
- `reproduction-requirements.lock`: recorded R0/H1 environment, including optional
  released-paper dependencies (NumPy 1.26.4, Torch/Torchaudio 2.11.0, sklearn 1.4.2).
- `h2-requirements.txt`: original isolated backend-check overlay. The H2 reports
  identify the actual execution environment; this overlay alone is not that lock.

The smaller `baseline` extra supports the refactored H1/H2 numeric modules. Exact
published AST/CNN/ArcFace/DANN checkpoint replays still use their original pinned
reproduction environment and release files. The new package environment is not
retroactive evidence of a historical training reproduction.

Original local environments are preserved in the artifact archive. Invoke their
`bin/python` explicitly; relocated shell entry-point shebangs may still name the
old directory. Do not run `uv sync` inside that immutable archive.
