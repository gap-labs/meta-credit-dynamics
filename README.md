# Meta-Credit Dynamics

> Note: Public snapshot of the meta-credit-dynamics research project. Development happens in a private repository.

This repository contains a minimal research implementation exploring
**repair, stabilization, and meta-credit dynamics** in capital-like systems.

The focus is on **forensic analysis**, not optimization:
repair mechanisms are introduced only to observe how they delay collapse,
shift risk, and create emergent broker paths.

The project is intentionally:
- non-optimizing
- semantically minimal
- scale-invariant
- research-oriented

## Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gap-labs/meta-credit-dynamics/blob/main/notebooks/team_demo.ipynb)

Public demo notebook:
- `notebooks/team_demo.ipynb` (generated from `notebooks/team.ipynb` during publish)

Notes:
- The Colab link opens the notebook directly.
- For security reasons, Colab does not auto-run notebooks via URL.
- In Colab, run `Runtime -> Run all` once.

## Structure

- `capitalmarket/` – core implementation (runtime, selectors, repair/stabilization, telemetry)
- `capitalselector/` – package namespace entrypoint
- `docs/` – public-facing specs, architecture notes, and release notes
- `notebooks/` – public demo notebook (`team_demo.ipynb`, generated during publish)
- `tests/` – representative, reproducible tests for invariants and phase behavior

## Status

Phase D (Repair & Stabilization) is complete.

Phase F / G (production-readiness path in v1) are complete.

Phase H (GPU runtime contract and backend proof path in v2) is complete.

Current public snapshot references release **v0.7.20**
with updated notebook-based regime comparison and robustness analysis.

Reference docs:
- v1 baseline: `docs/v1/math-v1.md`, `docs/v1/architecture.md`, `docs/v1/interface.md`
- v2 baseline: `docs/v2/math-v2.md`, `docs/v2/architecture.md`, `docs/v2/README.md`
- release notes: `docs/v2/release_notes_v0.7.20.md`

### Tests
Run tests with pytest (no Makefile required):

- All tests: `pytest -q`
- CPU-focused run (skip CUDA/GPU tests): `CAPM_SKIP_CUDA_TESTS=1 pytest -q`

Optional public-facing subset:
- `pytest -q tests/test_invariants.py tests/tests_phase_c.py tests/tests_phase_d.py tests/tests_phase_e.py`

## License

This project is released under a **restricted research-use license**.
See `LICENSE.md` for details.

For commercial use, derivative works, or extended permissions,
please contact the author.
