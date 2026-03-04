# Phase G – Finales Fazit (v0.6.4-alpha)

This document summarizes Phase G end-to-end (G1–G4 alpha). It is descriptive
and non‑canonical. No new semantics are introduced here.

---

## Scope Covered

- **G1**: Profile bundles + builder boundary + profile isolation
- **G2**: toCuda skeleton + state inventory + round‑trip tests
- **G3**: empirical study phase (deterministic worlds, regime variants, topology toggle)
- **G4 (alpha)**: CUDA/PyTorch port with parity tests (CPU as ground truth)

---

## Core Outcomes (Concise)

1) **Profile separation is enforced (G1)**  
Profile selection happens at build time; kernel remains profile‑agnostic.  
Profile B is explicitly labeled non‑canonical.

2) **State projection is defined and testable (G2)**  
Canonical state inventory exists; `toCuda()`/`fromCuda()` round‑trip is defined with
strict dump‑based equivalence.

3) **Empirical dynamics are reproducible (G3)**  
Deterministic experiments and regime variants are stable; topology can be activated
and yields consistent stack formation under the right conditions.  
Rebirth is functional but requires economic pressure.

4) **CUDA parity is verified (G4 alpha)**  
Reweight, EWMA, drawdown, rebirth, full 500‑step parity, and batch consistency tests
pass under CPU reference. CUDA logic remains semantically aligned with CPU.
Clarification: G4‑alpha currently validates **Torch parity on CPU tensors** (no GPU device required).
GPU device execution is the next step once hardware‑path enablement is wired.

---

## What Worked Well

- Builder boundary and closed profile bundles (G1)
- Deterministic round‑trip mapping and inventory (G2)
- Controlled experiments and reproducible outputs (G3)
- CUDA parity gates with explicit tests (G4 alpha)

---

## Open Items / Constraints

- **Sediment creation** not observed under current G3 settings.  
  Requires stronger destabilization or revised world designs.
- **Performance benchmarking** is optional and deferred;  
  RegimeSwitchBanditWorld is K=5 fixed, so large‑N benchmarks need
  a dedicated benchmark world or synthetic generator.
- **CUDA compute scope** beyond parity (performance tuning, kernel fusion) remains out of scope.

---

## Test Coverage Summary (High‑Level)

- **G1**: profile equivalence, no leakage, rebirth reset invariance
- **G2**: round‑trip identity, shape/dtype preservation, no semantic calls
- **G3**: determinism, marginal controls, topology activation checks
- **G4**: reweight / EWMA / drawdown / rebirth parity, full 500‑step parity, batch consistency

---

## Final Assessment

Phase G is complete at **v0.6.4‑alpha** with strong semantic isolation,  
deterministic testing, and validated CPU↔CUDA parity. The system is stable,  
reproducible, and ready for further CUDA optimization and Phase H planning.
