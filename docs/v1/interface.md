# External Interface Specification (Draft)

This document specifies the **external interface boundary** of the system.
It is normative for components that **interact** with the internal dynamics,
but are **not part** of the core model.

Reference: `architecture.md` §8 and `math-v1.md`.

---

## 1. Interface Principles

- **Non‑intervening:** External components do not mutate internal state.
- **State‑agnostic:** External components must not use internal state as input.
- **Deterministic hand‑off:** Inputs are defined, explicit, and logged.
- **No rewards:** No reward/target signal is allowed.

---

## 2. World Interface

### 2.1 Responsibilities
The World provides **exogenous signals**:
- Channel returns `r_k(t)`
- Aggregate cost input `c(t)` (scalar)

### 2.2 Constraints
- World has **no access** to internal weights, stats, or topology.
- World does **not** receive feedback from the system.
- World outputs are **read‑only** for the system.
- Time discretization (Δt) is internal to the system and not provided by the World in v1.

### 2.3 Minimal Contract
- `step(t) -> { r: array[float], c: float }`

---

## 3. Curriculum Interface

### 3.1 Responsibilities
A Curriculum defines a **sequence of worlds or parameters**.
It may vary:
- return distributions
- cost structures
- channel counts
- time horizons

### 3.2 Constraints
- Curriculum must remain **state‑agnostic**.
- Curriculum may be **parameter‑adaptive** (time‑based), not state‑adaptive.
- No direct modification of weights, stats, topology.

### 3.3 Minimal Contract
- `next(t) -> World`

---

## 4. Teacher Interface (Optional)

### 4.1 Responsibilities
A Teacher may:
- select Curriculum
- choose Profile (A/B)
- choose Mode (D6, D9)
- set configuration parameters

### 4.2 Constraints
A Teacher may **not**:
- modify internal state
- override stability decisions
- inject signals into dynamics

### 4.3 Minimal Contract
- `configure(run_id, profile, mode, params) -> void`

---

## 5. Config & Mode Selection

### 5.1 Profiles (Config Bundles)
Profiles are **internally consistent bundles** (see `math-v1.md` §18–§19):
- Profile A (Canonical / Prod‑v0)
- Profile B (Analytical / Research)

### 5.2 Modes (D6, D9)
Modes are explicit runtime selections:
- D6: alpha dynamics
- D9: sediment representation

**Rule:** Modes cannot override canonical invariants (sediment non‑causal, inhibition semantics).

---

## 6. Observability & Logging

- All external inputs must be logged with timestamps.
- World/Curriculum selection must be recorded per run.
- No hidden channels or side‑effects.

---

## 7. Non‑Goals

- No reward shaping
- No interactive agent feedback
- No stateful external control loops
