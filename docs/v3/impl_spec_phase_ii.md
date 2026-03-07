# Phase II – Implementation Specification

Status: Draft (normative target)

Purpose: define the **binding implementation contract** for Phase II so that the system implements the Selector‑4 / Selector‑5 mechanisms defined in `math_selector_v3.md` while introducing closed‑loop policy coupling.

This document is the **single authoritative implementation target** for Phase II.

Supersession notice (normative):

For Phase-II implementation decisions, this document supersedes conflicting or older implementation details in:

* `docs/v3/concept_v3.md`
* `docs/v3/architecture_phase_ii.md`
* `docs/v3/phase_ii_state_model.md`

In particular, the binding H1/H2 definitions and acceptance criteria are those defined in this document.

---

# 0. Phase‑II Objective

Phase II introduces a **closed‑loop economic system** where policy decisions influence realized outcomes.

Closed‑loop law:

```
policy → action → world transition → realized returns/costs → wealth → settlement → events → policy
```

However Phase II is **not only closed‑loop**. It must also operationalize the **Selector‑4 / Selector‑5 mechanisms** defined in the math specification:

Selector‑4:

* due‑curve awareness
* maturity structure modelling

Selector‑5:

* liquidity mismatch detection
* strategic credit / maturity transformation

Phase II is accepted only if these mechanisms exist as **explicit state, events, and update rules**.

---

# 1. Delta to Phase I

Phase I guarantees:

* policy‑sensitive internal state
* outcome‑invariant engine
* deterministic execution

Phase II introduces:

1. policy‑conditioned world transitions
2. due‑curve aware credit state
3. liquidity mismatch detection
4. strategic maturity transformation behaviour

Phase II does **not** remove Phase‑I guarantees:

* Phase‑I invariant runner remains mandatory
* deterministic execution remains mandatory
* CPU semantic path remains reference implementation

---

# 2. Binding Coupling Rule

Policy action affects the system **only through the world transition**.

Normative step order:

1. selector computes policy action
2. world receives action
3. world produces realized economic outcome
4. kernel books outcome
5. settlement processes obligations
6. events update selector state

Traceability note (non-normative):

For an implementation-oriented I/O sequence and the spec-to-code-to-test mapping,
see `docs/v3/sanierung_phase_ii.md`:

* `# 6. Specification Trace Matrix (Spec → Code → Test)`
* `# 8. Phase II Step Semantics (reference excerpt from impl_spec_phase_ii.md)`

If wording conflicts exist, this document remains authoritative.

Normative rule:

```
world_out = world.step(t, action)
wealth_next = wealth_prev + world_out.realized_return - world_out.costs
```

Kernel must **not apply exposure transforms** if the world already produced realized returns.

---

# 3. Economic Observables and Selector‑4 / Selector‑5 Contract

Phase II must explicitly support the following state components.

## 3.0 Binding ownership rule

For Phase II, the following ownership is normative:

* Economic State owns canonical observables:
    * `due_curve`
    * `expected_inflows`
    * `liquidity_mismatch`
    * obligations / claims
* Selector State owns learning/control variables:
    * `w`, `mu_term`, `rho`, `psi`
    * `strategic_credit_exposure`

Selector logic may consume derived features from Economic State, but Economic State remains the source of truth for obligations and due-curve observables.

Binding field ownership clarification:

`expected_inflows` is part of `PhaseIIEconomicState`.
It must not be treated as a canonical `PhaseIISelectorState` field.

## 3.1 Due Curve

Economic State maintains a maturity distribution of obligations.

```
due_curve[horizon_bin]
```

Semantic meaning:

Total obligations due in each horizon bucket.

Update source:

* rollover events
* settlement events

Required tests:

```
test_due_curve_update.py
```

---

## 3.2 Liquidity Mismatch

Economic State maintains a scalar measure of short‑term liquidity stress.

`expected_inflows_t` used in this section is the value owned by
`PhaseIIEconomicState.expected_inflows` at time `t`.

Definition:

```
liquidity_mismatch = near_term_obligations - expected_inflows
```

Normative formula (Phase II):

```text
expected_inflows_t = Σ_{h <= H_near} Σ_c w_t(c) * mu_term_t(c, h)
near_term_obligations_t = Σ_{h <= H_near} due_curve_t(h)
liquidity_mismatch_t = near_term_obligations_t - expected_inflows_t
```

Where:

* `H_near` is the configured set of near-term horizon bins (`h <= h_near_idx`)
* all terms are evaluated on the same time slice `t`
* all sums are computed over fixed horizon bins using deterministic iteration order

Near‑term obligations derive from the due curve.

Required tests:

```
test_liquidity_mismatch_detection.py
```

---

## 3.3 Strategic Credit Exposure

Selector may deliberately perform maturity transformation.

This behaviour is represented as

```
strategic_credit_exposure
```

This variable measures the extent to which long‑term lending is financed by shorter obligations.

Required tests:

```
test_strategic_credit_behavior.py
```

## 3.4 Policy Coupling Function (normative)

Policy coupling must be implemented as a deterministic functional mapping.
Threshold-only heuristics (for example `if x > 0: ...`) are not sufficient as the
binding Phase-II rule.

Let:

```text
w_t(c)                : pre-coupling selector weight for channel c
h(c)                  : deterministic horizon-bin mapping for channel c
H_near                : near-term horizon bound
stress_t              = max(0, liquidity_mismatch_t)
near_due_t(c)         = due_curve_t(h(c))
credit_headroom_t     = max(0, expected_inflows_t - near_term_obligations_t)
```

with fixed non-negative constants `alpha`, `beta`, `gamma`, `eta` from config.

Normative update:

```text
u_t(c) = log(max(eps, w_t(c)))
                 - alpha * stress_t * 1[h(c) <= H_near]
                 - beta  * near_due_t(c)
                 + gamma * credit_headroom_t * 1[h(c) > H_near]
                 + eta   * strategic_credit_exposure_t * 1[h(c) > H_near]

w_raw_t(c) = exp(u_t(c))
w'_t(c)    = gross_exposure_t * w_raw_t(c) / Σ_j w_raw_t(j)
```

where `eps > 0` is a fixed deterministic constant.

Requirements:

* The same input tuple must always produce the same `w'_t` on CPU and CUDA.
* `h(c)` must be configuration-defined and deterministic.
* `w'_t` must satisfy the `WorldAction` validation/normalization constraints.
* Implementation-specific alternatives are allowed only if mathematically
    equivalent to this mapping.

---

# 4. WorldAction Contract

The action emitted by the selector must follow a deterministic normalization rule.

```
@dataclass
class WorldAction:
    weights: ndarray[n_channels]
    gross_exposure: float = 1.0
    leverage_limit: float = 1.0
    allow_short: bool = False
```

Normalization rule (normative):

If shorting disabled:

```
weights >= 0
sum(weights) = gross_exposure
```

If shorting enabled:

```
sum(abs(weights)) = gross_exposure
```

Leverage constraint:

```
gross_exposure <= leverage_limit
```

Validation order (must be identical on CPU and CUDA):

1. check finite values
2. enforce sign constraints (if allow_short == False)
3. enforce leverage constraint
4. normalize exposure

Any invalid action must raise a deterministic error.

---

# 5. WorldStepResult Contract

World returns the realized economic outcome.

```
@dataclass
class WorldStepResult:

    realized_return: float
    costs: float

    channel_returns: ndarray
    cost_by_channel: ndarray

    freeze: bool
```

Field semantics:

* `realized_return` : exposure‑conditioned return booked by kernel
* `costs` : aggregate cost booked by kernel
* `channel_returns` : diagnostic vector used for selector statistics
* `cost_by_channel` : channel cost attribution

These fields allow deterministic mapping from world output to selector updates.

---

# 6. Event Contract

Events update Economic observables and selector state.

Mandatory event types:

```
RETURN
DUE_CASH
COST
ROLLOVER
FAIL
SETTLEMENT
```

Event roles:

RETURN
realized income event from world transition

DUE_CASH
obligation payment due at settlement horizon

COST
operational or funding cost event

ROLLOVER
obligation extended to future horizon

FAIL
settlement failure

SETTLEMENT
successful settlement of due obligation

Event → state mapping (normative):

```
RETURN -> selector reward / statistics update (derived from Economic/World outputs)
COST -> cost attribution update
DUE_CASH -> EconomicState.due_curve update
ROLLOVER -> EconomicState.due_curve update
FAIL -> psi update
SETTLEMENT -> EconomicState.liquidity_mismatch update
```

Event attribution must be deterministic and backend‑independent.

## 6.1 EventSummary (binding integration artifact)

To stabilize the world -> settlement -> selector boundary, each step must produce an `EventSummary` artifact.

```python
@dataclass
class EventSummary:
    event_counts: dict[str, int]
    last_event: str | None
    channel_event_vector: ndarray
```

Semantics:

* `event_counts`: cumulative or step-local counts over mandatory event kinds
* `last_event`: last event kind emitted in deterministic step order
* `channel_event_vector`: deterministic per-channel event attribution vector used by selector updates

Ownership and role:

* `EventSummary` is not a new source of truth for obligations.
* It is a deterministic integration artifact between Economic/Settlement outputs and Selector updates.

Update timing (normative):

1. world transition
2. kernel booking
3. settlement
4. construct `EventSummary`
5. selector updates from `EventSummary` + economic observables

---

# 7. Determinism Contract

Determinism must be testable.

## Same‑backend reproducibility

For fixed seed and configuration:

```
run(seed) == run(seed)
```

This must hold for CPU and CUDA independently.

Mandatory exact‑match fields:

* terminal state
* event counts
* rollover counts
* timestep indices

## CPU/CUDA parity

Numerical parity rules:

```
abs(cpu - cuda) <= atol + rtol * abs(cpu)
```

Default tolerances:

```
rtol = 1e-7
atol = 1e-9
```

These tolerances apply to:

* weights
* wealth
* returns
* selector statistics

---

# 8. Evaluation Contract

Phase II evaluation is **additive**.

```
run_phase_i.py
run_phase_ii.py
```

Phase‑I invariance tests remain mandatory.

## 8.1 Episode Termination Rule (normative)

Simulation terminates immediately after the first death event.

Normative metric semantics:

```
time_to_death = timestep index of the first terminal event
```

No further steps are executed after death.

`terminal_wealth` must be the wealth recorded at that terminal step.

---

# 9. Statistical Hypothesis Testing

Phase‑II hypothesis testing must follow the evaluation protocol defined in the concept documents.

Requirements:

* fixed seed list
* paired comparison per seed
* paired bootstrap
* 95% confidence interval

Evaluation artifacts must report:

```
terminal_wealth
time_to_death
rollover_failure_frequency
```

Hypothesis support must be derived from bootstrap confidence intervals.

Single‑run inequality checks are not sufficient.

## 9.1 Phase‑II Hypotheses (Binding)

H1 – Selector‑4 robustness

Selector‑4 improves survival relative to Selector‑3.

Comparison:

Selector‑4 vs Selector‑3

Metric:

```
time_to_death
```

Decision rule:

Let

```
Δ = time_to_death(Selector‑4) − time_to_death(Selector‑3)
```

H1 supported if

```
CI_lower(Δ) > 0
```

---

H2 – Selector‑5 credit stability

Selector‑5 reduces rollover failures relative to Selector‑4.

Comparison:

Selector‑5 vs Selector‑4

Metric:

```
rollover_failure_frequency
```

Decision rule:

Let

```
Δ = rollover_failure_frequency(Selector‑5)
    − rollover_failure_frequency(Selector‑4)
```

H2 supported if

```
CI_upper(Δ) < 0
```

Evaluation must use the same fixed seed list for all compared selectors.

Minimum run matrix:

```
selectors = [Selector‑3, Selector‑4, Selector‑5]
worlds    = evaluation_worlds
runs      ≥ 100 seeds per configuration
```

---

# 10. Required Test Suite

Phase II must include the following tests.

```
test_due_curve_update.py

test_liquidity_mismatch_detection.py

test_strategic_credit_behavior.py

test_world_action_validation.py

test_world_action_coupling.py

test_phase_ii_determinism.py

test_phase_ii_cpu_cuda_parity.py

test_phase_ii_evaluation_protocol.py

test_event_summary_mapping.py
```

Phase‑I tests must remain unchanged and continue to pass.

---

# 11. Acceptance Criteria

Phase II is complete when:

1. selector emits validated `WorldAction`
2. world transition depends on action
3. due‑curve state exists and updates deterministically
4. liquidity mismatch metric exists and is sourced from Economic State
5. strategic credit exposure metric exists
6. world returns include channel diagnostics
7. event mapping updates selector statistics
8. `EventSummary` is emitted deterministically and consumed by selector updates
9. Phase‑I invariance runner remains green
10. Phase‑II evaluation runner produces bootstrap artifacts
11. CPU/CUDA parity tests pass under documented tolerances
12. Episode terminates at first death event with no post-terminal continuation
13. Policy coupling uses the normative deterministic function from section 3.4

---

# 12. Architecture Law

Phase II obeys the following invariant architecture rule:

```
Policy influences outcomes only through World.step(action).
Kernel books realized results.
Economic observables and selector-derived state update through deterministic event mapping.
Phase‑I invariance remains protected.
Phase‑II evaluation uses paired bootstrap statistics.
```
