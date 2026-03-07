# Phase-II Closed Loop - Prompt Sequenz (impl-spec aligned)

Ziel: Phase II gemaess `docs/v3/impl_spec_phase_ii.md` implementieren.

Normative Regel:

* Bei Konflikten gilt `docs/v3/impl_spec_phase_ii.md`.
* `run_phase_i.py` und Phase-I-Guards bleiben bestehen.
* Phase II wird additiv aufgebaut (`run_phase_ii.py`).

---

# Prompt 1 - Contracts einziehen (WorldAction / WorldStepResult)

Dateien:

* `capitalmarket/capitalselector/interfaces.py`
* betroffene World-Implementierungen

Aufgabe:

* Fuehre bindende Dataklassen ein:

```python
@dataclass
class WorldAction:
    weights: np.ndarray
    gross_exposure: float = 1.0
    leverage_limit: float = 1.0
    allow_short: bool = False

@dataclass
class WorldStepResult:
    realized_return: float
    costs: float
    channel_returns: np.ndarray
    cost_by_channel: np.ndarray
    freeze: bool = False
```

* Erweitere World-Signatur auf:

```python
def step(self, t: int, action: WorldAction) -> WorldStepResult
```

Akzeptanz:

* Alle World-Pfade liefern `WorldStepResult`.
* Keine Legacy-Signatur ohne `action` im Phase-II-Pfad.

---

# Prompt 2 - Deterministische Action-Validation

Dateien:

* Selector-Policy-Pfad
* ggf. Validator-Modul

Aufgabe:

* Implementiere deterministische Validierung/Normalisierung in identischer Reihenfolge (CPU/CUDA):

1. finite check
2. sign constraints (`allow_short`)
3. leverage constraint
4. exposure normalization

Regeln:

```text
if allow_short == False: weights >= 0 and sum(weights) = gross_exposure
if allow_short == True:  sum(abs(weights)) = gross_exposure
gross_exposure <= leverage_limit
```

Akzeptanz:

* Invalid actions fail fast mit klarer Fehlermeldung.
* Reihenfolge ist backend-stabil.

---

# Prompt 3 - Binding Coupling Rule umsetzen

Dateien:

* World-Transition
* Kernel-Booking

Aufgabe:

* Implementiere den Phase-II-Kern:

```python
world_out = world.step(t, action)
wealth_next = wealth_prev + world_out.realized_return - world_out.costs
```

Wichtig:

* Kein zweites Exposure-Mapping im Kernel.
* Verboten im Phase-II-Path:

```python
wealth += dot(w, r_vec)
```

Akzeptanz:

* Outcome-Kopplung entsteht ausschliesslich via `World.step(action)`.

---

# Prompt 4 - Economic vs Selector State operationalisieren

Dateien:

* Economic-State / Settlement-Event-Pfad
* Selector-State / Core

Aufgabe:

* Fuehre verbindliche Phase-II-States mit klarer Ownership ein:

```text
Economic State (source of truth):
    due_curve[horizon_bin]
    expected_inflows
    liquidity_mismatch

Selector State:
    strategic_credit_exposure
    derived features from Economic State
```

* Implementiere die bindende Formel fuer Economic-State-Observables:

```text
expected_inflows_t = Σ_{h <= H_near} Σ_c w_t(c) * mu_term_t(c, h)
near_term_obligations_t = Σ_{h <= H_near} due_curve_t(h)
liquidity_mismatch_t = near_term_obligations_t - expected_inflows_t
```

* Alle Summen laufen ueber feste Horizon-Bins in deterministischer Iterationsreihenfolge.

* Definiere updatefaehige Datenstrukturen fuer CPU und CUDA.

Akzeptanz:

* Ownership ist explizit (Economic vs Selector) und testbar.

---

# Prompt 5 - Event Contract und Mapping implementieren

Dateien:

* Event-Erzeugung
* Selector-Update-Pfad

Aufgabe:

* Nutze verpflichtende Eventtypen:

```text
RETURN
DUE_CASH
COST
ROLLOVER
FAIL
SETTLEMENT
```

* Implementiere normatives Mapping:

```text
RETURN -> selector reward/statistics update
COST -> cost attribution update
DUE_CASH -> EconomicState.due_curve update
ROLLOVER -> EconomicState.due_curve update
FAIL -> psi update
SETTLEMENT -> EconomicState.liquidity_mismatch update
```

Akzeptanz:

* Event-Attribution ist deterministisch und backend-unabhaengig.

---

# Prompt 6 - EventSummary Boundary Contract

Dateien:

* Settlement/Economic-Event-Pfad
* Selector-Update-Pfad

Aufgabe:

* Fuehre einen deterministischen Boundary-Artefakt ein:

```python
@dataclass
class EventSummary:
    event_counts: dict[str, int]
    last_event: str | None
    channel_event_vector: np.ndarray
```

* Nutze `EventSummary` als stabile Uebergabe zwischen Economic/Settlement und Selector-Updates.

Akzeptanz:

* `EventSummary` wird pro Schritt deterministisch erzeugt.
* Selector-Updates lesen `EventSummary`, nicht implizite Seiteneffekte.

---

# Prompt 7 - Phase-II Tests (State/Validation/Coupling)

Neue Tests:

* `tests/test_due_curve_update.py`
* `tests/test_liquidity_mismatch_detection.py`
* `tests/test_strategic_credit_behavior.py`
* `tests/test_world_action_validation.py`
* `tests/test_world_action_coupling.py`
* `tests/test_event_summary_mapping.py`

Akzeptanz:

* Jeder Pflicht-State und jeder Pflicht-Vertrag hat einen dedizierten Test.

---

# Prompt 8 - Determinismus und CPU/CUDA-Paritaet

Neue Tests:

* `tests/test_phase_ii_determinism.py`
* `tests/test_phase_ii_cpu_cuda_parity.py`

Paritaetsregel:

* Exact-match: terminal flags, event counts, rollover counts, timestep indices
* Numerical parity:

```text
abs(cpu - cuda) <= atol + rtol * abs(cpu)
rtol = 1e-7
atol = 1e-9
```

Akzeptanz:

* Reproduzierbarkeit je Backend ist gruen.
* CPU/CUDA-Paritaet gemessen nach obiger Regel ist gruen.

---

# Prompt 9 - Additive Evaluation Pipeline

Datei:

* `capitalmarket/capitalselector/experiments/run_phase_ii.py`

Aufgabe:

* Implementiere Phase-II-Runner additiv.
* `run_phase_i.py` bleibt unveraendert als Invarianz-Runner.

Mindestmetriken:

* `terminal_wealth`
* `time_to_death`
* `rollover_failure_frequency`

Akzeptanz:

* Phase-I- und Phase-II-Runner koexistieren.
* Phase-I-Guards bleiben in CI aktiv.

---

# Prompt 10 - Statistik und bindende H1/H2

Datei:

* `run_phase_ii.py` + Auswertungsmodul

Aufgabe:

* Verwende fixe Seed-Liste, paired bootstrap, 95%-CI.
* Implementiere bindende Hypothesen:

H1 (Selector-4 robustness):

```text
Delta = time_to_death(S4) - time_to_death(S3)
H1 supported if CI_lower(Delta) > 0
```

H2 (Selector-5 credit stability):

```text
Delta = rollover_failure_frequency(S5) - rollover_failure_frequency(S4)
H2 supported if CI_upper(Delta) < 0
```

Minimum run matrix:

```text
selectors = [Selector-3, Selector-4, Selector-5]
runs >= 100 seeds per selector/world
```

Akzeptanz:

* Keine Single-Run-`!=`-Claims als Evidenz.
* H1/H2-Status kommt aus CI-Entscheidungsregeln.

---

# Prompt 11 - Protokolltest und CI-Absicherung

Neue Datei:

* `tests/test_phase_ii_evaluation_protocol.py`

Aufgabe:

* Teste Artefakterzeugung und Hypothesen-Reporting inkl. CI-Felder.
* Stelle sicher, dass Phase-I-Tests unveraendert weiterlaufen.

CI-Zielbild:

* Phase-I invariant path: gruen
* Phase-II closed-loop path: gruen

---

# Ergebnis (Definition of Done)

Phase II gilt als umgesetzt, wenn:

1. `WorldAction`/`WorldStepResult` verbindlich implementiert sind.
2. World-action coupling ueber `World.step(action)` aktiv ist.
3. Economic-State Observables (`due_curve`, `liquidity_mismatch`) und Selector-5-State (`strategic_credit_exposure`) explizit existieren.
4. Eventmapping deterministisch umgesetzt ist.
5. `EventSummary` als deterministischer Boundary-Artefakt vorhanden ist.
6. Phase-II Testsuite gruen ist.
7. H1/H2 auf paired-bootstrap-95%-CI basiert und reproduzierbar reportet wird.
8. Phase-I-Runner und Phase-I-Guards unveraendert gruen bleiben.
