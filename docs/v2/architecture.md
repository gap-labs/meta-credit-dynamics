# Architecture Overview – meta-credit-dynamics (Phase H1 / H2)

Status: Phase H2 complete
Scope: CPU + CUDA Runtime, Deterministic Lifecycle, Publish Policy, Meta Layer

---

# 1. Architectural Principles

1. **Closed Semantic Domain**
   CUDA Hot Path contains no CPU fallback logic.

2. **CPU as Oracle**
   CPU implementation defines semantic ground truth.

3. **Determinism First**
   All GPU operations must preserve deterministic replay under fixed seed.

4. **Explicit Publish Contract**
   Device→Host synchronization is governed by explicit policy.

5. **Test-Gated Evolution**
   Structural refactors are protected by parity + determinism + guardrail tests.

---

# 2. High-Level Component Architecture

```plantuml
@startuml
package "Core Semantics" {
  [math-v2]
  [Lifecycle Order (Chapter 6)]
}

package "Runtime Layer" {
  [CPU Core]
  [CUDA Core]
  [Publish Policy]
}

package "Meta Layer" {
  [Population Manager]
  [Fitness Accumulation]
  [Reweight Logic]
}

package "Infrastructure" {
  [Tests]
  [Profiler]
  [Makefile Targets]
}

[math-v2] --> [CPU Core]
[math-v2] --> [CUDA Core]

[CUDA Core] --> [Publish Policy]
[Publish Policy] --> [Population Manager]

[Population Manager] --> [Reweight Logic]
[Population Manager] --> [Fitness Accumulation]

[Tests] --> [CPU Core]
[Tests] --> [CUDA Core]
[Profiler] --> [CUDA Core]
@enduml
```

---

# 3. CUDA Runtime Structure

```plantuml
@startuml
class CudaCore {
  +step_with_tau()
  -_publish_selector_runtime()
}

class DeviceState
class KernelSemanticsCuda
class LifecycleCuda
class PublishPolicy

CudaCore --> DeviceState
CudaCore --> KernelSemanticsCuda
CudaCore --> PublishPolicy
KernelSemanticsCuda --> LifecycleCuda

@enduml
```

Responsibilities:

* **DeviceState**: Owns all tensor state (wealth, claims, masks, etc.)
* **KernelSemanticsCuda**: Pure tensor transformation (Chapter 6 phases)
* **LifecycleCuda**: Orchestrates semantic phase order
* **PublishPolicy**: Controls D2H synchronization surface
* **CudaCore**: Runtime wrapper coordinating state transitions

---

# 4. Lifecycle Sequence (One Tau Step)

```plantuml
@startuml
participant PopulationManager
participant CudaCore
participant KernelSemanticsCuda
participant PublishPolicy

PopulationManager -> CudaCore: step_with_tau(tau)
CudaCore -> KernelSemanticsCuda: state -> state'
KernelSemanticsCuda --> CudaCore: updated state
CudaCore -> PublishPolicy: publish(state')
PublishPolicy --> PopulationManager: minimal/full sync surface
PopulationManager -> PopulationManager: fitness + reweight
@enduml
```

Sequence Notes:

* Kernel executes entirely on device
* Publish layer decides sync surface
* Meta layer may influence next step

---

# 5. State Model (CUDA)

```plantuml
@startuml
class DeviceState {
  wealth
  weights
  claim_amount
  claim_target
  claim_active_mask
  dead_mask
  generation
}
@enduml
```

All fields are device-resident tensors.

Slot model:

* Structure-of-Arrays (SoA)
* Fixed `max_claims_per_process`
* Prefix-append insertion
* Deterministic prefix-compaction

---

# 6. Publish Policy Modes

| Mode    | Purpose       | D2H Surface   |
| ------- | ------------- | ------------- |
| minimal | Bench / Prod  | Required only |
| full    | Tests / Debug | Full mirror   |

Required surface typically includes:

* wealth
* necessary masks
* reweight inputs

Not required:

* full claim dump
* full weight vector (unless needed by meta logic)

---

# 7. Test Architecture

```plantuml
@startuml
package "Parity Tests" {
  [CPU vs CUDA parity]
}

package "Determinism Tests" {
  [Seed reproducibility]
}

package "Guardrails" {
  [No process loop]
  [No CPU fallback]
}

package "Bench" {
  [MassSim]
}

[Parity Tests] --> [CUDA Core]
[Determinism Tests] --> [CUDA Core]
[Guardrails] --> [CUDA Core]
[Bench] --> [CUDA Core]
@enduml
```

---

# 8. Evolution Summary

Phase H1:

* CUDA kernel path
* Deterministic state contract
* GPU validation

Phase H2:

* Scalar hygiene
* Batch core
* GPU-native claim ingestion
* Prefix-compaction
* Publish policy hardening

---

# 9. Known Constraints

* Single-GPU architecture
* Fixed slot capacity
* CPU-based meta layer (fitness + reweight)
* Determinism prioritized over maximal fusion

---

# 10. Design Philosophy

Correctness → Determinism → Structural Clarity → Performance

Never invert that order.

---

End of Architecture Document
