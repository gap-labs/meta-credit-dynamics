# Phase H / STEP 0 Runbook

This runbook uses the existing Docker-based Make targets only.

## 1) CPU baseline (build + full pytest in container)

```bash
make test-cpu
```

## 2) GPU baseline (build + full pytest in container)

```bash
make test-gpu
```

## 3) Run only the two STEP-0 safety tests in CPU container

```bash
make build-cpu
docker run --rm \
	-v "$PWD":/workspace \
	-w /workspace \
	-u "$(id -u):$(id -g)" \
	dl:cpu bash -lc "PYTHONPATH=/workspace pytest -q tests/test_reference_cpu_phase_g.py tests/test_cuda_full_parity_500_steps.py"
```

## 4) Determinism mode (code-level)

Use the centralized helper:

```python
from capitalmarket.capitalselector.determinism import enable_determinism

enable_determinism(seed=0)
```

Or via runtime config:

```python
from capitalmarket.capitalselector.runtime import RuntimeConfig

cfg = RuntimeConfig(profile="A", deterministic=True, seed=0)
```

Determinism is opt-in and not enabled by default.
