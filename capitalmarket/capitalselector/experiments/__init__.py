"""Experiment runners (non-kernel utilities)."""

from .g3_3_sweep import run_g3_3_sweep


def run_phase_i_evaluation(*args, **kwargs):
    """Lazy import to avoid runpy warnings for `python -m ...run_phase_i`."""
    from .run_phase_i import run_phase_i_evaluation as _run_phase_i_evaluation

    return _run_phase_i_evaluation(*args, **kwargs)


def run_phase_ii_episode(*args, **kwargs):
    """Lazy import additive closed-loop runner for Phase II."""
    from .run_phase_ii import run_phase_ii_episode as _run_phase_ii_episode

    return _run_phase_ii_episode(*args, **kwargs)


def run_phase_ii_evaluation(*args, **kwargs):
    """Lazy import Phase-II paired-bootstrap evaluation protocol."""
    from .phase_ii_evaluation import run_phase_ii_evaluation as _run_phase_ii_evaluation

    return _run_phase_ii_evaluation(*args, **kwargs)

__all__ = [
    "run_g3_3_sweep",
    "run_phase_i_evaluation",
    "run_phase_ii_episode",
    "run_phase_ii_evaluation",
]
