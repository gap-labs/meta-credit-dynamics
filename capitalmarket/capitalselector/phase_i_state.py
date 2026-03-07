from __future__ import annotations

import numpy as np


DEFAULT_TERM_GAMMA = np.asarray([1.0, 0.9, 0.7], dtype=float)
DEFAULT_LAMBDA_RISK = 0.5


def allocate_term_mu(channel_count: int, horizon_count: int) -> np.ndarray:
    channels = max(0, int(channel_count))
    horizons = max(1, int(horizon_count))
    return np.zeros((channels, horizons), dtype=float)


def ewma_update(previous: float, value: float, beta: float) -> float:
    beta_f = float(beta)
    return (1.0 - beta_f) * float(previous) + beta_f * float(value)


def term_channel_score(mu_term: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu_term, dtype=float)
    gam = np.asarray(gamma, dtype=float)

    if mu.ndim != 2:
        raise ValueError("mu_term must be a 2D array [channel, horizon]")
    if gam.ndim != 1:
        raise ValueError("gamma must be a 1D array [horizon]")
    if mu.shape[1] != gam.shape[0]:
        raise ValueError("mu_term horizon axis must match gamma length")

    if mu.shape[0] == 0:
        return np.zeros(0, dtype=float)

    return mu @ gam


def validate_lambda_risk(value: float) -> float:
    lambda_risk = float(value)
    if lambda_risk <= 0.0:
        raise ValueError("lambda_risk must be > 0")
    return lambda_risk


def term_risk_channel_score(mu_term: np.ndarray, gamma: np.ndarray, rho: np.ndarray, lambda_risk: float) -> np.ndarray:
    term_score = term_channel_score(mu_term, gamma)
    rho_vec = np.asarray(rho, dtype=float)
    if rho_vec.ndim != 1:
        raise ValueError("rho must be a 1D array [channel]")
    if rho_vec.shape[0] != term_score.shape[0]:
        raise ValueError("rho length must match channel count")

    lam = validate_lambda_risk(lambda_risk)
    return term_score - lam * rho_vec
