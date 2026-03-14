"""Baseline identifier using finite differences + least squares."""

from dataclasses import dataclass
import numpy as np


@dataclass
class EstimatedParams:
    m: float
    c: float
    k: float
    x0: float
    beta_u: float


def identify_ols(x: np.ndarray, u: np.ndarray, dt: float = 1.0) -> EstimatedParams:
    if len(x) != len(u):
        raise ValueError("x and u must have same length")
    if len(x) < 5:
        raise ValueError("need at least 5 samples")

    v = np.gradient(x, dt)
    a = np.gradient(v, dt)
    x0 = float(np.mean(x))

    # a = b1 * v + b2 * (x - x0) + b3 * u
    X = np.column_stack([v, x - x0, u])
    coef, *_ = np.linalg.lstsq(X, a, rcond=None)

    b1, b2, b3 = coef
    m = 1.0
    c = float(-b1 * m)
    k = float(-b2 * m)
    beta_u = float(b3 * m)

    return EstimatedParams(m=m, c=c, k=k, x0=x0, beta_u=beta_u)
