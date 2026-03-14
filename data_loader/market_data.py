"""Simple data loader and synthetic data generator."""

import numpy as np


def generate_synthetic_macro(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    trend = np.linspace(-0.2, 0.2, n)
    cyc = 0.3 * np.sin(np.linspace(0, 6 * np.pi, n))
    noise = 0.1 * rng.standard_normal(n)
    return trend + cyc + noise
