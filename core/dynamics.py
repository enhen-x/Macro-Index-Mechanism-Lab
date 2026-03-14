"""Core dynamics module for the forced damped system."""

from dataclasses import dataclass
import numpy as np


@dataclass
class SystemParams:
    m: float = 1.0
    c: float = 0.2
    k: float = 0.5
    x0: float = 0.0
    beta_u: float = 1.0


def euler_step(x: float, v: float, u: float, params: SystemParams, dt: float = 1.0) -> tuple[float, float]:
    """One-step Euler discretization for x'' = (-c v - k(x-x0) + beta_u u) / m."""
    a = (-params.c * v - params.k * (x - params.x0) + params.beta_u * u) / params.m
    v_next = v + dt * a
    x_next = x + dt * v_next
    return x_next, v_next


def simulate_response(u: np.ndarray, params: SystemParams, x_init: float = 0.0, v_init: float = 0.0, dt: float = 1.0) -> np.ndarray:
    """Simulate x(t) response for a macro forcing sequence u(t)."""
    x = np.zeros_like(u, dtype=float)
    pos = float(x_init)
    vel = float(v_init)
    for i, ui in enumerate(u):
        pos, vel = euler_step(pos, vel, float(ui), params, dt=dt)
        x[i] = pos
    return x
