"""Run a minimal end-to-end baseline experiment."""

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dynamics import SystemParams, simulate_response
from data_loader.market_data import generate_synthetic_macro
from identification.ols_identifier import identify_ols
from backtest.simulator import mse, mae, directional_accuracy


def main() -> None:
    n = 300
    seed = 42

    u = generate_synthetic_macro(n=n, seed=seed)
    true_params = SystemParams(m=1.0, c=0.25, k=0.45, x0=0.0, beta_u=0.9)

    x = simulate_response(u=u, params=true_params, x_init=0.0, v_init=0.0, dt=1.0)
    x = x + 0.02 * np.random.default_rng(seed).standard_normal(n)

    split = int(n * 0.7)
    x_train, u_train = x[:split], u[:split]
    x_test, u_test = x[split:], u[split:]

    est = identify_ols(x=x_train, u=u_train, dt=1.0)
    est_params = SystemParams(m=est.m, c=est.c, k=est.k, x0=est.x0, beta_u=est.beta_u)
    x_pred_test = simulate_response(u=u_test, params=est_params, x_init=float(x_train[-1]), v_init=0.0, dt=1.0)

    print("=== Estimated Params ===")
    print(est)
    print("=== Test Metrics ===")
    print(f"MSE: {mse(x_test, x_pred_test):.6f}")
    print(f"MAE: {mae(x_test, x_pred_test):.6f}")
    print(f"Directional Accuracy: {directional_accuracy(x_test, x_pred_test):.4f}")


if __name__ == "__main__":
    main()
