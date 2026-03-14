import numpy as np

from core.dynamics import SystemParams, simulate_response
from data_loader.market_data import generate_synthetic_macro
from identification.ols_identifier import identify_ols


def test_smoke_pipeline() -> None:
    u = generate_synthetic_macro(80, seed=1)
    x = simulate_response(u, SystemParams(), x_init=0.0, v_init=0.0)
    est = identify_ols(x, u)

    assert np.isfinite(est.c)
    assert np.isfinite(est.k)
    assert np.isfinite(est.beta_u)
