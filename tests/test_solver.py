"""Tests unitaires et de régression pour epidemio_sim.solvers.ode_solver."""

from __future__ import annotations

import numpy as np
import pytest

from epidemio_sim.solvers.ode_solver import euler_explicit, solve_ode


def _exp_decay_rhs(t, y, params):
    (k,) = params
    return -k * y


def test_solve_ode_matches_known_analytic_solution():
    k = 0.7
    y0 = np.array([2.0])
    t_eval = np.linspace(0.0, 5.0, 50)
    result = solve_ode(_exp_decay_rhs, y0, (0.0, 5.0), np.array([k]), t_eval=t_eval)

    expected = 2.0 * np.exp(-k * t_eval)
    np.testing.assert_allclose(result.y[0], expected, rtol=1e-6, atol=1e-8)


def test_solve_ode_raises_on_failure():
    def blowing_up_rhs(t, y, params):
        return np.array([np.nan])

    with pytest.raises(RuntimeError):
        solve_ode(blowing_up_rhs, np.array([1.0]), (0.0, 1.0), np.array([]))


def test_t_eval_is_respected():
    t_eval = np.linspace(0.0, 2.0, 11)
    result = solve_ode(_exp_decay_rhs, np.array([1.0]), (0.0, 2.0), np.array([0.3]), t_eval=t_eval)
    np.testing.assert_array_equal(result.t, t_eval)


def test_solve_ode_is_model_agnostic():
    # Aucune classe EpidemicModel impliquée : rhs est une simple lambda.
    rhs = lambda t, y, params: np.array([params[0] * y[0]])  # noqa: E731
    result = solve_ode(rhs, np.array([1.0]), (0.0, 1.0), np.array([0.5]))
    assert result.success


def test_euler_explicit_converges_as_dt_shrinks():
    k = 1.0
    t_span = (0.0, 2.0)
    y0 = np.array([1.0])
    expected_final = np.exp(-k * t_span[1])

    coarse = euler_explicit(_exp_decay_rhs, y0, t_span, np.array([k]), n_steps=10)
    fine = euler_explicit(_exp_decay_rhs, y0, t_span, np.array([k]), n_steps=1000)

    error_coarse = abs(coarse.y[0, -1] - expected_final)
    error_fine = abs(fine.y[0, -1] - expected_final)
    assert error_fine < error_coarse

    rk45 = solve_ode(_exp_decay_rhs, y0, t_span, np.array([k]))
    error_rk45 = abs(rk45.y[0, -1] - expected_final)
    assert error_rk45 < error_coarse


def test_euler_explicit_not_wired_into_simulate():
    import inspect

    from epidemio_sim.core import EpidemicModel

    source = inspect.getsource(EpidemicModel.simulate)
    assert "euler" not in source.lower()
