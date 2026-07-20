"""Tests d'intégration pour epidemio_sim.calibration.fit et epidemio_sim.io.loaders."""

from __future__ import annotations

import numpy as np
import pytest

from epidemio_sim.calibration.fit import (
    calibrate_with_scipy,
    compare_calibrations,
    residuals,
)
from epidemio_sim.calibration.optimizers import SGD, Adam, Momentum, RMSprop
from epidemio_sim.core.models import SIRD, SIRDVital
from epidemio_sim.io.loaders import generate_synthetic_data

T_GRID = np.linspace(0.0, 60.0, 61)


def test_generate_synthetic_data_reproducible():
    model = SIRD(population=1000.0)
    true_params = SIRD.params(beta=0.4, gamma=0.1, mu=0.01)
    initial_state = {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}

    data1 = generate_synthetic_data(model, initial_state, true_params, T_GRID, seed=42)
    data2 = generate_synthetic_data(model, initial_state, true_params, T_GRID, seed=42)
    data3 = generate_synthetic_data(model, initial_state, true_params, T_GRID, seed=43)

    for c in data1.observations:
        np.testing.assert_array_equal(data1.observations[c], data2.observations[c])
    assert not np.allclose(data1.observations["I"], data3.observations["I"])


def test_generate_synthetic_data_noise_scales_with_level():
    model = SIRD(population=1000.0)
    true_params = SIRD.params(beta=0.4, gamma=0.1, mu=0.01)
    initial_state = {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}
    sim = model.simulate(initial_state, true_params, t_span=(0.0, 60.0), t_eval=T_GRID)

    low = generate_synthetic_data(
        model, initial_state, true_params, T_GRID, noise_level=0.01, seed=1
    )
    high = generate_synthetic_data(
        model, initial_state, true_params, T_GRID, noise_level=0.3, seed=1
    )

    std_low = np.std(low.observations["I"] - sim.get("I"))
    std_high = np.std(high.observations["I"] - sim.get("I"))
    assert std_high > std_low


def test_residuals_near_zero_at_true_params_noiseless():
    model = SIRD(population=1000.0)
    true_params = SIRD.params(beta=0.4, gamma=0.1, mu=0.01)
    initial_state = {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}
    data = generate_synthetic_data(
        model, initial_state, true_params, T_GRID, noise_level=0.0, seed=0
    )

    r = residuals(np.log(true_params), model, initial_state, data)
    np.testing.assert_allclose(r, np.zeros_like(r), atol=1e-6)


def test_calibrate_with_scipy_recovers_true_params():
    model = SIRD(population=1000.0)
    true_params = SIRD.params(beta=0.4, gamma=0.1, mu=0.01)
    initial_state = {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}
    data = generate_synthetic_data(
        model, initial_state, true_params, T_GRID, noise_level=0.03, seed=7
    )

    x0 = true_params * 1.3
    result = calibrate_with_scipy(model, initial_state, data, x0)

    relative_error = np.abs(result.fitted_params - true_params) / true_params
    assert np.all(relative_error < 0.15)


@pytest.mark.parametrize("optimizer_cls", [SGD, Momentum, RMSprop, Adam])
def test_calibrate_with_each_custom_optimizer_recovers_true_params(optimizer_cls):
    model = SIRD(population=1000.0)
    true_params = SIRD.params(beta=0.4, gamma=0.1, mu=0.01)
    initial_state = {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}
    data = generate_synthetic_data(
        model, initial_state, true_params, T_GRID, noise_level=0.03, seed=7
    )

    x0 = true_params * 1.3
    optimizers = {optimizer_cls.__name__: optimizer_cls(n_params=model.n_params)}
    results = compare_calibrations(model, initial_state, data, x0, optimizers, n_iter=800)

    result = results[optimizer_cls.__name__]
    tolerance = 0.6 if optimizer_cls is SGD else 0.3
    relative_error = np.abs(result.fitted_params - true_params) / true_params
    assert np.all(relative_error < tolerance)


def test_calibration_generic_on_sirdvital():
    model = SIRDVital(population=1000.0)
    true_params = SIRDVital.params(
        beta=0.4, gamma=0.1, mu=0.01, birth_rate=0.001, natural_death_rate=0.001
    )
    initial_state = {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}
    data = generate_synthetic_data(
        model, initial_state, true_params, T_GRID, noise_level=0.03, seed=11
    )

    x0 = true_params * 1.2
    result = calibrate_with_scipy(model, initial_state, data, x0)
    relative_error = np.abs(result.fitted_params - true_params) / true_params
    assert np.all(relative_error < 0.3)


def test_compare_calibrations_all_keys_present_and_loss_decreases():
    model = SIRD(population=1000.0)
    true_params = SIRD.params(beta=0.4, gamma=0.1, mu=0.01)
    initial_state = {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}
    data = generate_synthetic_data(
        model, initial_state, true_params, T_GRID, noise_level=0.03, seed=5
    )

    x0 = true_params * 1.3
    optimizers = {"sgd": SGD(n_params=model.n_params), "adam": Adam(n_params=model.n_params)}
    results = compare_calibrations(model, initial_state, data, x0, optimizers, n_iter=300)

    assert set(results) == {"sgd", "adam", "scipy_L-BFGS-B"}
    for result in results.values():
        if result.loss_history is not None:
            assert result.loss_history[-1] <= result.loss_history[0]
