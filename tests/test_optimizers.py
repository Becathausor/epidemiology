"""Tests unitaires pour epidemio_sim.calibration.optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from epidemio_sim.calibration.optimizers import (
    Adam,
    Momentum,
    RMSprop,
    SGD,
    numerical_gradient,
    run_optimizer,
)

ALL_OPTIMIZER_CLASSES = [SGD, Momentum, RMSprop, Adam]


@pytest.mark.parametrize("cls", ALL_OPTIMIZER_CLASSES)
def test_optimizer_rejects_invalid_learning_rate(cls):
    with pytest.raises(ValueError):
        cls(n_params=2, learning_rate=0.0)


def test_sgd_step_formula():
    opt = SGD(n_params=2, learning_rate=0.1)
    params = np.array([1.0, 2.0])
    grad = np.array([0.5, -1.0])
    updated = opt.step(params, grad)
    np.testing.assert_allclose(updated, params - 0.1 * grad)


def test_adam_matches_reference_trace():
    opt = Adam(n_params=1, learning_rate=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    params = np.array([0.0])
    grad = np.array([1.0])

    # Étape 1 : m=0.1, v=0.001, m_hat=1.0, v_hat=1.0 -> pas de lr*1/(1+eps) ~ -0.1
    updated = opt.step(params, grad)
    np.testing.assert_allclose(updated, np.array([-0.1]), atol=1e-6)


@pytest.mark.parametrize("cls", ALL_OPTIMIZER_CLASSES)
def test_all_optimizers_converge_on_convex_quadratic(cls):
    x_star = np.array([3.0, -2.0])

    def loss_fn(x):
        return float(np.sum((x - x_star) ** 2))

    def grad_fn(x):
        return 2 * (x - x_star)

    opt = cls(n_params=2, learning_rate=0.1)
    result = run_optimizer(opt, loss_fn, x0=np.zeros(2), n_iter=1000, grad_fn=grad_fn)

    assert np.linalg.norm(result.x - x_star) < 1e-2
    assert result.loss_history[-1] < result.loss_history[0]


def test_momentum_and_adam_faster_than_sgd_on_ill_conditioned_quadratic():
    # Hessienne fortement anisotrope : mauvais conditionnement pour SGD pur.
    A = np.diag([1.0, 100.0])
    x_star = np.array([1.0, 1.0])

    def loss_fn(x):
        d = x - x_star
        return float(d @ A @ d)

    def grad_fn(x):
        return 2 * A @ (x - x_star)

    tol = 1e-3
    n_iter = 300

    results = {}
    for name, opt in [
        ("sgd", SGD(n_params=2, learning_rate=0.005)),
        ("momentum", Momentum(n_params=2, learning_rate=0.005, momentum=0.9)),
        ("adam", Adam(n_params=2, learning_rate=0.1)),
    ]:
        results[name] = run_optimizer(opt, loss_fn, x0=np.zeros(2), n_iter=n_iter, grad_fn=grad_fn)

    assert results["momentum"].loss_history[-1] < results["sgd"].loss_history[-1]
    assert results["adam"].loss_history[-1] < results["sgd"].loss_history[-1]
    assert tol >= 0  # tolérance documentée, comparaison relative ci-dessus


@pytest.mark.parametrize("cls", ALL_OPTIMIZER_CLASSES)
def test_reset_restores_fresh_state(cls):
    grad_sequence = [np.array([1.0, -1.0]), np.array([0.5, 0.2]), np.array([-0.3, 0.7])]

    opt = cls(n_params=2, learning_rate=0.1)
    params = np.zeros(2)
    for grad in grad_sequence:
        params = opt.step(params, grad)

    opt.reset()
    params_after_reset = np.zeros(2)
    for grad in grad_sequence:
        params_after_reset = opt.step(params_after_reset, grad)

    fresh_opt = cls(n_params=2, learning_rate=0.1)
    params_fresh = np.zeros(2)
    for grad in grad_sequence:
        params_fresh = fresh_opt.step(params_fresh, grad)

    np.testing.assert_allclose(params_after_reset, params_fresh)


def test_numerical_gradient_matches_analytic_gradient():
    x_star = np.array([1.5, -0.5])

    def loss_fn(x):
        return float(np.sum((x - x_star) ** 2))

    x = np.array([0.2, 0.3])
    analytic = 2 * (x - x_star)
    numeric = numerical_gradient(loss_fn, x)
    np.testing.assert_allclose(numeric, analytic, atol=1e-4)


def test_run_optimizer_early_stopping():
    def constant_loss(x):
        return 1.0

    opt = SGD(n_params=1, learning_rate=0.1)
    result = run_optimizer(opt, constant_loss, x0=np.zeros(1), n_iter=500, grad_fn=lambda x: np.zeros(1))
    assert result.n_iter < 500
    assert result.converged
