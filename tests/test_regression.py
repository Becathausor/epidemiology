"""Tests de régression numérique : conservation, comportements asymptotiques connus."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from epidemio_sim.core import SIRD, SIRDVital

T_GRID = np.linspace(0.0, 100.0, 200)


@pytest.mark.parametrize(
    "beta,gamma,mu",
    [(0.3, 0.1, 0.02), (0.5, 0.2, 0.0), (0.8, 0.05, 0.05)],
)
def test_sird_conservation_over_time(beta, gamma, mu):
    model = SIRD(population=1000.0)
    params = SIRD.params(beta, gamma, mu)
    result = model.simulate(
        {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}, params, t_span=(0.0, 100.0), t_eval=T_GRID
    )
    np.testing.assert_allclose(result.total_population(), 1000.0, atol=1e-6)


def test_sird_early_time_exponential_growth():
    beta, gamma, mu = 0.5, 0.1, 0.01
    model = SIRD(population=100_000.0)
    I0 = 10.0
    params = SIRD.params(beta, gamma, mu)
    t_eval = np.linspace(0.0, 2.0, 20)
    result = model.simulate(
        {"S": 100_000.0 - I0, "I": I0, "R": 0.0, "D": 0.0},
        params,
        t_span=(0.0, 2.0),
        t_eval=t_eval,
    )

    # Régime approximatif I0 << N, valable en tout début de trajectoire :
    # ce n'est PAS une solution analytique exacte du SIR complet.
    expected = I0 * np.exp((beta - gamma - mu) * t_eval)
    np.testing.assert_allclose(result.get("I"), expected, rtol=0.02)


def test_sird_final_size_relation():
    # Relation de taille finale de Kermack-McKendrick (mu=0, cas SIR pur) :
    # 1 - R_inf/N = exp(-R0 * R_inf/N)
    beta, gamma = 0.4, 0.1
    N = 1000.0
    R0 = beta / gamma
    model = SIRD(population=N)
    params = SIRD.params(beta, gamma, mu=0.0)
    t_eval = np.linspace(0.0, 300.0, 500)
    result = model.simulate(
        {"S": N - 1.0, "I": 1.0, "R": 0.0, "D": 0.0}, params, t_span=(0.0, 300.0), t_eval=t_eval
    )
    r_inf_simulated = result.get("R")[-1] / N

    def final_size_eq(r_inf_frac):
        return 1 - r_inf_frac - np.exp(-R0 * r_inf_frac)

    r_inf_theoretical = brentq(final_size_eq, 1e-6, 1 - 1e-9)
    assert r_inf_simulated == pytest.approx(r_inf_theoretical, abs=1e-3)


def test_sirdvital_reduces_to_sird_when_vital_rates_zero():
    # A t=0 (D=0), la population vivante S+I+R == population totale, donc
    # SIRDVital._effective_population (dynamique) coïncide avec
    # SIRD._effective_population (constante). L'égalité n'est PAS garantie
    # aux instants ultérieurs où D > 0 : elle est vérifiée ici uniquement à
    # l'état initial, ce qui suffit à prouver la réutilisation de l'algèbre.
    y0 = np.array([990.0, 10.0, 0.0, 0.0])
    beta, gamma, mu = 0.3, 0.1, 0.02

    sird = SIRD(population=1000.0)
    sird_dy = sird.rhs(0.0, y0, SIRD.params(beta, gamma, mu))

    sirdvital = SIRDVital(population=1000.0)
    sirdvital_dy = sirdvital.rhs(
        0.0, y0, SIRDVital.params(beta, gamma, mu, birth_rate=0.0, natural_death_rate=0.0)
    )

    np.testing.assert_allclose(sirdvital_dy, sird_dy)


def test_sirdvital_death_compartment_monotonic():
    model = SIRDVital(population=1000.0)
    params = SIRDVital.params(
        beta=0.4, gamma=0.1, mu=0.02, birth_rate=0.001, natural_death_rate=0.001
    )
    result = model.simulate(
        {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}, params, t_span=(0.0, 100.0), t_eval=T_GRID
    )
    assert np.all(np.diff(result.get("D")) >= -1e-9)


def test_sirdvital_mass_balance():
    model = SIRDVital(population=1000.0)
    beta, gamma, mu, birth_rate, natural_death_rate = 0.4, 0.1, 0.02, 0.01, 0.005
    params = SIRDVital.params(beta, gamma, mu, birth_rate, natural_death_rate)
    y0 = np.array([990.0, 10.0, 0.0, 0.0])

    dy = model.rhs(0.0, y0, params)
    S, I, R, _D = y0
    n_alive = S + I + R
    expected_dN = birth_rate * n_alive - natural_death_rate * (S + I + R)
    assert np.sum(dy) == pytest.approx(expected_dN, abs=1e-9)

    t_eval = np.linspace(0.0, 50.0, 500)
    result = model.simulate(
        {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}, params, t_span=(0.0, 50.0), t_eval=t_eval
    )
    n_alive_traj = result.get("S") + result.get("I") + result.get("R")
    births = birth_rate * n_alive_traj
    natural_deaths = natural_death_rate * n_alive_traj
    expected_growth = np.trapezoid(births - natural_deaths, t_eval)
    actual_growth = result.total_population()[-1] - result.total_population()[0]
    assert actual_growth == pytest.approx(expected_growth, rel=1e-2)


def test_golden_trajectory_regression():
    model = SIRD(population=1000.0)
    params = SIRD.params(beta=0.35, gamma=0.12, mu=0.015)
    t_eval = np.array([0.0, 10.0, 30.0, 60.0])
    result = model.simulate(
        {"S": 995.0, "I": 5.0, "R": 0.0, "D": 0.0}, params, t_span=(0.0, 60.0), t_eval=t_eval
    )

    # Trajectoire de référence figée : garde-fou contre une régression
    # silencieuse suite à un futur refactor du modèle ou du solveur.
    expected_I = np.array([5.0, 14.72377062, 249.87503072, 132.02067393])
    np.testing.assert_allclose(result.get("I"), expected_I, rtol=1e-5)
