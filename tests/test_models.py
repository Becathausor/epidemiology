"""Tests unitaires pour les modèles épidémiologiques de epidemio_sim.core."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import numpy as np
import pytest

from epidemio_sim.core import (
    EpidemicModel,
    SIRD,
    SIRDParams,
    SIRDVitalParams,
)


def test_abstract_model_cannot_be_instantiated():
    with pytest.raises(TypeError):
        EpidemicModel(1000)  # type: ignore[abstract]


def test_missing_classvars_raises_at_class_definition():
    with pytest.raises(TypeError):

        class Broken(EpidemicModel):
            def rhs(self, t, y, params):
                return y


def test_sird_rhs_algebraic_conservation():
    model = SIRD(population=1000.0)
    y = np.array([700.0, 200.0, 90.0, 10.0])
    params = SIRD.params(beta=0.3, gamma=0.1, mu=0.02)
    dy = model.rhs(0.0, y, params)
    assert np.sum(dy) == pytest.approx(0.0, abs=1e-10)


def test_sird_rhs_matches_hand_computed_values():
    model = SIRD(population=1000.0)
    S, I, R, D = 700.0, 200.0, 90.0, 10.0
    beta, gamma, mu = 0.3, 0.1, 0.02
    y = np.array([S, I, R, D])
    params = SIRD.params(beta, gamma, mu)

    new_infections = beta * S * I / 1000.0
    expected = np.array(
        [
            -new_infections,
            new_infections - gamma * I - mu * I,
            gamma * I,
            mu * I,
        ]
    )
    np.testing.assert_allclose(model.rhs(0.0, y, params), expected)


def test_params_roundtrip():
    params = SIRDParams(0.3, 0.1, 0.02)
    assert SIRDParams.from_array(params.to_array()) == params


def test_sirdvital_params_inherits_sird_fields():
    names = tuple(f.name for f in dataclasses.fields(SIRDVitalParams))
    assert names[:3] == ("beta", "gamma", "mu")
    assert names[3:] == ("birth_rate", "natural_death_rate")


def test_initial_state_validation_rejects_wrong_keys():
    model = SIRD(population=1000.0)
    with pytest.raises(ValueError):
        model.initial_state_from_dict({"S": 990.0, "I": 10.0})  # manque R, D


def test_initial_state_validation_rejects_negative_values():
    model = SIRD(population=1000.0)
    with pytest.raises(ValueError):
        model.initial_state_from_dict({"S": 1010.0, "I": -10.0, "R": 0.0, "D": 0.0})


def test_initial_state_validation_rejects_sum_mismatch():
    model = SIRD(population=1000.0)
    with pytest.raises(ValueError):
        model.initial_state_from_dict({"S": 500.0, "I": 10.0, "R": 0.0, "D": 0.0})


def test_custom_toy_model_end_to_end():
    """Un modèle-jouet minimal prouve que l'ABC est extensible sans toucher au solveur."""

    class ExpDecay(EpidemicModel):
        compartments: ClassVar[tuple[str, ...]] = ("X",)
        param_names: ClassVar[tuple[str, ...]] = ("k",)

        def rhs(self, t, y, params):
            (k,) = params
            return -k * y

    model = ExpDecay(population=1.0)
    result = model.simulate(
        {"X": 1.0}, np.array([0.5]), t_span=(0.0, 1.0), t_eval=np.array([0.0, 1.0])
    )
    np.testing.assert_allclose(result.get("X"), [1.0, np.exp(-0.5)], rtol=1e-6)
