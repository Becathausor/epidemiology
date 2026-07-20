"""Modèle SIRD (Susceptible-Infecté-Rétabli-Décédé) à population constante."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from epidemio_sim.core.epidemic_model import EpidemicModel
from epidemio_sim.core.sird_params import SIRDParams
from epidemio_sim.core.types import ParamVector, StateVector


class SIRD(EpidemicModel):
    """Modèle SIRD (Susceptible-Infecté-Rétabli-Décédé) à population constante.

    Équations
    ---------
    dS/dt = -beta*S*I/N
    dI/dt = beta*S*I/N - gamma*I - mu*I
    dR/dt = gamma*I
    dD/dt = mu*I
    """

    compartments: ClassVar[tuple[str, ...]] = ("S", "I", "R", "D")
    param_names: ClassVar[tuple[str, ...]] = SIRDParams.names()

    @staticmethod
    def params(beta: float, gamma: float, mu: float) -> ParamVector:
        """Construit un vecteur de paramètres SIRD ordonné, de façon ergonomique."""
        return SIRDParams(beta, gamma, mu).to_array()

    def rhs(self, t: float, y: StateVector, params: ParamVector) -> StateVector:
        S, I, R, D = y
        beta, gamma, mu = params
        n_eff = self._effective_population(y)
        new_infections = beta * S * I / n_eff
        new_recoveries = gamma * I
        new_deaths = mu * I
        return np.array(
            [
                -new_infections,
                new_infections - new_recoveries - new_deaths,
                new_recoveries,
                new_deaths,
            ]
        )
