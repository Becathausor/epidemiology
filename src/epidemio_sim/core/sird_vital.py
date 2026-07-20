"""Extension du modèle SIRD avec natalité et mortalité naturelle."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import numpy.typing as npt

from epidemio_sim.core.sird import SIRD
from epidemio_sim.core.sird_vital_params import SIRDVitalParams
from epidemio_sim.core.types import ParamVector, StateVector


class SIRDVital(SIRD):
    """Extension du modèle SIRD avec natalité et mortalité naturelle.

    Réutilise l'algèbre infection/guérison/mortalité-maladie de ``SIRD.rhs``
    (via ``super().rhs``) et y ajoute les flux vitaux, plutôt que de
    dupliquer les équations. La population effective utilisée dans la force
    d'infection devient la population vivante dynamique ``S+I+R`` (``D``
    exclu), via la surcharge de :meth:`_effective_population`.

    Équations
    ---------
    dS/dt = -beta*S*I/N + birth_rate*N - natural_death_rate*S
    dI/dt = beta*S*I/N - gamma*I - mu*I - natural_death_rate*I
    dR/dt = gamma*I - natural_death_rate*R
    dD/dt = mu*I

    avec ``N = S+I+R``.
    """

    compartments: ClassVar[tuple[str, ...]] = ("S", "I", "R", "D")
    param_names: ClassVar[tuple[str, ...]] = SIRDVitalParams.names()

    @staticmethod
    def params(  # type: ignore[override]
        beta: float, gamma: float, mu: float, birth_rate: float, natural_death_rate: float
    ) -> ParamVector:
        """Construit un vecteur de paramètres SIRDVital ordonné, de façon ergonomique."""
        return SIRDVitalParams(beta, gamma, mu, birth_rate, natural_death_rate).to_array()

    def _effective_population(self, y: StateVector) -> float | npt.NDArray[np.float64]:
        S, I, R, _D = y
        return S + I + R

    def rhs(self, t: float, y: StateVector, params: ParamVector) -> StateVector:
        beta, gamma, mu, birth_rate, natural_death_rate = params
        sird_params = params[: len(SIRD.param_names)]
        base = super().rhs(t, y, sird_params)
        S, I, R, _D = y
        n_alive = self._effective_population(y)
        births = np.array([birth_rate * n_alive, 0.0, 0.0, 0.0])
        natural_deaths = np.array(
            [natural_death_rate * S, natural_death_rate * I, natural_death_rate * R, 0.0]
        )
        return base + births - natural_deaths
