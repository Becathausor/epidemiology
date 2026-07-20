"""Paramètres du modèle SIRD avec dynamique vitale."""

from __future__ import annotations

from dataclasses import dataclass

from epidemio_sim.core.sird_params import SIRDParams


@dataclass(frozen=True, slots=True)
class SIRDVitalParams(SIRDParams):
    """Paramètres du modèle SIRD avec dynamique vitale.

    Parameters
    ----------
    beta : float
        Taux de transmission.
    gamma : float
        Taux de guérison.
    mu : float
        Taux de mortalité liée à la maladie.
    birth_rate : float
        Taux de natalité (nouvelles naissances par individu vivant et par
        unité de temps).
    natural_death_rate : float
        Taux de mortalité naturelle (hors maladie).
    """

    birth_rate: float
    natural_death_rate: float
