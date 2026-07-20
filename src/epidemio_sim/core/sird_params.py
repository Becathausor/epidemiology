"""Paramètres du modèle SIRD."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from epidemio_sim.core.types import ParamVector


@dataclass(frozen=True, slots=True)
class SIRDParams:
    """Paramètres du modèle SIRD.

    Parameters
    ----------
    beta : float
        Taux de transmission.
    gamma : float
        Taux de guérison.
    mu : float
        Taux de mortalité liée à la maladie.
    """

    beta: float
    gamma: float
    mu: float

    def to_array(self) -> ParamVector:
        """Convertit les paramètres en tableau numpy ordonné."""
        return np.array(dataclasses.astuple(self), dtype=np.float64)

    @classmethod
    def from_array(cls, arr: ParamVector) -> "SIRDParams":
        """Construit les paramètres à partir d'un tableau numpy ordonné."""
        return cls(*(float(x) for x in arr))

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Noms des paramètres, dans l'ordre du tableau numpy."""
        return tuple(f.name for f in dataclasses.fields(cls))
