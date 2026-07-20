"""Résultat d'une simulation épidémiologique."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from epidemio_sim.core.types import ParamVector


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Résultat d'une simulation, enveloppe autour d'``IntegrationResult``.

    Parameters
    ----------
    t : ndarray of shape (n_t,)
        Instants d'évaluation.
    y : ndarray of shape (n_compartments, n_t)
        Trajectoire simulée, une ligne par compartiment.
    compartments : tuple of str
        Noms des compartiments, dans l'ordre des lignes de ``y``.
    params : ndarray of shape (n_params,)
        Paramètres utilisés pour la simulation.
    param_names : tuple of str
        Noms des paramètres, dans l'ordre de ``params``.
    model_name : str
        Nom de la classe de modèle utilisée.
    """

    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    compartments: tuple[str, ...]
    params: ParamVector
    param_names: tuple[str, ...]
    model_name: str

    def get(self, compartment: str) -> npt.NDArray[np.float64]:
        """Retourne la trajectoire d'un compartiment donné."""
        return self.y[self.compartments.index(compartment)]

    def total_population(self) -> npt.NDArray[np.float64]:
        """Somme de tous les compartiments à chaque instant."""
        return self.y.sum(axis=0)

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit le résultat en ``DataFrame`` pandas (colonnes t + compartiments)."""
        df = pd.DataFrame(self.y.T, columns=list(self.compartments))
        df.insert(0, "t", self.t)
        return df
