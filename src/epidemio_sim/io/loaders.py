"""Chargement de données réelles et génération de données synthétiques bruitées."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from epidemio_sim.calibration.fit import CalibrationData
from epidemio_sim.core import EpidemicModel

ParamVector = npt.NDArray[np.float64]


def generate_synthetic_data(
    model: EpidemicModel,
    initial_state: Mapping[str, float],
    true_params: ParamVector,
    t: npt.NDArray[np.float64],
    noise: Literal["gaussian", "poisson"] = "gaussian",
    noise_level: float = 0.05,
    compartments: Sequence[str] | None = None,
    seed: int | None = None,
) -> CalibrationData:
    """Simule une trajectoire puis y ajoute du bruit, pour tester la calibration.

    Parameters
    ----------
    model : EpidemicModel
        Modèle utilisé pour générer la vraie trajectoire.
    initial_state : mapping
        État initial nommé.
    true_params : ndarray of shape (n_params,)
        Vrais paramètres, utilisés pour simuler puis conservés dans le
        résultat pour permettre de vérifier que la calibration les retrouve.
    t : ndarray of shape (n_t,)
        Instants d'observation.
    noise : {"gaussian", "poisson"}, default "gaussian"
        Type de bruit ajouté. "gaussian" ajoute un bruit proportionnel à la
        valeur simulée ; "poisson" est adapté aux compartiments de comptage
        (I, D).
    noise_level : float, default 0.05
        Écart-type relatif du bruit gaussien (ignoré si ``noise="poisson"``).
    compartments : sequence of str, optional
        Compartiments à observer. Par défaut, tous les compartiments du
        modèle.
    seed : int, optional
        Graine du générateur pseudo-aléatoire, pour la reproductibilité.

    Returns
    -------
    CalibrationData
        Données bruitées, avec ``true_params`` renseigné.
    """
    rng = np.random.default_rng(seed)
    sim = model.simulate(initial_state, true_params, t_span=(t[0], t[-1]), t_eval=t)
    compartments = compartments or model.compartments

    observations: dict[str, npt.NDArray[np.float64]] = {}
    for c in compartments:
        true_values = sim.get(c)
        if noise == "gaussian":
            noisy = true_values + rng.normal(0.0, noise_level * true_values)
            observations[c] = np.clip(noisy, 0.0, None)
        else:
            observations[c] = rng.poisson(np.clip(true_values, 0.0, None)).astype(np.float64)

    return CalibrationData(t=t, observations=observations, true_params=true_params)


def load_csv(
    path: str | Path,
    time_column: str = "t",
    compartment_columns: Sequence[str] | None = None,
) -> CalibrationData:
    """Charge des données observées depuis un fichier CSV.

    Parameters
    ----------
    path : str or Path
        Chemin du fichier CSV.
    time_column : str, default "t"
        Nom de la colonne des instants d'observation.
    compartment_columns : sequence of str, optional
        Colonnes à charger comme compartiments observés. Par défaut, toutes
        les colonnes autres que ``time_column``.

    Returns
    -------
    CalibrationData
        Données chargées (``true_params`` vaut ``None``).
    """
    df = pd.read_csv(path)
    columns = compartment_columns or [c for c in df.columns if c != time_column]
    observations = {c: df[c].to_numpy(dtype=np.float64) for c in columns}
    return CalibrationData(
        t=df[time_column].to_numpy(dtype=np.float64), observations=observations
    )
