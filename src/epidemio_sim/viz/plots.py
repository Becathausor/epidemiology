"""Visualisation : courbes épidémiques et trajectoires de convergence des optimiseurs."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from epidemio_sim.calibration.fit import CalibrationData, CalibrationResult
from epidemio_sim.core.models import EpidemicModel, SimulationResult


def plot_epidemic_curves(result: SimulationResult, ax: Axes | None = None) -> Axes:
    """Trace une courbe par compartiment en fonction du temps.

    Parameters
    ----------
    result : SimulationResult
        Résultat d'une simulation.
    ax : matplotlib.axes.Axes, optional
        Axes sur lesquels tracer. Si ``None``, une nouvelle figure est créée.

    Returns
    -------
    matplotlib.axes.Axes
        Les axes utilisés.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots()

    for compartment in result.compartments:
        ax.plot(result.t, result.get(compartment), label=compartment)
    ax.set_xlabel("temps")
    ax.set_ylabel("population")
    ax.set_title(f"Trajectoire {result.model_name}")
    ax.legend()
    return ax


def plot_calibration_fit(
    model: EpidemicModel,
    initial_state: Mapping[str, float],
    data: CalibrationData,
    fitted_params: npt.NDArray[np.float64],
    compartments: Sequence[str] | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Superpose les observations bruitées et la courbe simulée avec les paramètres ajustés.

    Parameters
    ----------
    model : EpidemicModel
        Modèle calibré.
    initial_state : mapping
        État initial nommé utilisé pour la calibration.
    data : CalibrationData
        Données observées (potentiellement bruitées).
    fitted_params : ndarray of shape (n_params,)
        Paramètres ajustés, en espace naturel.
    compartments : sequence of str, optional
        Compartiments à tracer. Par défaut, ceux présents dans ``data.observations``.
    ax : matplotlib.axes.Axes, optional
        Axes sur lesquels tracer. Si ``None``, une nouvelle figure est créée.

    Returns
    -------
    matplotlib.axes.Axes
        Les axes utilisés.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots()

    compartments = compartments or tuple(data.observations)
    sim = model.simulate(
        initial_state, fitted_params, t_span=(data.t[0], data.t[-1]), t_eval=data.t
    )
    for compartment in compartments:
        line = ax.plot(data.t, sim.get(compartment), label=f"{compartment} (ajusté)")[0]
        ax.scatter(
            data.t,
            data.observations[compartment],
            s=10,
            color=line.get_color(),
            alpha=0.5,
            label=f"{compartment} (observé)",
        )
    ax.set_xlabel("temps")
    ax.set_ylabel("population")
    ax.set_title("Ajustement de calibration")
    ax.legend()
    return ax


def plot_convergence(
    results: Mapping[str, CalibrationResult],
    ax: Axes | None = None,
    yscale: str = "log",
) -> Axes:
    """Trace la perte en fonction de l'itération, une courbe par optimiseur.

    Les entrées sans ``loss_history`` exploitable (``None``) sont ignorées
    silencieusement.

    Parameters
    ----------
    results : mapping of str to CalibrationResult
        Résultats de calibration, indexés par nom d'optimiseur.
    ax : matplotlib.axes.Axes, optional
        Axes sur lesquels tracer. Si ``None``, une nouvelle figure est créée.
    yscale : str, default "log"
        Échelle de l'axe des ordonnées.

    Returns
    -------
    matplotlib.axes.Axes
        Les axes utilisés.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _fig, ax = plt.subplots()

    for name, result in results.items():
        if result.loss_history is None:
            continue
        ax.plot(result.loss_history, label=name)
    ax.set_yscale(yscale)
    ax.set_xlabel("itération")
    ax.set_ylabel("perte")
    ax.set_title("Convergence des optimiseurs")
    ax.legend()
    return ax
