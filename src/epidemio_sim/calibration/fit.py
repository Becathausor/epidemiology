"""Calibration des paramètres d'un ``EpidemicModel`` sur des données bruitées.

Compare nos optimiseurs from scratch (:mod:`epidemio_sim.calibration.optimizers`)
à ``scipy.optimize.minimize`` pris comme référence.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import scipy.optimize

from epidemio_sim.calibration.optimizers import Optimizer, run_optimizer
from epidemio_sim.core.models import EpidemicModel

ParamVector = npt.NDArray[np.float64]

_PENALTY = 1e6


@dataclass(frozen=True, slots=True)
class CalibrationData:
    """Données observées à calibrer.

    Parameters
    ----------
    t : ndarray of shape (n_t,)
        Instants d'observation.
    observations : dict of str to ndarray
        Trajectoires observées (potentiellement bruitées), par compartiment.
    true_params : ndarray, optional
        Paramètres ayant servi à générer les données, si elles sont
        synthétiques (utile pour évaluer la calibration dans les tests).
    """

    t: npt.NDArray[np.float64]
    observations: dict[str, npt.NDArray[np.float64]]
    true_params: ParamVector | None = None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Résultat d'une calibration.

    Parameters
    ----------
    optimizer_name : str
        Nom de l'optimiseur utilisé (ou ``"scipy_<method>"``).
    fitted_params : ndarray of shape (n_params,)
        Paramètres ajustés, en espace naturel (positifs).
    param_names : tuple of str
        Noms des paramètres, dans l'ordre de ``fitted_params``.
    loss_history : ndarray, optional
        Historique de la perte au cours de l'optimisation (``None`` si non
        disponible, par exemple pour certaines méthodes scipy).
    n_iter : int
        Nombre d'itérations effectuées.
    success : bool
        Indique si l'optimisation a convergé/réussi.
    final_loss : float
        Valeur finale de la perte.
    """

    optimizer_name: str
    fitted_params: ParamVector
    param_names: tuple[str, ...]
    loss_history: npt.NDArray[np.float64] | None
    n_iter: int
    success: bool
    final_loss: float


def residuals(
    theta: ParamVector,
    model: EpidemicModel,
    initial_state: Mapping[str, float],
    data: CalibrationData,
    compartments_to_fit: Sequence[str] | None = None,
) -> npt.NDArray[np.float64]:
    """Résidus (simulé - observé), normalisés et concaténés sur les compartiments.

    Parameters
    ----------
    theta : ndarray of shape (n_params,)
        Paramètres en espace log (``params = exp(theta)``), ce qui garantit
        leur positivité sans que l'optimiseur ait à gérer de contrainte.
    model : EpidemicModel
        Modèle à calibrer.
    initial_state : mapping
        État initial nommé.
    data : CalibrationData
        Données observées.
    compartments_to_fit : sequence of str, optional
        Compartiments à inclure dans le résidu. Par défaut, tous ceux
        présents dans ``data.observations``.

    Returns
    -------
    ndarray
        Vecteur de résidus normalisés. Si l'intégration échoue (paramètres
        aberrants proposés en cours d'optimisation), retourne un vecteur de
        pénalité de norme élevée plutôt que de propager l'exception.
    """
    params_array = np.exp(theta)
    compartments = compartments_to_fit or tuple(data.observations)
    try:
        sim = model.simulate(
            initial_state, params_array, t_span=(data.t[0], data.t[-1]), t_eval=data.t
        )
    except RuntimeError:
        n_total = sum(len(data.observations[c]) for c in compartments)
        return np.full(n_total, _PENALTY)

    parts = [
        (sim.get(c) - data.observations[c]) / (np.std(data.observations[c]) + 1e-8)
        for c in compartments
    ]
    return np.concatenate(parts)


def loss(
    theta: ParamVector,
    model: EpidemicModel,
    initial_state: Mapping[str, float],
    data: CalibrationData,
    compartments_to_fit: Sequence[str] | None = None,
) -> float:
    """Somme des carrés des résidus (à un facteur 1/2 près)."""
    r = residuals(theta, model, initial_state, data, compartments_to_fit)
    return 0.5 * float(np.sum(r**2))


def calibrate_with_optimizer(
    optimizer: Optimizer,
    model: EpidemicModel,
    initial_state: Mapping[str, float],
    data: CalibrationData,
    theta0: ParamVector,
    n_iter: int = 500,
    compartments_to_fit: Sequence[str] | None = None,
) -> CalibrationResult:
    """Calibre ``model`` avec un optimiseur from scratch, en espace log.

    Parameters
    ----------
    optimizer : Optimizer
        Optimiseur à utiliser (SGD, Momentum, RMSprop ou Adam).
    model : EpidemicModel
        Modèle à calibrer.
    initial_state : mapping
        État initial nommé.
    data : CalibrationData
        Données observées.
    theta0 : ndarray of shape (n_params,)
        Point de départ, en espace log.
    n_iter : int, default 500
        Nombre maximal d'itérations.
    compartments_to_fit : sequence of str, optional
        Compartiments à ajuster.

    Returns
    -------
    CalibrationResult
        Résultat de la calibration, avec ``fitted_params`` en espace naturel.
    """
    loss_fn = functools.partial(
        loss,
        model=model,
        initial_state=initial_state,
        data=data,
        compartments_to_fit=compartments_to_fit,
    )
    result = run_optimizer(optimizer, loss_fn, theta0, n_iter=n_iter)
    return CalibrationResult(
        optimizer_name=type(optimizer).__name__,
        fitted_params=np.exp(result.x),
        param_names=model.param_names,
        loss_history=result.loss_history,
        n_iter=result.n_iter,
        success=True,
        final_loss=float(result.loss_history[-1]),
    )


def calibrate_with_scipy(
    model: EpidemicModel,
    initial_state: Mapping[str, float],
    data: CalibrationData,
    x0: ParamVector,
    compartments_to_fit: Sequence[str] | None = None,
    method: str = "L-BFGS-B",
) -> CalibrationResult:
    """Calibre ``model`` avec ``scipy.optimize.minimize``, pris comme référence.

    À la différence de :func:`calibrate_with_optimizer`, travaille
    directement en espace naturel (pas de reparamétrisation log), avec des
    bornes de boîte ``[1e-8, +inf[`` sur chaque paramètre pour garantir leur
    positivité.

    Parameters
    ----------
    model : EpidemicModel
        Modèle à calibrer.
    initial_state : mapping
        État initial nommé.
    data : CalibrationData
        Données observées.
    x0 : ndarray of shape (n_params,)
        Point de départ, en espace naturel.
    compartments_to_fit : sequence of str, optional
        Compartiments à ajuster.
    method : str, default "L-BFGS-B"
        Méthode de ``scipy.optimize.minimize``.

    Returns
    -------
    CalibrationResult
        Résultat de la calibration.
    """
    history: list[float] = []

    def loss_natural(p: ParamVector) -> float:
        return loss(np.log(p), model, initial_state, data, compartments_to_fit)

    def record(xk: ParamVector) -> None:
        history.append(loss_natural(xk))

    res = scipy.optimize.minimize(
        loss_natural,
        x0=x0,
        method=method,
        bounds=[(1e-8, None)] * model.n_params,
        callback=record,
    )
    return CalibrationResult(
        optimizer_name=f"scipy_{method}",
        fitted_params=res.x,
        param_names=model.param_names,
        loss_history=np.array(history, dtype=np.float64) if history else None,
        n_iter=res.nit,
        success=bool(res.success),
        final_loss=float(res.fun),
    )


def compare_calibrations(
    model: EpidemicModel,
    initial_state: Mapping[str, float],
    data: CalibrationData,
    x0: ParamVector,
    optimizers: Mapping[str, Optimizer],
    n_iter: int = 500,
    compartments_to_fit: Sequence[str] | None = None,
) -> dict[str, CalibrationResult]:
    """Compare plusieurs optimiseurs custom à scipy sur la même calibration.

    Façade d'orchestration de la calibration : générique par rapport au
    modèle (``SIRD`` ou ``SIRDVital``) puisqu'elle ne dépend que de
    l'interface abstraite ``EpidemicModel``.

    Parameters
    ----------
    model : EpidemicModel
        Modèle à calibrer.
    initial_state : mapping
        État initial nommé.
    data : CalibrationData
        Données observées.
    x0 : ndarray of shape (n_params,)
        Point de départ, en espace naturel (converti en log pour nos
        optimiseurs).
    optimizers : mapping of str to Optimizer
        Optimiseurs custom à comparer, indexés par nom.
    n_iter : int, default 500
        Nombre maximal d'itérations pour chaque optimiseur custom.
    compartments_to_fit : sequence of str, optional
        Compartiments à ajuster.

    Returns
    -------
    dict of str to CalibrationResult
        Un résultat par optimiseur custom, plus une entrée ``"scipy_<method>"``.
    """
    theta0 = np.log(x0)
    results: dict[str, CalibrationResult] = {
        name: calibrate_with_optimizer(
            opt, model, initial_state, data, theta0, n_iter=n_iter,
            compartments_to_fit=compartments_to_fit,
        )
        for name, opt in optimizers.items()
    }
    scipy_result = calibrate_with_scipy(
        model, initial_state, data, x0, compartments_to_fit=compartments_to_fit
    )
    results[scipy_result.optimizer_name] = scipy_result
    return results
