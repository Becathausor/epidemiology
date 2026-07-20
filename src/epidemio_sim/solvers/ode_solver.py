"""Wrapper générique autour de ``scipy.integrate.solve_ivp``.

Ce module ne connaît aucun modèle épidémiologique : il manipule uniquement
des fonctions ``rhs`` respectant le contrat ``(t, y, params) -> dy/dt``,
ce qui le rend réutilisable pour n'importe quelle équation différentielle
ordinaire.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp

RhsFunc = Callable[
    [float, npt.NDArray[np.float64], npt.NDArray[np.float64]], npt.NDArray[np.float64]
]


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    """Résultat brut d'une intégration numérique.

    Parameters
    ----------
    t : ndarray of shape (n_t,)
        Instants d'évaluation.
    y : ndarray of shape (n_states, n_t)
        Trajectoire intégrée, une ligne par variable d'état.
    success : bool
        Indique si l'intégration s'est terminée avec succès.
    message : str
        Message informatif renvoyé par le solveur.
    nfev : int
        Nombre d'évaluations de ``rhs`` effectuées.
    """

    t: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    success: bool
    message: str
    nfev: int


def solve_ode(
    rhs: RhsFunc,
    y0: npt.NDArray[np.float64],
    t_span: tuple[float, float],
    params: npt.NDArray[np.float64],
    t_eval: npt.NDArray[np.float64] | None = None,
    method: str = "RK45",
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step: float = np.inf,
) -> IntegrationResult:
    """Intègre ``dy/dt = rhs(t, y, params)`` sur ``t_span``.

    Wrapper mince autour de :func:`scipy.integrate.solve_ivp`, utilisé en
    production par tous les modèles épidémiologiques de ce package.

    Parameters
    ----------
    rhs : callable
        Fonction ``(t, y, params) -> dy/dt``, vectorisée sur les variables
        d'état.
    y0 : ndarray of shape (n_states,)
        État initial.
    t_span : tuple of float
        Bornes ``(t0, tf)`` de l'intégration.
    params : ndarray of shape (n_params,)
        Paramètres passés tels quels à ``rhs`` via ``args``.
    t_eval : ndarray, optional
        Instants où la solution doit être renvoyée. Si ``None``, le solveur
        choisit lui-même les instants.
    method : str, default "RK45"
        Méthode d'intégration de ``solve_ivp``.
    rtol, atol : float
        Tolérances relative et absolue.
    max_step : float, default inf
        Pas maximal autorisé.

    Returns
    -------
    IntegrationResult
        Résultat de l'intégration.

    Raises
    ------
    RuntimeError
        Si l'intégration échoue (``sol.success`` est ``False``).
    """
    sol = solve_ivp(
        fun=rhs,
        t_span=t_span,
        y0=y0,
        args=(params,),
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f"Échec de l'intégration ODE (méthode={method}) : {sol.message}")
    return IntegrationResult(
        t=sol.t, y=sol.y, success=sol.success, message=sol.message, nfev=sol.nfev
    )


def euler_explicit(
    rhs: RhsFunc,
    y0: npt.NDArray[np.float64],
    t_span: tuple[float, float],
    params: npt.NDArray[np.float64],
    n_steps: int,
) -> IntegrationResult:
    """Intégrateur d'Euler explicite à pas fixe — option pédagogique uniquement.

    Ne jamais utiliser en production : la précision est bien inférieure à
    celle de :func:`solve_ode` (RK45) à budget de calcul comparable. Fournie
    uniquement pour comparer, à des fins d'apprentissage, la stabilité et la
    précision d'un schéma naïf face à un solveur adaptatif.

    La boucle sur les pas de temps est une récurrence séquentielle
    intrinsèque (chaque pas dépend du précédent) : c'est une exception
    assumée à la règle générale « pas de boucle Python sur des tableaux »,
    isolée dans cette seule fonction. Chaque évaluation individuelle de
    ``rhs`` reste vectorisée sur les compartiments.

    Parameters
    ----------
    rhs : callable
        Fonction ``(t, y, params) -> dy/dt``.
    y0 : ndarray of shape (n_states,)
        État initial.
    t_span : tuple of float
        Bornes ``(t0, tf)`` de l'intégration.
    params : ndarray of shape (n_params,)
        Paramètres passés à ``rhs``.
    n_steps : int
        Nombre de pas de temps fixes.

    Returns
    -------
    IntegrationResult
        Résultat de l'intégration, avec ``message="euler-explicit"``.
    """
    t0, tf = t_span
    dt = (tf - t0) / n_steps
    t = t0 + dt * np.arange(n_steps + 1)
    y = np.empty((y0.shape[0], n_steps + 1))
    y[:, 0] = y0
    for k in range(n_steps):
        y[:, k + 1] = y[:, k] + dt * rhs(t[k], y[:, k], params)
    return IntegrationResult(t=t, y=y, success=True, message="euler-explicit", nfev=n_steps)
