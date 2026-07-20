"""Classe abstraite de base pour les modèles épidémiologiques compartimentaux."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Mapping

import numpy as np
import numpy.typing as npt

from epidemio_sim.core.simulation_result import SimulationResult
from epidemio_sim.core.types import ParamVector, StateVector
from epidemio_sim.solvers.ode_solver import solve_ode


class EpidemicModel(ABC):
    """Classe abstraite de base pour tout modèle épidémiologique compartimental.

    Les sous-classes concrètes doivent définir les attributs de classe
    ``compartments`` et ``param_names``, et implémenter :meth:`rhs`. La
    méthode :meth:`simulate` est la façade publique : elle cache l'appel au
    solveur ODE générique (:func:`epidemio_sim.solvers.ode_solver.solve_ode`).

    Parameters
    ----------
    population : float
        Taille totale de la population (strictement positive).
    """

    compartments: ClassVar[tuple[str, ...]]
    param_names: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "compartments", None) or not getattr(cls, "param_names", None):
            raise TypeError(
                f"{cls.__name__} doit définir des attributs de classe "
                "'compartments' et 'param_names' non vides."
            )

    def __init__(self, population: float) -> None:
        if population <= 0:
            raise ValueError("population doit être strictement positive")
        self.population = float(population)

    @property
    def n_compartments(self) -> int:
        """Nombre de compartiments du modèle."""
        return len(self.compartments)

    @property
    def n_params(self) -> int:
        """Nombre de paramètres du modèle."""
        return len(self.param_names)

    @abstractmethod
    def rhs(self, t: float, y: StateVector, params: ParamVector) -> StateVector:
        """Second membre de l'EDO : ``dy/dt = f(t, y, params)``.

        Signature imposée par ``scipy.integrate.solve_ivp`` (``fun(t, y,
        *args)``). L'implémentation doit être vectorisée (algèbre numpy
        directe sur les compartiments), sans boucle Python.

        Parameters
        ----------
        t : float
            Instant courant (peut être inutilisé si le système est autonome).
        y : ndarray of shape (n_compartments,)
            État courant.
        params : ndarray of shape (n_params,)
            Paramètres du modèle.

        Returns
        -------
        ndarray of shape (n_compartments,)
            Dérivée de l'état par rapport au temps.
        """

    def _effective_population(self, y: StateVector) -> float | npt.NDArray[np.float64]:
        """Population utilisée pour normaliser la force d'infection ``beta*S*I/N``.

        Par défaut, retourne la population totale constante (cas ``SIRD``).
        Hook de type *template method*, surchargé par ``SIRDVital`` pour
        utiliser la population vivante dynamique ``S+I+R``.
        """
        return self.population

    def initial_state_from_dict(self, state: Mapping[str, float]) -> StateVector:
        """Valide et ordonne un état initial nommé selon ``self.compartments``.

        Parameters
        ----------
        state : mapping
            État initial, une valeur par nom de compartiment.

        Returns
        -------
        ndarray of shape (n_compartments,)
            État initial ordonné selon ``self.compartments``.

        Raises
        ------
        ValueError
            Si des clés sont manquantes ou en trop, si une valeur est
            négative, ou si la somme des valeurs ne correspond pas à
            ``self.population`` (tolérance relative ``1e-6``).
        """
        expected = set(self.compartments)
        got = set(state.keys())
        if got != expected:
            raise ValueError(
                f"Clés d'état initial invalides : attendu {sorted(expected)}, reçu {sorted(got)}"
            )
        y0 = np.array([state[c] for c in self.compartments], dtype=np.float64)
        if np.any(y0 < 0):
            raise ValueError("Les valeurs de l'état initial doivent être positives ou nulles")
        total = float(y0.sum())
        if not np.isclose(total, self.population, rtol=1e-6):
            raise ValueError(
                f"La somme de l'état initial ({total}) ne correspond pas à la "
                f"population ({self.population})"
            )
        return y0

    def simulate(
        self,
        initial_state: Mapping[str, float],
        params: ParamVector,
        t_span: tuple[float, float],
        t_eval: npt.NDArray[np.float64] | None = None,
        method: str = "RK45",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        **solver_kwargs: Any,
    ) -> SimulationResult:
        """Simule le modèle sur ``t_span``.

        Façade publique : seule méthode que l'utilisateur final a besoin
        d'appeler pour simuler un modèle. Cache l'appel au solveur ODE
        générique.

        Parameters
        ----------
        initial_state : mapping
            État initial nommé (une valeur par compartiment).
        params : ndarray of shape (n_params,)
            Paramètres du modèle, dans l'ordre de ``self.param_names``.
        t_span : tuple of float
            Bornes ``(t0, tf)`` de la simulation.
        t_eval : ndarray, optional
            Instants où la solution doit être renvoyée.
        method : str, default "RK45"
            Méthode d'intégration.
        rtol, atol : float
            Tolérances relative et absolue de l'intégration.
        **solver_kwargs
            Arguments additionnels transmis au solveur.

        Returns
        -------
        SimulationResult
            Trajectoire simulée.
        """
        y0 = self.initial_state_from_dict(initial_state)
        raw = solve_ode(
            self.rhs,
            y0,
            t_span,
            params,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
            **solver_kwargs,
        )
        return SimulationResult(
            t=raw.t,
            y=raw.y,
            compartments=self.compartments,
            params=params,
            param_names=self.param_names,
            model_name=type(self).__name__,
        )
