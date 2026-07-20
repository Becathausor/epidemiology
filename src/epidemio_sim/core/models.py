"""Modèles épidémiologiques : classe abstraite ``EpidemicModel`` et modèles concrets.

Fournit ``SIRD`` (modèle Susceptible-Infecté-Rétabli-Décédé à population
constante) et ``SIRDVital`` (extension avec natalité/mortalité naturelle),
au-dessus d'une façade commune qui cache l'appel au solveur ODE.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from epidemio_sim.solvers.ode_solver import solve_ode

ParamVector = npt.NDArray[np.float64]
StateVector = npt.NDArray[np.float64]


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
