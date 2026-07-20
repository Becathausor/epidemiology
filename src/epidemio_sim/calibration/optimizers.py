"""Optimiseurs de descente de gradient, réimplémentés from scratch.

Interface commune : classe abstraite :class:`Optimizer` avec méthode
``step(params, grad) -> params``. Ce module est indépendant de toute
logique épidémiologique : il est testable sur n'importe quelle fonction de
perte convexe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np
import numpy.typing as npt

ParamVector = npt.NDArray[np.float64]
LossFunc = Callable[[ParamVector], float]
GradFunc = Callable[[ParamVector], ParamVector]


class Optimizer(ABC):
    """Interface commune à tous les optimiseurs de descente de gradient.

    Parameters
    ----------
    n_params : int
        Dimension du vecteur de paramètres optimisé.
    learning_rate : float
        Taux d'apprentissage (strictement positif).
    """

    def __init__(self, n_params: int, learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate doit être strictement positif")
        self.n_params = n_params
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, params: ParamVector, grad: ParamVector) -> ParamVector:
        """Calcule les paramètres mis à jour à partir du gradient courant.

        Effet de bord : met à jour l'état interne (buffers de momentum,
        second moment, compteur d'itérations...) stocké sur l'instance. Une
        instance représente une trajectoire d'optimisation.
        """

    @abstractmethod
    def reset(self) -> None:
        """Remet l'état interne à zéro, pour relancer une optimisation."""


class SGD(Optimizer):
    """Descente de gradient stochastique simple, sans état interne."""

    def __init__(self, n_params: int, learning_rate: float = 0.01) -> None:
        super().__init__(n_params, learning_rate)

    def step(self, params: ParamVector, grad: ParamVector) -> ParamVector:
        return params - self.learning_rate * grad

    def reset(self) -> None:
        pass


class Momentum(Optimizer):
    """Descente de gradient avec momentum (moyenne mobile de la vitesse).

    Parameters
    ----------
    momentum : float, default 0.9
        Coefficient de conservation de la vitesse précédente.
    """

    def __init__(self, n_params: int, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__(n_params, learning_rate)
        self.momentum = momentum
        self._velocity = np.zeros(n_params, dtype=np.float64)

    def step(self, params: ParamVector, grad: ParamVector) -> ParamVector:
        self._velocity = self.momentum * self._velocity - self.learning_rate * grad
        return params + self._velocity

    def reset(self) -> None:
        self._velocity[:] = 0.0


class RMSprop(Optimizer):
    """RMSprop : normalise le gradient par une moyenne mobile de son carré.

    Parameters
    ----------
    decay : float, default 0.9
        Taux de décroissance de la moyenne mobile du carré du gradient.
    eps : float, default 1e-8
        Terme de stabilité numérique évitant une division par zéro.
    """

    def __init__(
        self,
        n_params: int,
        learning_rate: float = 0.01,
        decay: float = 0.9,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(n_params, learning_rate)
        self.decay = decay
        self.eps = eps
        self._sq_grad_avg = np.zeros(n_params, dtype=np.float64)

    def step(self, params: ParamVector, grad: ParamVector) -> ParamVector:
        self._sq_grad_avg = self.decay * self._sq_grad_avg + (1 - self.decay) * grad**2
        return params - self.learning_rate * grad / (np.sqrt(self._sq_grad_avg) + self.eps)

    def reset(self) -> None:
        self._sq_grad_avg[:] = 0.0


class Adam(Optimizer):
    """Adam : moments d'ordre 1 et 2 du gradient avec correction de biais.

    Parameters
    ----------
    beta1 : float, default 0.9
        Taux de décroissance du moment d'ordre 1.
    beta2 : float, default 0.999
        Taux de décroissance du moment d'ordre 2.
    eps : float, default 1e-8
        Terme de stabilité numérique évitant une division par zéro.
    """

    def __init__(
        self,
        n_params: int,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(n_params, learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._m = np.zeros(n_params, dtype=np.float64)
        self._v = np.zeros(n_params, dtype=np.float64)
        self._t = 0

    def step(self, params: ParamVector, grad: ParamVector) -> ParamVector:
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad**2
        m_hat = self._m / (1 - self.beta1**self._t)
        v_hat = self._v / (1 - self.beta2**self._t)
        return params - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        self._m[:] = 0.0
        self._v[:] = 0.0
        self._t = 0


def numerical_gradient(loss_fn: LossFunc, x: ParamVector, eps: float = 1e-6) -> ParamVector:
    """Gradient par différences finies centrées.

    Notes
    -----
    Boucle Python sur les ``n_params`` dimensions : exception assumée à la
    règle générale de vectorisation, car ``n_params`` est petit (quelques
    unités dans ce projet) et chaque évaluation de ``loss_fn`` implique déjà
    une intégration ODE complète, non batchable en un seul appel numpy.

    Parameters
    ----------
    loss_fn : callable
        Fonction de perte scalaire.
    x : ndarray of shape (n_params,)
        Point où évaluer le gradient.
    eps : float, default 1e-6
        Pas de différenciation.

    Returns
    -------
    ndarray of shape (n_params,)
        Gradient estimé.
    """
    grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        dx = np.zeros_like(x)
        dx[i] = eps
        grad[i] = (loss_fn(x + dx) - loss_fn(x - dx)) / (2 * eps)
    return grad


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    """Résultat d'une optimisation.

    Parameters
    ----------
    x : ndarray of shape (n_params,)
        Paramètres finaux.
    loss_history : ndarray of shape (n_iter_effectif,)
        Valeur de la perte à chaque itération effectuée.
    n_iter : int
        Nombre d'itérations effectuées.
    converged : bool
        Indique si l'arrêt anticipé a été déclenché avant ``n_iter`` max.
    """

    x: ParamVector
    loss_history: npt.NDArray[np.float64]
    n_iter: int
    converged: bool


def run_optimizer(
    optimizer: Optimizer,
    loss_fn: LossFunc,
    x0: ParamVector,
    n_iter: int = 500,
    grad_fn: GradFunc | None = None,
    tol: float = 1e-10,
) -> OptimizationResult:
    """Boucle d'optimisation générique autour d'un :class:`Optimizer`.

    Parameters
    ----------
    optimizer : Optimizer
        Instance d'optimiseur (SGD, Momentum, RMSprop ou Adam).
    loss_fn : callable
        Fonction de perte scalaire à minimiser.
    x0 : ndarray of shape (n_params,)
        Point de départ.
    n_iter : int, default 500
        Nombre maximal d'itérations.
    grad_fn : callable, optional
        Fonction de gradient. Si ``None``, utilise
        ``numerical_gradient(loss_fn, ·)``.
    tol : float, default 1e-10
        Arrêt anticipé si ``|loss_k - loss_{k-1}| < tol``.

    Returns
    -------
    OptimizationResult
        Trajectoire d'optimisation.
    """
    if grad_fn is None:
        grad_fn = lambda x: numerical_gradient(loss_fn, x)  # noqa: E731

    x = np.array(x0, dtype=np.float64)
    loss_history = [loss_fn(x)]
    converged = False
    n_done = 0
    for _ in range(n_iter):
        grad = grad_fn(x)
        x = optimizer.step(x, grad)
        loss_history.append(loss_fn(x))
        n_done += 1
        if abs(loss_history[-1] - loss_history[-2]) < tol:
            converged = True
            break

    return OptimizationResult(
        x=x,
        loss_history=np.array(loss_history, dtype=np.float64),
        n_iter=n_done,
        converged=converged,
    )
