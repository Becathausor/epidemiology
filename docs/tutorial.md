# Tutoriel

## Installation

```bash
pip install -e ".[dev]"
```

## Simuler un modèle SIRD

La façade publique de chaque modèle est la méthode
{py:meth}`~epidemio_sim.core.epidemic_model.EpidemicModel.simulate` : elle
cache l'appel au solveur ODE générique
({py:func}`~epidemio_sim.solvers.ode_solver.solve_ode`, basé sur
`scipy.integrate.solve_ivp`).

```python
import numpy as np

from epidemio_sim import SIRD

model = SIRD(population=1_000.0)
params = model.params(beta=0.4, gamma=0.1, mu=0.01)

result = model.simulate(
    initial_state={"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0},
    params=params,
    t_span=(0.0, 100.0),
    t_eval=np.linspace(0.0, 100.0, 200),
)

infectes = result.get("I")
df = result.to_dataframe()
```

`result` est un {py:class}`~epidemio_sim.core.simulation_result.SimulationResult` :
une trajectoire par compartiment, convertible en `DataFrame` pandas via
{py:meth}`~epidemio_sim.core.simulation_result.SimulationResult.to_dataframe`.

## Calibrer les paramètres sur des données

Le sous-package {doc}`calibration <api/calibration>` réimplémente plusieurs
optimiseurs (`SGD`, `Momentum`, `Adam`, `RMSprop`) derrière une interface
commune {py:class}`~epidemio_sim.calibration.optimizers.Optimizer`, et permet
de comparer leur convergence à celle de `scipy.optimize.minimize` via
{py:func}`~epidemio_sim.calibration.fit.compare_calibrations`.

## Pour aller plus loin

Voir la référence API complète : {doc}`api/core`, {doc}`api/solvers`,
{doc}`api/calibration`, {doc}`api/io`, {doc}`api/viz`.
