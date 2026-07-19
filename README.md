# epidemio-sim

Bibliothèque Python de simulation et de calibration de modèles
épidémiologiques compartimentaux (SIRD, et son extension avec dynamique
vitale SIRDVital), avec une suite d'optimiseurs de descente de gradient
réimplémentés from scratch pour la calibration des paramètres.

Projet de démonstration de compétences en ingénierie logicielle appliquée
au calcul scientifique : architecture orientée objet propre, vectorisation
numpy, intégration numérique via `scipy.integrate.solve_ivp`, tests
systématiques (unitaires, intégration, régression numérique), et type
hints stricts (mypy strict).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Utilisation rapide

```python
import numpy as np
from epidemio_sim import SIRD

model = SIRD(population=1000.0)
params = SIRD.params(beta=0.3, gamma=0.1, mu=0.02)

result = model.simulate(
    initial_state={"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0},
    params=params,
    t_span=(0.0, 100.0),
    t_eval=np.linspace(0.0, 100.0, 200),
)

result.to_dataframe().head()
```

### Calibration sur données synthétiques bruitées

```python
from epidemio_sim import SGD, Adam, compare_calibrations, generate_synthetic_data

t = np.linspace(0.0, 60.0, 61)
data = generate_synthetic_data(
    model, {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}, params, t,
    noise="gaussian", noise_level=0.05, seed=42,
)

x0 = params * 1.3  # point de départ perturbé
optimizers = {"sgd": SGD(n_params=model.n_params), "adam": Adam(n_params=model.n_params)}
results = compare_calibrations(
    model, {"S": 990.0, "I": 10.0, "R": 0.0, "D": 0.0}, data, x0, optimizers,
)

for name, res in results.items():
    print(name, res.fitted_params)
```

## Architecture

```
src/epidemio_sim/
├── core/models.py          # EpidemicModel (ABC), SIRD, SIRDVital
├── solvers/ode_solver.py    # wrapper générique autour de solve_ivp
├── calibration/
│   ├── optimizers.py        # SGD, Momentum, RMSprop, Adam (from scratch)
│   └── fit.py                # calibration : nos optimizers vs scipy.optimize
├── io/
│   ├── loaders.py            # données synthétiques bruitées, chargement CSV
│   └── exporters.py          # export CSV / JSON des résultats
└── viz/plots.py               # courbes épidémiques, convergence des optimizers
```

Le solveur ODE (`solvers/ode_solver.py`) est totalement générique : il ne
connaît aucun modèle épidémiologique et ne manipule que des fonctions
`rhs(t, y, params) -> dy/dt`. `EpidemicModel` s'appuie dessus pour exposer
une façade simple (`simulate()`), et `SIRDVital` réutilise l'algèbre de
`SIRD` par héritage plutôt que de la dupliquer.

## Tests et qualité

```bash
pytest                        # tests unitaires, intégration, régression numérique
mypy src/epidemio_sim         # vérification de types stricte
ruff check src tests          # lint
```

## Documentation (à faire)

La documentation Sphinx (docstrings au format NumPy) se construit depuis
`docs/` :

```bash
cd docs && make html
```
