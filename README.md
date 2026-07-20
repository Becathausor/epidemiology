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

## Interface web (API FastAPI + visualisation)

Une API FastAPI expose le modèle SIRD/SIRDVital, et une page statique
(`src/webclient/static/`) permet de lancer des simulations et de visualiser
les courbes S/I/R/D depuis le navigateur, sans écrire de code Python.

### Installation

```bash
pip install -e ".[dev,web]"
```

### Lancement

```bash
uvicorn webclient.server:app --reload
```

Puis ouvrir [http://127.0.0.1:8000](http://127.0.0.1:8000) dans un navigateur :
le formulaire est pré-rempli avec les valeurs par défaut du README, permet de
choisir entre SIRD et SIRDVital, d'ajuster le temps de simulation, la
précision (nombre de points), et d'ajouter du bruit gaussien (niveau + seed)
pour comparer trajectoire simulée et observations bruitées sur le graphique.

### API

- `GET /api/health` — healthcheck.

  ```bash
  curl http://127.0.0.1:8000/api/health
  # {"status": "ok"}
  ```

- `POST /api/simulate` — simulation (tous les champs sont optionnels, les
  valeurs par défaut reprennent l'exemple de la section « Utilisation
  rapide » ci-dessus) :

  ```bash
  curl -X POST http://127.0.0.1:8000/api/simulate \
    -H "Content-Type: application/json" \
    -d '{
      "model": "SIRD",
      "population": 1000.0,
      "s0": 990.0, "i0": 10.0, "r0": 0.0, "d0": 0.0,
      "beta": 0.3, "gamma": 0.1, "mu": 0.02,
      "t_max": 100.0, "n_points": 200,
      "noise": {"enabled": true, "level": 0.05, "seed": 42}
    }'
  ```

  Réponse : `t`, `compartments` (trajectoire simulée, une liste par
  compartiment `S/I/R/D`), et `noisy_compartments` (observations bruitées,
  `null` si `noise.enabled` est `false`). Pour `model: "SIRDVital"`, les
  champs `birth_rate` et `natural_death_rate` sont pris en compte.

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

src/webclient/
├── server.py                  # API FastAPI (healthcheck + simulation)
└── static/                    # page web (formulaire + graphique canvas)
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

## Documentation

Documentation Sphinx (référence API générée depuis les docstrings NumPy +
tutoriel) dans `docs/` :

```bash
sphinx-build -b html docs docs/_build/html
```

Puis ouvrir `docs/_build/html/index.html`.

## Description des modèles

Le modèle SIRD modélise l'épidémie en identifiant 4 populations:
- S: Sensibles
- I: Infectés
- R: Rémission
- D: Décédé

On ajoute la population $N = S + I + R + D$ du nombre d'individus.

Pour le modèle on a 5 paramètres:
- $\beta$ le coefficient d'infection
- $\gamma$ le coefficient de rémission
- $\mu$ le coefficient de mortalité
- $\mu_{nat}$ le coefficient de mort naturelle
- $\alpha$ le coefficient de natalité naturelle

Le système suit ainsi ces équations différentielles:

$$\frac{dS}{dt} = -\beta \frac{I}{N}S - \mu_{nat}S + \alpha N$$
$$\frac{dI}{dt} = \beta \frac{I}{N}S -\gamma I - \mu I - \mu_{nat}I$$
$$\frac{dR}{dt} = \gamma I - \mu_{nat} R$$
$$\frac{dD}{dt} = \mu I + \mu_{nat} N$$

Dans la simulation disponible via l'API FastAPI, on a le cas SIRDVital qui prend en compte la mortalité et la natalité naturelle et la considérant comme indépendante de la maladie. Dans le cas SIRD classique, on a les paramètres naturels fixés à 0. 