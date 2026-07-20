# Contexte du projet

Bibliothèque Python de simulation et calibration de modèles épidémiologiques
(SIRD et extension avec dynamique vitale), développée comme démonstration
de compétences pour un poste d'ingénieur R&D → industrialisation logicielle.

## Objectifs pédagogiques du projet (à respecter dans toutes les implémentations)
- Code orienté objet propre : classe abstraite `EpidemicModel`, sous-classes concrètes
- API publique simple qui cache la complexité interne (façade)
- Vectorisation numpy partout où c'est pertinent (pas de boucles Python sur des tableaux)
- scipy.integrate.solve_ivp pour la résolution des EDO (pas d'Euler naïf en prod, seulement en option pédagogique)
- Tests systématiques : unitaires, intégration, et régression numérique
  (ex: conservation S+I+R+D=N à chaque pas de temps)
- Docstrings au format NumPy pour compatibilité Sphinx autodoc
- Type hints partout (mypy strict)

## Architecture
- `core/models.py` : définition des modèles épidémiologiques (équations, paramètres)
- `solvers/ode_solver.py` : wrapper générique autour de scipy.integrate
- `calibration/optimizers.py` : réimplémentation from scratch de SGD, Momentum, Adam, RMSprop
  (interface commune : classe abstraite `Optimizer` avec méthode `.step()`)
- `calibration/fit.py` : calibration des paramètres du modèle (β, γ, μ...) sur des données
  bruitées, en comparant nos optimizers vs scipy.optimize.minimize
- `io/` : chargement de données réelles/synthétiques, export résultats
- `viz/` : visualisation (courbes épidémiques, trajectoires de convergence des optimizers)

## Ordre de développement suggéré
1. `core/models.py` : implémenter `EpidemicModel` (ABC) puis `SIRD`
2. `solvers/ode_solver.py` : wrapper solve_ivp, tester sur SIRD
3. `tests/test_models.py` + `tests/test_solver.py` : valider avant d'aller plus loin
4. `calibration/optimizers.py` : implémenter les optimizers un par un avec tests
5. `calibration/fit.py` : brancher les optimizers sur la calibration de SIRD
6. `viz/plots.py` : visualisation
7. `core/models.py` : ajouter `SIRDVital` (extension avec natalité/mortalité) en réutilisant l'architecture

## Conventions de code
- Commits atomiques et descriptifs (pas de "wip", "fix", "update")
- Chaque fonctionnalité = une branche + tests avant merge
- Pas de code mort, pas de print() de debug laissés dans le code final