"""epidemio_sim : simulation et calibration de modèles épidémiologiques.

Façade publique : expose les classes et fonctions les plus utilisées
directement depuis le package racine.
"""

from epidemio_sim.calibration import (
    Adam,
    CalibrationData,
    CalibrationResult,
    Momentum,
    Optimizer,
    RMSprop,
    SGD,
    calibrate_with_optimizer,
    calibrate_with_scipy,
    compare_calibrations,
)
from epidemio_sim.core import EpidemicModel, SIRD, SIRDVital, SimulationResult
from epidemio_sim.io import generate_synthetic_data

__version__ = "0.1.0"

__all__ = [
    "Adam",
    "CalibrationData",
    "CalibrationResult",
    "EpidemicModel",
    "Momentum",
    "Optimizer",
    "RMSprop",
    "SGD",
    "SIRD",
    "SIRDVital",
    "SimulationResult",
    "calibrate_with_optimizer",
    "calibrate_with_scipy",
    "compare_calibrations",
    "generate_synthetic_data",
]
