"""Optimiseurs et calibration de modèles épidémiologiques."""

from epidemio_sim.calibration.fit import (
    CalibrationData,
    CalibrationResult,
    calibrate_with_optimizer,
    calibrate_with_scipy,
    compare_calibrations,
    loss,
    residuals,
)
from epidemio_sim.calibration.optimizers import (
    Adam,
    Momentum,
    Optimizer,
    OptimizationResult,
    RMSprop,
    SGD,
    numerical_gradient,
    run_optimizer,
)

__all__ = [
    "Adam",
    "CalibrationData",
    "CalibrationResult",
    "Momentum",
    "OptimizationResult",
    "Optimizer",
    "RMSprop",
    "SGD",
    "calibrate_with_optimizer",
    "calibrate_with_scipy",
    "compare_calibrations",
    "loss",
    "numerical_gradient",
    "residuals",
    "run_optimizer",
]
