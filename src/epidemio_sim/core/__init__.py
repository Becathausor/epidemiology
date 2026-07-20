"""Modèles épidémiologiques."""

from epidemio_sim.core.epidemic_model import EpidemicModel
from epidemio_sim.core.sird import SIRD
from epidemio_sim.core.sird_params import SIRDParams
from epidemio_sim.core.sird_vital import SIRDVital
from epidemio_sim.core.sird_vital_params import SIRDVitalParams
from epidemio_sim.core.simulation_result import SimulationResult

__all__ = [
    "EpidemicModel",
    "SIRD",
    "SIRDParams",
    "SIRDVital",
    "SIRDVitalParams",
    "SimulationResult",
]
