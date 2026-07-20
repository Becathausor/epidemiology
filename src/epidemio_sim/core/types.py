"""Alias de types numpy partagés par les modèles épidémiologiques."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

ParamVector = npt.NDArray[np.float64]
StateVector = npt.NDArray[np.float64]
