"""Export des résultats de simulation et de calibration."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from epidemio_sim.calibration.fit import CalibrationResult
from epidemio_sim.core.models import SimulationResult


def export_simulation_csv(result: SimulationResult, path: str | Path) -> None:
    """Exporte une trajectoire simulée au format CSV (colonnes t + compartiments)."""
    result.to_dataframe().to_csv(path, index=False)


def export_calibration_json(results: dict[str, CalibrationResult], path: str | Path) -> None:
    """Exporte un ensemble de résultats de calibration au format JSON.

    Les tableaux numpy sont convertis en listes pour être sérialisables.
    """
    serializable = {}
    for name, result in results.items():
        payload = dataclasses.asdict(result)
        payload["fitted_params"] = result.fitted_params.tolist()
        payload["loss_history"] = (
            result.loss_history.tolist() if result.loss_history is not None else None
        )
        serializable[name] = payload

    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
