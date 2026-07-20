"""Application web FastAPI exposant epidemio_sim : API de simulation + frontend statique.

Expose un healthcheck et un endpoint de simulation (SIRD / SIRDVital, avec
option de bruit gaussien) au-dessus de la façade ``EpidemicModel.simulate``,
et sert une page statique (``static/``) qui consomme cette API pour
visualiser les trajectoires simulées.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from epidemio_sim import SIRD, SIRDVital, generate_synthetic_data
from epidemio_sim.core.models import EpidemicModel

app = FastAPI(title="epidemio-sim API", version="0.1.0")


class NoiseOptions(BaseModel):
    """Option de bruit gaussien à appliquer à la trajectoire simulée."""

    enabled: bool = False
    level: float = Field(default=0.05, gt=0.0)
    seed: int | None = None


class SimulationRequest(BaseModel):
    """Paramètres d'une requête de simulation.

    Toutes les valeurs par défaut reprennent l'exemple du README, de sorte
    qu'un corps de requête vide (``{}``) produit une simulation valide.
    """

    model: Literal["SIRD", "SIRDVital"] = "SIRD"
    population: float = Field(default=1000.0, gt=0.0)
    s0: float = Field(default=990.0, ge=0.0)
    i0: float = Field(default=10.0, ge=0.0)
    r0: float = Field(default=0.0, ge=0.0)
    d0: float = Field(default=0.0, ge=0.0)
    beta: float = Field(default=0.3, ge=0.0)
    gamma: float = Field(default=0.1, ge=0.0)
    mu: float = Field(default=0.02, ge=0.0)
    birth_rate: float = Field(default=0.01, ge=0.0)
    natural_death_rate: float = Field(default=0.005, ge=0.0)
    t_max: float = Field(default=100.0, gt=0.0)
    n_points: int = Field(default=200, ge=2)
    noise: NoiseOptions = Field(default_factory=NoiseOptions)


class SimulationResponse(BaseModel):
    """Résultat d'une simulation, sérialisable en JSON."""

    model_name: str
    param_names: list[str]
    params: list[float]
    t: list[float]
    compartments: dict[str, list[float]]
    noisy_compartments: dict[str, list[float]] | None


@app.get("/api/health")
def health() -> dict[str, str]:
    """Healthcheck de l'API."""
    return {"status": "ok"}


@app.post("/api/simulate")
def simulate(req: SimulationRequest) -> SimulationResponse:
    """Simule un modèle SIRD ou SIRDVital et retourne la trajectoire (+ bruit optionnel)."""
    initial_state = {"S": req.s0, "I": req.i0, "R": req.r0, "D": req.d0}
    t_span = (0.0, req.t_max)
    t_eval = np.linspace(0.0, req.t_max, req.n_points)

    try:
        model: EpidemicModel
        if req.model == "SIRD":
            model = SIRD(population=req.population)
            params = SIRD.params(beta=req.beta, gamma=req.gamma, mu=req.mu)
        else:
            model = SIRDVital(population=req.population)
            params = SIRDVital.params(
                beta=req.beta,
                gamma=req.gamma,
                mu=req.mu,
                birth_rate=req.birth_rate,
                natural_death_rate=req.natural_death_rate,
            )

        result = model.simulate(initial_state, params, t_span=t_span, t_eval=t_eval)
        compartments = {c: result.get(c).tolist() for c in result.compartments}

        noisy_compartments: dict[str, list[float]] | None = None
        if req.noise.enabled:
            calib_data = generate_synthetic_data(
                model,
                initial_state,
                params,
                t_eval,
                noise="gaussian",
                noise_level=req.noise.level,
                seed=req.noise.seed,
            )
            noisy_compartments = {c: arr.tolist() for c, arr in calib_data.observations.items()}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return SimulationResponse(
        model_name=result.model_name,
        param_names=list(result.param_names),
        params=result.params.tolist(),
        t=result.t.tolist(),
        compartments=compartments,
        noisy_compartments=noisy_compartments,
    )


_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("webclient.server:app", host="127.0.0.1", port=8000, reload=True)
