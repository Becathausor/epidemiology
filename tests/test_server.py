"""Tests d'intégration pour l'API FastAPI de webclient.server."""

from __future__ import annotations

from fastapi.testclient import TestClient

from webclient.server import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_simulate_defaults_returns_expected_shape():
    response = client.post("/api/simulate", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "SIRD"
    assert len(body["t"]) == 200
    assert set(body["compartments"].keys()) == {"S", "I", "R", "D"}
    for values in body["compartments"].values():
        assert len(values) == 200
    assert body["noisy_compartments"] is None


def test_simulate_bad_initial_state_returns_422():
    response = client.post(
        "/api/simulate",
        json={"population": 1000.0, "s0": 500.0, "i0": 10.0, "r0": 0.0, "d0": 0.0},
    )
    assert response.status_code == 422


def test_simulate_sirdvital_works():
    response = client.post(
        "/api/simulate",
        json={"model": "SIRDVital", "birth_rate": 0.01, "natural_death_rate": 0.005},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "SIRDVital"
    assert body["param_names"] == ["beta", "gamma", "mu", "birth_rate", "natural_death_rate"]


def test_simulate_with_noise_returns_noisy_compartments():
    response = client.post(
        "/api/simulate",
        json={"noise": {"enabled": True, "level": 0.05, "seed": 42}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["noisy_compartments"] is not None
    assert set(body["noisy_compartments"].keys()) == {"S", "I", "R", "D"}
    for values in body["noisy_compartments"].values():
        assert len(values) == 200
