"""
Unit tests for FastAPI endpoints using TestClient.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["backend"] == "FastAPI"

def test_algorithms():
    res = client.get("/api/algorithms")
    assert res.status_code == 200
    algos = res.json()["algorithms"]
    assert len(algos) == 3
    ids = [a["id"] for a in algos]
    assert "q_learning" in ids
    assert "dqn" in ids
    assert "baseline" in ids

def test_scenarios():
    res = client.get("/api/scenarios")
    assert res.status_code == 200
    scenarios = res.json()["scenarios"]
    assert len(scenarios) >= 5

def test_config_get_and_post():
    res = client.get("/api/config")
    assert res.status_code == 200
    assert "timing" in res.json()

    # Update config
    update_res = client.post("/api/config", json={"min_green": 12.0, "queue_weight": 0.8})
    assert update_res.status_code == 200
    updated = update_res.json()["config"]
    assert updated["timing"]["min_green"] == 12.0
    assert updated["rewards"]["queue_weight"] == 0.8

def test_simulation_state_and_step():
    res = client.get("/api/simulation/state")
    assert res.status_code == 200
    data = res.json()
    assert "signals" in data
    assert "queues" in data
    assert "vehicles" in data

    # Step simulation
    step_res = client.post("/api/simulation/step", json={"action": 0, "controller_type": "baseline"})
    assert step_res.status_code == 200
    assert step_res.json()["status"] == "stepped"
