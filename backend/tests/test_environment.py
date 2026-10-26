"""
Unit tests for Traffic Environment microscopic simulation and dynamics.
"""
import pytest
import numpy as np
from app.simulation.environment import TrafficEnvironment
from app.simulation.vehicle import Vehicle

def test_environment_reset():
    env = TrafficEnvironment(scenario_id="balanced", episode_duration=100.0)
    disc_s, cont_s = env.reset(seed=42)

    assert isinstance(disc_s, (int, np.integer))
    assert 0 <= disc_s < 288
    assert isinstance(cont_s, np.ndarray)
    assert len(cont_s) == 10
    assert env.sim_time == 0.0
    assert env.step_count == 0
    assert len(env.completed_vehicles) == 0

def test_environment_step():
    env = TrafficEnvironment(scenario_id="balanced", episode_duration=50.0)
    env.reset(seed=42)

    disc_s, cont_s, reward, done, info = env.step(action=0)
    assert env.sim_time == 1.0
    assert env.step_count == 1
    assert isinstance(reward, float)
    assert done is False
    assert "signals" in info
    assert "queues_per_approach" in info

def test_vehicle_waiting_time_accumulation():
    # Create vehicle stopped directly at the stop line
    veh = Vehicle(id=1, approach="N", lane=0, position=98.0, speed=0.0)
    # Stopline at 100.0, red light (can_proceed=False)
    veh.update_physics(
        dt=1.0,
        lead_vehicle_pos=None,
        lead_vehicle_length=4.5,
        stop_line_pos=100.0,
        can_proceed_through_stopline=False,
        road_end_pos=200.0
    )
    assert veh.is_queued
    assert veh.waiting_time == 1.0

def test_vehicle_crossing_and_departure():
    veh = Vehicle(id=1, approach="N", lane=0, position=98.0, speed=10.0)
    # Green light (can_proceed=True)
    veh.update_physics(
        dt=1.0,
        lead_vehicle_pos=None,
        lead_vehicle_length=4.5,
        stop_line_pos=100.0,
        can_proceed_through_stopline=True,
        road_end_pos=105.0  # short end for testing departure
    )
    assert veh.position > 100.0
    assert veh.has_departed is True

def test_discrete_state_bounds():
    env = TrafficEnvironment()
    env.reset()
    for _ in range(50):
        disc_s, cont_s, _, _, _ = env.step(action=0)
        assert 0 <= disc_s < 288
        assert len(cont_s) == 10
        assert np.all(cont_s >= 0.0)
