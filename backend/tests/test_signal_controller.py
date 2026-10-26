"""
Unit tests for Traffic Signal Controller and safety invariants.
"""
import pytest
from app.simulation.signal_controller import SignalController, SignalPhase, SignalState

def test_initial_state():
    ctrl = SignalController(min_green=10.0, max_green=60.0, yellow_duration=3.0, all_red_duration=2.0)
    assert ctrl.current_phase == SignalPhase.NS_GREEN
    assert ctrl.phase_elapsed_time == 0.0
    assert not ctrl.is_transitioning

    signals = ctrl.get_approach_signals()
    assert signals["N"] == SignalState.GREEN
    assert signals["S"] == SignalState.GREEN
    assert signals["E"] == SignalState.RED
    assert signals["W"] == SignalState.RED

def test_min_green_constraint():
    ctrl = SignalController(min_green=10.0, max_green=60.0)
    ctrl.reset(SignalPhase.NS_GREEN)

    # Attempt to switch before min_green has elapsed
    changed, triggered = ctrl.step(action=1, dt=2.0)
    assert not changed
    assert not triggered
    assert ctrl.current_phase == SignalPhase.NS_GREEN
    assert ctrl.phase_elapsed_time == 2.0

def test_valid_switch_and_transitions():
    ctrl = SignalController(min_green=10.0, max_green=60.0, yellow_duration=3.0, all_red_duration=2.0)
    ctrl.reset(SignalPhase.NS_GREEN)

    # Advance past min_green
    ctrl.step(action=0, dt=10.0)
    assert ctrl.phase_elapsed_time == 10.0

    # Request switch -> should enter NS_YELLOW
    changed, triggered = ctrl.step(action=1, dt=1.0)
    assert ctrl.current_phase == SignalPhase.NS_YELLOW
    assert ctrl.is_transitioning

    # Yellow clearance duration (3.0s) -> should transition to ALL_RED_TO_EW
    ctrl.step(action=0, dt=3.0)
    assert ctrl.current_phase == SignalPhase.ALL_RED_TO_EW
    signals = ctrl.get_approach_signals()
    assert all(sig == SignalState.RED for sig in signals.values())

    # All-red clearance duration (2.0s) -> should transition to EW_GREEN
    ctrl.step(action=0, dt=2.0)
    assert ctrl.current_phase == SignalPhase.EW_GREEN
    assert not ctrl.is_transitioning
    signals = ctrl.get_approach_signals()
    assert signals["E"] == SignalState.GREEN
    assert signals["W"] == SignalState.GREEN
    assert signals["N"] == SignalState.RED
    assert signals["S"] == SignalState.RED

def test_max_green_forced_switch():
    ctrl = SignalController(min_green=10.0, max_green=60.0)
    ctrl.reset(SignalPhase.NS_GREEN)

    # Keep action=0 up to max_green
    ctrl.step(action=0, dt=59.0)
    assert ctrl.current_phase == SignalPhase.NS_GREEN

    # At 60.0s, forced switch occurs even if action is 0 (KEEP)
    changed, triggered = ctrl.step(action=0, dt=1.0)
    assert ctrl.current_phase == SignalPhase.NS_YELLOW
    assert not triggered  # forced by safety, not action

def test_safety_invariants_never_conflicting_greens():
    ctrl = SignalController()
    # Step through 200 cycles and verify invariant hold on every step
    for _ in range(200):
        ctrl.step(action=1, dt=1.0)
        assert ctrl.verify_safety_invariants() is True
