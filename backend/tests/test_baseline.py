"""
Unit tests for Fixed-Time Baseline Controller.
"""
import pytest
from app.training.baseline import FixedTimeController

def test_baseline_selection():
    ctrl = FixedTimeController(fixed_green=30.0)

    # When elapsed green is below fixed duration -> KEEP (0)
    action = ctrl.select_action(phase_elapsed=15.0, is_transitioning=False)
    assert action == 0

    # When elapsed green reaches or exceeds fixed duration -> SWITCH (1)
    action = ctrl.select_action(phase_elapsed=30.0, is_transitioning=False)
    assert action == 1

    # When in yellow or clearance transition -> KEEP (0)
    action = ctrl.select_action(phase_elapsed=35.0, is_transitioning=True)
    assert action == 0
