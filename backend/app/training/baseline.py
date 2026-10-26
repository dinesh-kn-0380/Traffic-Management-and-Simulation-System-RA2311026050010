"""
Fixed-Time Baseline Traffic Signal Controller.
Cycles deterministically between North-South and East-West green phases
using configurable green times and standard clearance intervals.
"""
from typing import Dict, Any

class FixedTimeController:
    def __init__(self, fixed_green: float = 30.0):
        self.fixed_green = fixed_green

    def select_action(self, phase_elapsed: float, is_transitioning: bool) -> int:
        """
        If current green has reached the preset fixed green duration,
        request phase switch (Action 1). Otherwise, keep current phase (Action 0).
        """
        if not is_transitioning and phase_elapsed >= self.fixed_green:
            return 1  # SWITCH
        return 0  # KEEP
