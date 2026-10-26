"""
Traffic signal phase controller enforcing clearance intervals, minimum/maximum green times,
and strict safety constraints preventing conflicting greens.
"""
from enum import IntEnum
from typing import Dict, Tuple

class SignalPhase(IntEnum):
    NS_GREEN = 0
    NS_YELLOW = 1
    ALL_RED_TO_EW = 2
    EW_GREEN = 3
    EW_YELLOW = 4
    ALL_RED_TO_NS = 5

class SignalState:
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"

class SignalController:
    def __init__(
        self,
        min_green: float = 10.0,
        max_green: float = 60.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 2.0,
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration

        self.current_phase: SignalPhase = SignalPhase.NS_GREEN
        self.phase_elapsed_time: float = 0.0
        self.switch_count: int = 0
        self.just_switched: bool = False

    def reset(self, initial_phase: SignalPhase = SignalPhase.NS_GREEN):
        self.current_phase = initial_phase
        self.phase_elapsed_time = 0.0
        self.switch_count = 0
        self.just_switched = False

    @property
    def is_transitioning(self) -> bool:
        """Returns True if controller is currently in yellow or all-red clearance."""
        return self.current_phase in (
            SignalPhase.NS_YELLOW,
            SignalPhase.ALL_RED_TO_EW,
            SignalPhase.EW_YELLOW,
            SignalPhase.ALL_RED_TO_NS,
        )

    @property
    def current_green_phase_id(self) -> int:
        """
        Returns 0 for North-South active green (or clearance from NS),
        and 1 for East-West active green (or clearance from EW).
        """
        if self.current_phase in (SignalPhase.NS_GREEN, SignalPhase.NS_YELLOW, SignalPhase.ALL_RED_TO_EW):
            return 0
        return 1

    def get_approach_signals(self) -> Dict[str, str]:
        """
        Returns mapping of approach ('N', 'S', 'E', 'W') to current signal state
        ('GREEN', 'YELLOW', 'RED').
        """
        if self.current_phase == SignalPhase.NS_GREEN:
            return {"N": SignalState.GREEN, "S": SignalState.GREEN, "E": SignalState.RED, "W": SignalState.RED}
        elif self.current_phase == SignalPhase.NS_YELLOW:
            return {"N": SignalState.YELLOW, "S": SignalState.YELLOW, "E": SignalState.RED, "W": SignalState.RED}
        elif self.current_phase == SignalPhase.ALL_RED_TO_EW:
            return {"N": SignalState.RED, "S": SignalState.RED, "E": SignalState.RED, "W": SignalState.RED}
        elif self.current_phase == SignalPhase.EW_GREEN:
            return {"N": SignalState.RED, "S": SignalState.RED, "E": SignalState.GREEN, "W": SignalState.GREEN}
        elif self.current_phase == SignalPhase.EW_YELLOW:
            return {"N": SignalState.RED, "S": SignalState.RED, "E": SignalState.YELLOW, "W": SignalState.YELLOW}
        elif self.current_phase == SignalPhase.ALL_RED_TO_NS:
            return {"N": SignalState.RED, "S": SignalState.RED, "E": SignalState.RED, "W": SignalState.RED}
        else:
            raise ValueError(f"Unknown signal phase: {self.current_phase}")

    def verify_safety_invariants(self) -> bool:
        """Strict check: conflicting approaches must NEVER both have non-RED signals."""
        signals = self.get_approach_signals()
        ns_active = signals["N"] in (SignalState.GREEN, SignalState.YELLOW) or signals["S"] in (SignalState.GREEN, SignalState.YELLOW)
        ew_active = signals["E"] in (SignalState.GREEN, SignalState.YELLOW) or signals["W"] in (SignalState.GREEN, SignalState.YELLOW)
        if ns_active and ew_active:
            raise RuntimeError(f"FATAL SAFETY VIOLATION: Simultaneous conflicting active signals! {signals}")
        return True

    def can_proceed(self, approach: str) -> bool:
        """Check if vehicles on given approach have right-of-way to cross stop line."""
        signals = self.get_approach_signals()
        # Vehicles may proceed through green; at yellow, vehicles past or at stop line may clear
        return signals.get(approach) == SignalState.GREEN

    def step(self, action: int, dt: float) -> Tuple[bool, bool]:
        """
        Advance signal controller by dt seconds.
        action: 0 = KEEP current phase, 1 = SWITCH phase.
        Returns:
            (phase_changed: bool, switch_triggered_by_action: bool)
        """
        self.phase_elapsed_time += dt
        self.just_switched = False
        action_triggered_switch = False

        if self.is_transitioning:
            # Automatic progression through clearance intervals
            if self.current_phase == SignalPhase.NS_YELLOW and self.phase_elapsed_time >= self.yellow_duration:
                self.current_phase = SignalPhase.ALL_RED_TO_EW
                self.phase_elapsed_time = 0.0
                self.just_switched = True
            elif self.current_phase == SignalPhase.ALL_RED_TO_EW and self.phase_elapsed_time >= self.all_red_duration:
                self.current_phase = SignalPhase.EW_GREEN
                self.phase_elapsed_time = 0.0
                self.just_switched = True
            elif self.current_phase == SignalPhase.EW_YELLOW and self.phase_elapsed_time >= self.yellow_duration:
                self.current_phase = SignalPhase.ALL_RED_TO_NS
                self.phase_elapsed_time = 0.0
                self.just_switched = True
            elif self.current_phase == SignalPhase.ALL_RED_TO_NS and self.phase_elapsed_time >= self.all_red_duration:
                self.current_phase = SignalPhase.NS_GREEN
                self.phase_elapsed_time = 0.0
                self.just_switched = True
        else:
            # Active green phase: NS_GREEN or EW_GREEN
            must_force_switch = self.phase_elapsed_time >= self.max_green
            requested_switch = (action == 1) and (self.phase_elapsed_time >= self.min_green)

            if must_force_switch or requested_switch:
                if self.current_phase == SignalPhase.NS_GREEN:
                    self.current_phase = SignalPhase.NS_YELLOW
                    self.phase_elapsed_time = 0.0
                    self.switch_count += 1
                    self.just_switched = True
                    action_triggered_switch = requested_switch
                elif self.current_phase == SignalPhase.EW_GREEN:
                    self.current_phase = SignalPhase.EW_YELLOW
                    self.phase_elapsed_time = 0.0
                    self.switch_count += 1
                    self.just_switched = True
                    action_triggered_switch = requested_switch

        # Always verify safety invariants
        self.verify_safety_invariants()

        return self.just_switched, action_triggered_switch
