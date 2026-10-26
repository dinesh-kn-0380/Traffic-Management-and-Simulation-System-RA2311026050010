"""
Pre-defined traffic demand scenarios for evaluation and training.
"""
from dataclasses import dataclass
from typing import Dict, Any, Callable
import math
import random

@dataclass
class TrafficScenario:
    id: str
    name: str
    description: str
    base_rates: Dict[str, float]  # veh/sec per approach: 'N', 'S', 'E', 'W'

    def get_rates(self, time_s: float) -> Dict[str, float]:
        """Return instantaneous arrival rates for each approach at time t (seconds)."""
        rates = self.base_rates.copy()

        if self.id == "morning_rush":
            # Inbound commuter surge (North & East high, peaking around t=60-180s)
            peak_factor = 1.0 + 0.8 * math.sin(min(math.pi, max(0.0, (time_s - 30.0) / 120.0 * math.pi)))
            rates["N"] *= peak_factor
            rates["E"] *= peak_factor
        elif self.id == "evening_rush":
            # Outbound commuter surge (South & West high, peaking around t=60-180s)
            peak_factor = 1.0 + 0.8 * math.sin(min(math.pi, max(0.0, (time_s - 30.0) / 120.0 * math.pi)))
            rates["S"] *= peak_factor
            rates["W"] *= peak_factor
        elif self.id == "stochastic":
            # Time-varying sinusoidal fluctuations + random noise
            for app in rates:
                fluctuation = 0.5 * math.sin(time_s / 20.0 + (hash(app) % 10))
                rates[app] = max(0.02, rates[app] * (1.0 + fluctuation))
        elif self.id == "surge":
            # Normal baseline, then sudden massive burst on North & South between t=40s and t=90s
            if 40.0 <= time_s <= 90.0:
                rates["N"] += 0.35
                rates["S"] += 0.30

        return rates


SCENARIOS: Dict[str, TrafficScenario] = {
    "balanced": TrafficScenario(
        id="balanced",
        name="Balanced Traffic",
        description="Evenly distributed moderate traffic across all four approaches (0.2 veh/s each).",
        base_rates={"N": 0.20, "S": 0.20, "E": 0.20, "W": 0.20}
    ),
    "heavy_north_south": TrafficScenario(
        id="heavy_north_south",
        name="Heavy North–South",
        description="Major arterial flow along the North-South corridor (0.35 veh/s) with light cross-traffic (0.08 veh/s).",
        base_rates={"N": 0.35, "S": 0.35, "E": 0.08, "W": 0.08}
    ),
    "heavy_east_west": TrafficScenario(
        id="heavy_east_west",
        name="Heavy East–West",
        description="Major arterial flow along the East-West corridor (0.35 veh/s) with light cross-traffic (0.08 veh/s).",
        base_rates={"N": 0.08, "S": 0.08, "E": 0.35, "W": 0.35}
    ),
    "morning_rush": TrafficScenario(
        id="morning_rush",
        name="Morning Rush Hour",
        description="Inbound commuter wave with dynamic traffic surge entering from North and East corridors.",
        base_rates={"N": 0.25, "S": 0.10, "E": 0.25, "W": 0.10}
    ),
    "evening_rush": TrafficScenario(
        id="evening_rush",
        name="Evening Rush Hour",
        description="Outbound commuter wave with dynamic traffic surge heading along South and West corridors.",
        base_rates={"N": 0.10, "S": 0.25, "E": 0.10, "W": 0.25}
    ),
    "stochastic": TrafficScenario(
        id="stochastic",
        name="Stochastic & Dynamic",
        description="Fluctuating arrival rates with sinusoidal variations and random demand peaks.",
        base_rates={"N": 0.18, "S": 0.18, "E": 0.18, "W": 0.18}
    ),
    "surge": TrafficScenario(
        id="surge",
        name="Sudden Traffic Surge",
        description="Sudden unannounced surge of vehicles on North-South corridor testing adaptive responsiveness.",
        base_rates={"N": 0.12, "S": 0.12, "E": 0.12, "W": 0.12}
    ),
}
