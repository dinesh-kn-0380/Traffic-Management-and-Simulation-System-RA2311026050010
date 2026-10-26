"""
Traffic generator handling vehicle arrivals and microscopic spawning.
"""
from typing import List, Dict, Optional
import random
import math
from app.simulation.vehicle import Vehicle
from app.simulation.scenarios import TrafficScenario, SCENARIOS

class TrafficGenerator:
    def __init__(self, scenario_id: str = "balanced", seed: Optional[int] = None):
        self.scenario_id = scenario_id
        self.scenario: TrafficScenario = SCENARIOS.get(scenario_id, SCENARIOS["balanced"])
        self.rng = random.Random(seed)
        self.next_vehicle_id = 1

    def set_seed(self, seed: Optional[int]):
        self.rng = random.Random(seed)
        self.next_vehicle_id = 1

    def set_scenario(self, scenario_id: str):
        if scenario_id in SCENARIOS:
            self.scenario_id = scenario_id
            self.scenario = SCENARIOS[scenario_id]

    def generate_arrivals(
        self,
        time_s: float,
        dt: float,
        vehicles_per_approach: Dict[str, List[Vehicle]],
        min_entry_gap: float = 8.0
    ) -> List[Vehicle]:
        """
        Check Poisson arrival process for each approach.
        If arrival occurs and there is enough clear space at road entry (position 0),
        spawn a new vehicle.
        """
        rates = self.scenario.get_rates(time_s)
        new_vehicles = []

        for approach in ["N", "S", "E", "W"]:
            arrival_rate = rates.get(approach, 0.20)
            # Poisson arrival probability in interval dt: P = 1 - exp(-rate * dt)
            prob = 1.0 - math.exp(-arrival_rate * dt)

            if self.rng.random() < prob:
                # Check entry space
                existing_on_approach = vehicles_per_approach.get(approach, [])
                can_spawn = True

                if existing_on_approach:
                    # Find vehicle nearest to the road entry (smallest position)
                    entry_veh = min(existing_on_approach, key=lambda v: v.position)
                    if entry_veh.position < min_entry_gap:
                        can_spawn = False

                if can_spawn:
                    new_veh = Vehicle(
                        id=self.next_vehicle_id,
                        approach=approach,
                        lane=0,
                        position=0.0,
                        speed=min(8.0, 13.89),
                        arrival_time=time_s
                    )
                    self.next_vehicle_id += 1
                    new_vehicles.append(new_veh)

        return new_vehicles
