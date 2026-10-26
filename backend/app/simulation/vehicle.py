"""
Vehicle model for microscopic traffic simulation.
Tracks position, speed, waiting time, and microscopic car-following behavior.
"""
from dataclasses import dataclass, field
from typing import Optional
import random

@dataclass
class Vehicle:
    id: int
    approach: str  # 'N', 'S', 'E', 'W'
    lane: int  # 0: straight/right
    position: float  # distance along road (meters). 0 = entry point, road_length = stop line
    speed: float = 0.0  # current speed in m/s
    max_speed: float = 13.89  # ~50 km/h in m/s
    acceleration: float = 2.5  # max acceleration m/s^2
    deceleration: float = 4.0  # max deceleration m/s^2
    length: float = 4.5  # vehicle length in meters
    min_gap: float = 2.0  # min distance to vehicle ahead (meters)
    waiting_time: float = 0.0  # cumulative seconds vehicle has been stopped/queued
    arrival_time: float = 0.0  # simulation timestamp when vehicle entered
    departure_time: Optional[float] = None
    has_departed: bool = False
    color: str = field(default_factory=lambda: random.choice([
        "#3B82F6", "#60A5FA", "#10B981", "#34D399", "#F59E0B",
        "#EC4899", "#8B5CF6", "#06B6D4", "#F97316"
    ]))

    @property
    def is_queued(self) -> bool:
        """Vehicle is considered in a queue if moving slower than 0.5 m/s."""
        return self.speed < 0.5 and not self.has_departed

    def update_physics(
        self,
        dt: float,
        lead_vehicle_pos: Optional[float],
        lead_vehicle_length: float,
        stop_line_pos: float,
        can_proceed_through_stopline: bool,
        road_end_pos: float
    ):
        """
        Microscopic driver model:
        - Maintains safe distance from lead vehicle.
        - Respects stop line when signal does not permit entry.
        - Accelerates towards max_speed on open road.
        - Smooth deceleration before obstacle / red light.
        """
        if self.has_departed:
            return

        # Target stopping distance & position
        target_stop_pos = None

        # 1. Lead vehicle constraint
        if lead_vehicle_pos is not None:
            # Must stop behind lead vehicle
            target_stop_pos = lead_vehicle_pos - lead_vehicle_length - self.min_gap

        # 2. Traffic light constraint (only if vehicle hasn't already crossed stop line)
        if not can_proceed_through_stopline and self.position <= stop_line_pos:
            light_stop_pos = stop_line_pos - self.min_gap
            if target_stop_pos is None or light_stop_pos < target_stop_pos:
                target_stop_pos = light_stop_pos

        # Calculate desired acceleration / deceleration
        desired_accel = 0.0
        if target_stop_pos is not None:
            distance_to_target = target_stop_pos - self.position
            if distance_to_target <= 0.0:
                # Already at or past target point
                self.speed = 0.0
                desired_accel = 0.0
            else:
                # Safe stopping velocity v = sqrt(2 * a * d)
                safe_speed = (2.0 * self.deceleration * max(0.0, distance_to_target)) ** 0.5
                target_speed = min(self.max_speed, safe_speed)

                if self.speed < target_speed:
                    desired_accel = min(self.acceleration, (target_speed - self.speed) / dt)
                else:
                    desired_accel = -min(self.deceleration, (self.speed - target_speed) / dt)
        else:
            # Free flow: accelerate up to max_speed
            if self.speed < self.max_speed:
                desired_accel = min(self.acceleration, (self.max_speed - self.speed) / dt)
            else:
                desired_accel = 0.0

        # Update speed and position
        self.speed = max(0.0, self.speed + desired_accel * dt)
        self.position += self.speed * dt

        # Update waiting time if queued
        if self.speed < 0.5 and self.position <= stop_line_pos:
            self.waiting_time += dt

        # Check if departed past road end
        if self.position >= road_end_pos:
            self.has_departed = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "approach": self.approach,
            "lane": self.lane,
            "position": round(self.position, 2),
            "speed": round(self.speed, 2),
            "waiting_time": round(self.waiting_time, 2),
            "is_queued": self.is_queued,
            "has_departed": self.has_departed,
            "color": self.color,
            "length": self.length,
        }
