"""
Traffic Environment simulating a 4-way intersection with microscopic vehicle dynamics.
Implements standard RL agent interface (reset, step, get_state) for Q-Learning and DQN.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from app.simulation.vehicle import Vehicle
from app.simulation.signal_controller import SignalController, SignalPhase, SignalState
from app.simulation.traffic_generator import TrafficGenerator
from app.simulation.scenarios import SCENARIOS

class TrafficEnvironment:
    def __init__(
        self,
        scenario_id: str = "balanced",
        episode_duration: float = 300.0,
        dt: float = 1.0,
        road_length: float = 100.0,  # entry to stop line
        intersection_length: float = 25.0,  # through intersection
        departure_length: float = 75.0,  # departure zone
        min_green: float = 10.0,
        max_green: float = 60.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 2.0,
        queue_weight: float = 0.5,
        waiting_weight: float = 0.5,
        switch_weight: float = 1.0,
        throughput_weight: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.scenario_id = scenario_id
        self.episode_duration = episode_duration
        self.dt = dt
        self.road_length = road_length
        self.intersection_length = intersection_length
        self.departure_length = departure_length
        self.stop_line_pos = road_length
        self.road_end_pos = road_length + intersection_length + departure_length

        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration

        # Reward weights
        self.queue_weight = queue_weight
        self.waiting_weight = waiting_weight
        self.switch_weight = switch_weight
        self.throughput_weight = throughput_weight

        # Components
        self.signal_controller = SignalController(
            min_green=min_green,
            max_green=max_green,
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
        )
        self.traffic_generator = TrafficGenerator(scenario_id=scenario_id, seed=seed)

        # Simulation state
        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.vehicles: Dict[str, List[Vehicle]] = {"N": [], "S": [], "E": [], "W": []}
        self.completed_vehicles: List[Vehicle] = []
        self.cumulative_reward: float = 0.0
        self.seed = seed

        # Track incremental queues and waits for statistics
        self.queue_history: List[int] = []
        self.wait_time_history: List[float] = []

    def reset(self, seed: Optional[int] = None) -> Tuple[int, np.ndarray]:
        """
        Reset simulation to initial state.
        Returns:
            (discrete_state: int, continuous_state_vector: np.ndarray)
        """
        if seed is not None:
            self.seed = seed
            self.traffic_generator.set_seed(seed)

        self.sim_time = 0.0
        self.step_count = 0
        self.vehicles = {"N": [], "S": [], "E": [], "W": []}
        self.completed_vehicles = []
        self.cumulative_reward = 0.0
        self.queue_history = []
        self.wait_time_history = []

        self.signal_controller.reset(initial_phase=SignalPhase.NS_GREEN)

        discrete_state = self.get_discrete_state()
        continuous_state = self.get_continuous_state()
        return discrete_state, continuous_state

    def set_scenario(self, scenario_id: str):
        self.scenario_id = scenario_id
        self.traffic_generator.set_scenario(scenario_id)

    def get_queues(self) -> Dict[str, int]:
        """Number of queued vehicles (speed < 0.5 m/s before stop line) per approach."""
        return {
            app: sum(1 for v in self.vehicles[app] if v.is_queued and v.position <= self.stop_line_pos)
            for app in ["N", "S", "E", "W"]
        }

    def get_total_queue(self) -> int:
        return sum(self.get_queues().values())

    def get_aggregate_waiting_times(self) -> Dict[str, float]:
        """Aggregate waiting time of current active vehicles on each approach."""
        return {
            app: sum(v.waiting_time for v in self.vehicles[app] if not v.has_departed)
            for app in ["N", "S", "E", "W"]
        }

    def get_discrete_state(self) -> int:
        """
        Discretize state space into a compact integer index in [0, 287].
        Features:
        1. phase_id (2 values: 0 = NS active, 1 = EW active)
        2. ns_queue_bin (4 values: 0-2, 3-6, 7-11, 12+)
        3. ew_queue_bin (4 values: 0-2, 3-6, 7-11, 12+)
        4. green_time_bin (3 values: <15s, 15-30s, >30s)
        5. wait_ratio_bin (3 values: NS higher, EW higher, Balanced)
        """
        queues = self.get_queues()
        ns_queue = queues["N"] + queues["S"]
        ew_queue = queues["E"] + queues["W"]

        # Phase ID (0 for NS, 1 for EW)
        phase_id = self.signal_controller.current_green_phase_id

        # Queue bins
        def bin_queue(q: int) -> int:
            if q <= 2:
                return 0
            elif q <= 6:
                return 1
            elif q <= 11:
                return 2
            else:
                return 3

        ns_q_bin = bin_queue(ns_queue)
        ew_q_bin = bin_queue(ew_queue)

        # Green time bin
        elapsed = self.signal_controller.phase_elapsed_time
        if elapsed < 15.0:
            gt_bin = 0
        elif elapsed <= 30.0:
            gt_bin = 1
        else:
            gt_bin = 2

        # Wait comparison bin
        waits = self.get_aggregate_waiting_times()
        ns_wait = waits["N"] + waits["S"]
        ew_wait = waits["E"] + waits["W"]

        if ns_wait - ew_wait > 15.0:
            wait_bin = 0  # NS urgency higher
        elif ew_wait - ns_wait > 15.0:
            wait_bin = 1  # EW urgency higher
        else:
            wait_bin = 2  # Balanced

        # Encode into flat discrete index: 2 * 4 * 4 * 3 * 3 = 288
        index = (
            phase_id * (4 * 4 * 3 * 3)
            + ns_q_bin * (4 * 3 * 3)
            + ew_q_bin * (3 * 3)
            + gt_bin * (3)
            + wait_bin
        )
        return int(index)

    def get_continuous_state(self) -> np.ndarray:
        """
        Normalized continuous state vector for DQN of size 10:
        [N_q/15, S_q/15, E_q/15, W_q/15, N_wait/60, S_wait/60, E_wait/60, W_wait/60, phase_id, elapsed/60]
        """
        queues = self.get_queues()
        waits = self.get_aggregate_waiting_times()

        state = [
            min(2.0, queues["N"] / 15.0),
            min(2.0, queues["S"] / 15.0),
            min(2.0, queues["E"] / 15.0),
            min(2.0, queues["W"] / 15.0),
            min(3.0, waits["N"] / 60.0),
            min(3.0, waits["S"] / 60.0),
            min(3.0, waits["E"] / 60.0),
            min(3.0, waits["W"] / 60.0),
            float(self.signal_controller.current_green_phase_id),
            min(2.0, self.signal_controller.phase_elapsed_time / 60.0),
        ]
        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[int, np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step of the environment with the given action.
        action: 0 (KEEP) or 1 (SWITCH).
        Returns:
            (discrete_state, continuous_state, reward, done, info)
        """
        # 1. Advance signal controller
        phase_changed, switch_triggered = self.signal_controller.step(action=action, dt=self.dt)

        # 2. Generate vehicle arrivals
        new_vehicles = self.traffic_generator.generate_arrivals(
            time_s=self.sim_time,
            dt=self.dt,
            vehicles_per_approach=self.vehicles,
        )
        for v in new_vehicles:
            self.vehicles[v.approach].append(v)

        # 3. Update physics for all active vehicles
        departed_this_step = 0
        incremental_wait_this_step = 0.0

        for approach in ["N", "S", "E", "W"]:
            active_list = self.vehicles[approach]
            # Sort by position ascending so lead vehicle is last
            active_list.sort(key=lambda v: v.position)

            can_proceed = self.signal_controller.can_proceed(approach)

            surviving_vehicles = []
            for i, veh in enumerate(active_list):
                # Lead vehicle is the one immediately ahead in the list
                lead_pos = active_list[i + 1].position if (i + 1 < len(active_list)) else None
                lead_length = active_list[i + 1].length if (i + 1 < len(active_list)) else 4.5

                prev_wait = veh.waiting_time
                veh.update_physics(
                    dt=self.dt,
                    lead_vehicle_pos=lead_pos,
                    lead_vehicle_length=lead_length,
                    stop_line_pos=self.stop_line_pos,
                    can_proceed_through_stopline=can_proceed,
                    road_end_pos=self.road_end_pos,
                )
                incremental_wait_this_step += (veh.waiting_time - prev_wait)

                if veh.has_departed:
                    veh.departure_time = self.sim_time + self.dt
                    self.completed_vehicles.append(veh)
                    departed_this_step += 1
                else:
                    surviving_vehicles.append(veh)

            self.vehicles[approach] = surviving_vehicles

        # 4. Advance time
        self.sim_time += self.dt
        self.step_count += 1

        # 5. Calculate reward
        total_queue = self.get_total_queue()
        self.queue_history.append(total_queue)
        self.wait_time_history.append(incremental_wait_this_step)

        # Reward formulation:
        # Penalize queues, incremental waiting time, and unnecessary signal switching.
        # Reward completed vehicle departures.
        switch_penalty = 1.0 if switch_triggered else 0.0
        step_reward = -(
            self.queue_weight * (total_queue / 10.0)
            + self.waiting_weight * (incremental_wait_this_step / 10.0)
            + self.switch_weight * switch_penalty
        ) + self.throughput_weight * departed_this_step

        self.cumulative_reward += step_reward

        # 6. Check termination
        done = self.sim_time >= self.episode_duration

        # 7. Next observations
        discrete_state = self.get_discrete_state()
        continuous_state = self.get_continuous_state()

        info = {
            "sim_time": self.sim_time,
            "step_count": self.step_count,
            "departed_this_step": departed_this_step,
            "total_departures": len(self.completed_vehicles),
            "total_queue": total_queue,
            "queues_per_approach": self.get_queues(),
            "signals": self.signal_controller.get_approach_signals(),
            "current_phase": int(self.signal_controller.current_phase),
            "phase_elapsed": self.signal_controller.phase_elapsed_time,
            "switch_count": self.signal_controller.switch_count,
            "switch_triggered": switch_triggered,
            "incremental_wait": incremental_wait_this_step,
            "cumulative_reward": self.cumulative_reward,
        }

        return discrete_state, continuous_state, step_reward, done, info

    def get_summary_metrics(self) -> Dict[str, float]:
        """Compute end-of-episode evaluation statistics."""
        completed_count = len(self.completed_vehicles)
        if completed_count > 0:
            avg_wait_completed = float(np.mean([v.waiting_time for v in self.completed_vehicles]))
            avg_travel_time = float(np.mean([(v.departure_time - v.arrival_time) for v in self.completed_vehicles if v.departure_time]))
        else:
            avg_wait_completed = 0.0
            avg_travel_time = 0.0

        all_active_vehicles = [v for app in self.vehicles for v in self.vehicles[app]]
        if all_active_vehicles:
            avg_wait_active = float(np.mean([v.waiting_time for v in all_active_vehicles]))
        else:
            avg_wait_active = 0.0

        mean_queue = float(np.mean(self.queue_history)) if self.queue_history else 0.0
        max_queue = int(np.max(self.queue_history)) if self.queue_history else 0

        return {
            "total_departures": completed_count,
            "mean_waiting_time": round(avg_wait_completed if completed_count > 0 else avg_wait_active, 2),
            "mean_queue_length": round(mean_queue, 2),
            "max_queue_length": max_queue,
            "total_switches": self.signal_controller.switch_count,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "avg_travel_time": round(avg_travel_time, 2),
            "active_vehicles": len(all_active_vehicles),
        }

    def render_dict(self) -> Dict[str, Any]:
        """Serialize current state for UI rendering via WebSocket or REST."""
        queues = self.get_queues()
        waits = self.get_aggregate_waiting_times()

        active_vehicles_dict = []
        for app in ["N", "S", "E", "W"]:
            for v in self.vehicles[app]:
                active_vehicles_dict.append(v.to_dict())

        return {
            "sim_time": round(self.sim_time, 1),
            "current_phase": int(self.signal_controller.current_phase),
            "phase_name": self.signal_controller.current_phase.name,
            "phase_elapsed": round(self.signal_controller.phase_elapsed_time, 1),
            "is_transitioning": self.signal_controller.is_transitioning,
            "signals": self.signal_controller.get_approach_signals(),
            "queues": queues,
            "total_queue": sum(queues.values()),
            "aggregate_waits": {k: round(v, 1) for k, v in waits.items()},
            "vehicles": active_vehicles_dict,
            "completed_count": len(self.completed_vehicles),
            "switch_count": self.signal_controller.switch_count,
            "cumulative_reward": round(self.cumulative_reward, 2),
            "geometry": {
                "stop_line_pos": self.stop_line_pos,
                "intersection_length": self.intersection_length,
                "road_end_pos": self.road_end_pos,
            }
        }
