"""
Evaluation and benchmark comparison engine for Fixed-Time Baseline vs. RL agents.
"""
from typing import Dict, Any, List, Optional
import numpy as np

from app.simulation.environment import TrafficEnvironment
from app.training.baseline import FixedTimeController
from app.agents.base_agent import BaseAgent

def calculate_improvement(baseline_val: float, rl_val: float, lower_is_better: bool = True) -> float:
    """Calculate percentage improvement handling zero denominators safely."""
    if baseline_val == 0.0:
        return 0.0

    if lower_is_better:
        improvement = ((baseline_val - rl_val) / baseline_val) * 100.0
    else:
        improvement = ((rl_val - baseline_val) / baseline_val) * 100.0

    return round(float(improvement), 2)

class EvaluationEngine:
    def __init__(self, env: TrafficEnvironment):
        self.env = env

    def evaluate_controller(
        self,
        controller_type: str,  # 'baseline', 'q_learning', 'dqn'
        agent: Optional[BaseAgent] = None,
        fixed_green: float = 30.0,
        num_episodes: int = 3,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run multiple evaluation episodes for a given controller under fixed seeds.
        """
        if seeds is None:
            seeds = [42 + i * 101 for i in range(num_episodes)]
        else:
            seeds = seeds[:num_episodes]

        fixed_controller = FixedTimeController(fixed_green=fixed_green)
        episode_results = []

        for seed in seeds:
            disc_state, cont_state = self.env.reset(seed=seed)
            done = False

            while not done:
                if controller_type == "baseline":
                    action = fixed_controller.select_action(
                        phase_elapsed=self.env.signal_controller.phase_elapsed_time,
                        is_transitioning=self.env.signal_controller.is_transitioning,
                    )
                elif controller_type == "q_learning":
                    action = agent.select_action(disc_state, deterministic=True)
                elif controller_type == "dqn":
                    action = agent.select_action(cont_state, deterministic=True)
                else:
                    action = 0

                disc_state, cont_state, reward, done, info = self.env.step(action)

            metrics = self.env.get_summary_metrics()
            metrics["seed"] = seed
            episode_results.append(metrics)

        # Aggregate across episodes
        avg_wait = float(np.mean([m["mean_waiting_time"] for m in episode_results]))
        std_wait = float(np.std([m["mean_waiting_time"] for m in episode_results]))

        avg_queue = float(np.mean([m["mean_queue_length"] for m in episode_results]))
        std_queue = float(np.std([m["mean_queue_length"] for m in episode_results]))

        max_queue = int(np.max([m["max_queue_length"] for m in episode_results]))

        avg_throughput = float(np.mean([m["total_departures"] for m in episode_results]))
        std_throughput = float(np.std([m["total_departures"] for m in episode_results]))

        avg_switches = float(np.mean([m["total_switches"] for m in episode_results]))
        avg_reward = float(np.mean([m["cumulative_reward"] for m in episode_results]))

        return {
            "controller": controller_type,
            "scenario": self.env.scenario_id,
            "num_episodes": num_episodes,
            "seeds": seeds,
            "mean_waiting_time": round(avg_wait, 2),
            "std_waiting_time": round(std_wait, 2),
            "mean_queue_length": round(avg_queue, 2),
            "std_queue_length": round(std_queue, 2),
            "max_queue_length": max_queue,
            "mean_throughput": round(avg_throughput, 2),
            "std_throughput": round(std_throughput, 2),
            "mean_switches": round(avg_switches, 2),
            "mean_reward": round(avg_reward, 2),
            "episodes": episode_results,
        }

    def compare(
        self,
        agent: BaseAgent,
        algorithm: str = "q_learning",
        fixed_green: float = 30.0,
        num_episodes: int = 3,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Head-to-head comparison between Fixed-Time Baseline and the RL agent
        under the exact same environment and random seeds.
        """
        if seeds is None:
            seeds = [100, 200, 300][:num_episodes]

        baseline_res = self.evaluate_controller("baseline", fixed_green=fixed_green, num_episodes=num_episodes, seeds=seeds)
        rl_res = self.evaluate_controller(algorithm, agent=agent, num_episodes=num_episodes, seeds=seeds)

        # Improvements
        wait_imp = calculate_improvement(baseline_res["mean_waiting_time"], rl_res["mean_waiting_time"], lower_is_better=True)
        queue_imp = calculate_improvement(baseline_res["mean_queue_length"], rl_res["mean_queue_length"], lower_is_better=True)
        max_q_imp = calculate_improvement(baseline_res["max_queue_length"], rl_res["max_queue_length"], lower_is_better=True)
        tp_imp = calculate_improvement(baseline_res["mean_throughput"], rl_res["mean_throughput"], lower_is_better=False)

        return {
            "scenario": self.env.scenario_id,
            "baseline": baseline_res,
            "rl": rl_res,
            "improvements": {
                "waiting_time_percent": wait_imp,
                "queue_length_percent": queue_imp,
                "max_queue_percent": max_q_imp,
                "throughput_percent": tp_imp,
            },
            "verdict": (
                "RL outperforms baseline controller"
                if (wait_imp > 0 and queue_imp > 0)
                else "Baseline competitive or superior in this scenario"
            ),
        }
