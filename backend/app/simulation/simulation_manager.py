"""
Simulation Manager for interactive live simulation sessions.
Handles play/pause/reset/speed, multiple controller modes (manual, baseline, Q-learning, DQN),
and continuous state broadcasting.
"""
from typing import Dict, Any, Optional
import threading
import time
import os

from app.simulation.environment import TrafficEnvironment
from app.simulation.signal_controller import SignalPhase
from app.training.baseline import FixedTimeController
from app.agents.q_learning import QLearningAgent
from app.agents.dqn import DQNAgent

class LiveSimulationManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.env = TrafficEnvironment(scenario_id="balanced", episode_duration=3600.0)
        self.controller_type: str = "baseline"  # 'baseline', 'q_learning', 'dqn', 'manual'
        self.baseline_controller = FixedTimeController(fixed_green=30.0)

        self.q_agent: Optional[QLearningAgent] = None
        self.dqn_agent: Optional[DQNAgent] = None

        self.is_running: bool = False
        self.speed_multiplier: float = 1.0  # 1x, 2x, 5x
        self.tick_interval: float = 0.5  # real-time seconds per simulation step

        self.sim_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Try loading latest trained models if available
        self._load_default_models()

    def _load_default_models(self):
        q_path = os.path.join(self.models_dir, "q_learning_latest.json")
        if os.path.exists(q_path):
            try:
                self.q_agent = QLearningAgent()
                self.q_agent.load(q_path)
            except Exception:
                self.q_agent = None

        dqn_path = os.path.join(self.models_dir, "dqn_latest.pt")
        if os.path.exists(dqn_path):
            try:
                self.dqn_agent = DQNAgent()
                self.dqn_agent.load(dqn_path)
            except Exception:
                self.dqn_agent = None

    def reset(self, scenario_id: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            if scenario_id:
                self.env.set_scenario(scenario_id)
            self.env.reset(seed=seed)
            return self.env.render_dict()

    def set_controller(self, controller_type: str, model_path: Optional[str] = None):
        with self.lock:
            self.controller_type = controller_type
            if controller_type == "q_learning" and model_path:
                self.q_agent = QLearningAgent()
                self.q_agent.load(model_path)
            elif controller_type == "dqn" and model_path:
                self.dqn_agent = DQNAgent()
                self.dqn_agent.load(model_path)

    def set_speed(self, speed: float):
        with self.lock:
            self.speed_multiplier = max(0.2, min(10.0, speed))

    def step(self, manual_action: int = 0) -> Dict[str, Any]:
        """Single simulation step under current controller."""
        with self.lock:
            disc_state = self.env.get_discrete_state()
            cont_state = self.env.get_continuous_state()

            if self.controller_type == "manual":
                action = manual_action
            elif self.controller_type == "baseline":
                action = self.baseline_controller.select_action(
                    phase_elapsed=self.env.signal_controller.phase_elapsed_time,
                    is_transitioning=self.env.signal_controller.is_transitioning,
                )
            elif self.controller_type == "q_learning":
                if self.q_agent:
                    action = self.q_agent.select_action(disc_state, deterministic=True)
                else:
                    action = self.baseline_controller.select_action(
                        phase_elapsed=self.env.signal_controller.phase_elapsed_time,
                        is_transitioning=self.env.signal_controller.is_transitioning,
                    )
            elif self.controller_type == "dqn":
                if self.dqn_agent:
                    action = self.dqn_agent.select_action(cont_state, deterministic=True)
                else:
                    action = self.baseline_controller.select_action(
                        phase_elapsed=self.env.signal_controller.phase_elapsed_time,
                        is_transitioning=self.env.signal_controller.is_transitioning,
                    )
            else:
                action = 0

            self.env.step(action)
            return self.env.render_dict()

    def start(self):
        with self.lock:
            if self.is_running:
                return
            self.is_running = True
            self.sim_thread = threading.Thread(target=self._loop, daemon=True)
            self.sim_thread.start()

    def pause(self):
        with self.lock:
            self.is_running = False

    def _loop(self):
        while self.is_running:
            self.step()
            sleep_time = max(0.01, (self.tick_interval / max(0.1, self.speed_multiplier)))
            time.sleep(sleep_time)

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            state = self.env.render_dict()
            state["controller_type"] = self.controller_type
            state["is_running"] = self.is_running
            state["speed_multiplier"] = self.speed_multiplier
            return state
