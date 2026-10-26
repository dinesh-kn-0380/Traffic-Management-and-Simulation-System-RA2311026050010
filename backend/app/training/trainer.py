"""
Training engine supporting background execution, pause, resume, stop,
and real-time metric broadcasting for Tabular Q-Learning and Deep Q-Network.
"""
from typing import Dict, Any, Optional, Callable
import threading
import time
import os

from app.simulation.environment import TrafficEnvironment
from app.agents.q_learning import QLearningAgent
from app.agents.dqn import DQNAgent
from app.agents.base_agent import BaseAgent

class TrainingManager:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)

        self.is_training: bool = False
        self.is_paused: bool = False
        self.stop_requested: bool = False

        self.current_algorithm: str = "q_learning"
        self.current_episode: int = 0
        self.total_episodes: int = 50
        self.episode_duration: float = 200.0
        self.scenario_id: str = "balanced"

        self.agent: Optional[BaseAgent] = None
        self.env: Optional[TrafficEnvironment] = None

        self.training_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Training history
        self.history: list[dict[str, Any]] = []
        self.status_message: str = "Idle"
        self.start_time: float = 0.0

        # Optional callback for notifying listeners (e.g. WebSocket or SSE)
        self.on_episode_end: Optional[Callable[[Dict[str, Any]], None]] = None

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            elapsed = time.time() - self.start_time if self.is_training else 0.0
            return {
                "is_training": self.is_training,
                "is_paused": self.is_paused,
                "current_episode": self.current_episode,
                "total_episodes": self.total_episodes,
                "progress_percent": round((self.current_episode / max(1, self.total_episodes)) * 100, 1),
                "algorithm": self.current_algorithm,
                "scenario": self.scenario_id,
                "status_message": self.status_message,
                "elapsed_seconds": round(elapsed, 1),
                "current_epsilon": getattr(self.agent, "epsilon", 0.0) if self.agent else 0.0,
                "history_length": len(self.history),
                "latest_metric": self.history[-1] if self.history else None,
            }

    def start_training(
        self,
        algorithm: str = "q_learning",
        episodes: int = 60,
        episode_duration: float = 200.0,
        scenario_id: str = "balanced",
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.992,
        seed: int = 42,
    ) -> bool:
        with self.lock:
            if self.is_training:
                return False

            self.current_algorithm = algorithm
            self.total_episodes = episodes
            self.episode_duration = episode_duration
            self.scenario_id = scenario_id
            self.current_episode = 0
            self.history = []
            self.stop_requested = False
            self.is_paused = False
            self.is_training = True
            self.status_message = "Training started..."
            self.start_time = time.time()

            # Initialize environment
            self.env = TrafficEnvironment(
                scenario_id=scenario_id,
                episode_duration=episode_duration,
                seed=seed,
            )

            # Initialize agent
            if algorithm == "q_learning":
                self.agent = QLearningAgent(
                    alpha=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    seed=seed,
                )
            elif algorithm == "dqn":
                self.agent = DQNAgent(
                    lr=alpha,
                    gamma=gamma,
                    epsilon=epsilon,
                    epsilon_min=epsilon_min,
                    epsilon_decay=epsilon_decay,
                    seed=seed,
                )
            else:
                self.is_training = False
                raise ValueError(f"Unknown algorithm: {algorithm}")

            self.training_thread = threading.Thread(target=self._run_training_loop, daemon=True)
            self.training_thread.start()
            return True

    def pause_training(self):
        with self.lock:
            if self.is_training and not self.is_paused:
                self.is_paused = True
                self.status_message = "Training paused"

    def resume_training(self):
        with self.lock:
            if self.is_training and self.is_paused:
                self.is_paused = False
                self.status_message = "Training resumed"

    def stop_training(self):
        with self.lock:
            if self.is_training:
                self.stop_requested = True
                self.status_message = "Stopping training..."

    def _run_training_loop(self):
        try:
            for ep in range(1, self.total_episodes + 1):
                # Check for stop
                if self.stop_requested:
                    break

                # Check for pause
                while self.is_paused and not self.stop_requested:
                    time.sleep(0.2)

                if self.stop_requested:
                    break

                # Seed variation across episodes for generalizability
                ep_seed = 1000 + ep * 7
                disc_state, cont_state = self.env.reset(seed=ep_seed)
                done = False
                ep_reward = 0.0

                while not done:
                    if self.stop_requested:
                        break

                    while self.is_paused and not self.stop_requested:
                        time.sleep(0.2)

                    if self.current_algorithm == "q_learning":
                        action = self.agent.select_action(disc_state, deterministic=False)
                        next_disc, next_cont, reward, done, info = self.env.step(action)
                        self.agent.update(disc_state, action, reward, next_disc, done)
                        disc_state = next_disc
                    else:  # DQN
                        action = self.agent.select_action(cont_state, deterministic=False)
                        next_disc, next_cont, reward, done, info = self.env.step(action)
                        self.agent.update(cont_state, action, reward, next_cont, done)
                        cont_state = next_cont

                    ep_reward += reward

                # End of episode decay
                if hasattr(self.agent, "decay_epsilon"):
                    self.agent.decay_epsilon()

                summary = self.env.get_summary_metrics()
                summary["episode"] = ep
                summary["epsilon"] = round(self.agent.epsilon, 4)
                summary["total_reward"] = round(ep_reward, 2)

                with self.lock:
                    self.current_episode = ep
                    self.history.append(summary)
                    self.status_message = f"Episode {ep}/{self.total_episodes} completed (Reward: {round(ep_reward, 1)})"

                if self.on_episode_end:
                    try:
                        self.on_episode_end(summary)
                    except Exception:
                        pass

            # Save checkpoint after training finishes
            if self.agent and self.current_episode > 0:
                save_name = f"{self.current_algorithm}_latest"
                save_path = os.path.join(
                    self.models_dir,
                    f"{save_name}.{'json' if self.current_algorithm == 'q_learning' else 'pt'}"
                )
                self.agent.save(save_path)

        except Exception as e:
            with self.lock:
                self.status_message = f"Training error: {str(e)}"
        finally:
            with self.lock:
                self.is_training = False
                self.is_paused = False
                if not self.stop_requested:
                    self.status_message = "Training completed successfully."
                else:
                    self.status_message = "Training stopped by user."
