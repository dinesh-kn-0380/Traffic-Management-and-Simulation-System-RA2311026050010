"""
Tabular Q-Learning Agent implemented from scratch.
"""
from typing import Dict, Any, List, Optional
import numpy as np
import json
import os
from app.agents.base_agent import BaseAgent

class QLearningAgent(BaseAgent):
    def __init__(
        self,
        num_states: int = 288,
        num_actions: int = 2,
        alpha: float = 0.10,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: Optional[int] = None,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.RandomState(seed)

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float64)
        self.training_steps: int = 0

    def select_action(self, state: int, deterministic: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        If deterministic is True, purely exploits max Q(s, a).
        Breaks ties randomly among maximizing actions.
        """
        if not deterministic and self.rng.rand() < self.epsilon:
            return int(self.rng.choice(self.num_actions))

        # Argmax with random tie-breaking
        q_vals = self.q_table[state]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ) -> float:
        """
        Bellman temporal difference update:
        Q(s, a) = Q(s, a) + alpha * [reward + (1 - done) * gamma * max_a' Q(s', a') - Q(s, a)]
        """
        old_val = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - old_val
        self.q_table[state, action] = old_val + self.alpha * td_error
        self.training_steps += 1
        return float(abs(td_error))

    def decay_epsilon(self):
        """Decay exploration rate after an episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, file_path: str):
        """Save Q-table and agent configuration to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        payload = {
            "algorithm": "q_learning",
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "training_steps": self.training_steps,
            "q_table": self.q_table.tolist(),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load(self, file_path: str):
        """Load Q-table and agent configuration from JSON."""
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.num_states = payload["num_states"]
        self.num_actions = payload["num_actions"]
        self.alpha = payload["alpha"]
        self.gamma = payload["gamma"]
        self.epsilon = payload["epsilon"]
        self.epsilon_min = payload["epsilon_min"]
        self.epsilon_decay = payload["epsilon_decay"]
        self.training_steps = payload.get("training_steps", 0)
        self.q_table = np.array(payload["q_table"], dtype=np.float64)

    def get_q_table_summary(self, max_entries: int = 50) -> Dict[str, Any]:
        """Generate human-readable summary of learned Q-values for UI."""
        non_zero_count = int(np.count_nonzero(self.q_table))
        entries = []

        # Find states with highest absolute Q-values or most active
        row_magnitudes = np.max(np.abs(self.q_table), axis=1)
        top_state_indices = np.argsort(row_magnitudes)[::-1][:max_entries]

        for s_idx in top_state_indices:
            q_keep = float(self.q_table[s_idx, 0])
            q_switch = float(self.q_table[s_idx, 1])

            # Decode state index into human readable variables:
            # index = phase*(4*4*3*3) + ns_q*(4*3*3) + ew_q*(3*3) + gt*(3) + wait_bin
            rem = s_idx
            phase = rem // (4 * 4 * 3 * 3)
            rem %= (4 * 4 * 3 * 3)
            ns_q = rem // (4 * 3 * 3)
            rem %= (4 * 3 * 3)
            ew_q = rem // (3 * 3)
            rem %= (3 * 3)
            gt = rem // 3
            wait_bin = rem % 3

            phase_label = "North-South Green" if phase == 0 else "East-West Green"
            queue_labels = ["0-2 veh", "3-6 veh", "7-11 veh", "12+ veh"]
            gt_labels = ["<15s", "15-30s", ">30s"]
            wait_labels = ["NS Urgency", "EW Urgency", "Balanced"]

            entries.append({
                "state_id": int(s_idx),
                "phase": phase_label,
                "ns_queue": queue_labels[ns_q],
                "ew_queue": queue_labels[ew_q],
                "green_time": gt_labels[gt],
                "wait_ratio": wait_labels[wait_bin],
                "q_keep": round(q_keep, 3),
                "q_switch": round(q_switch, 3),
                "optimal_action": "KEEP" if q_keep >= q_switch else "SWITCH",
                "advantage": round(abs(q_keep - q_switch), 3)
            })

        return {
            "total_states": self.num_states,
            "non_zero_states": non_zero_count,
            "mean_q_val": round(float(np.mean(self.q_table)), 4),
            "max_q_val": round(float(np.max(self.q_table)), 4),
            "min_q_val": round(float(np.min(self.q_table)), 4),
            "entries": entries,
        }
