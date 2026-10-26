"""
Deep Q-Network (DQN) implementation using PyTorch.
Includes online Q-network, target Q-network, experience replay, and Bellman mini-batch updates.
"""
from typing import Dict, Any, Optional
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.agents.base_agent import BaseAgent
from app.agents.replay_buffer import ReplayBuffer

class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 10, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DQNAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int = 10,
        action_dim: int = 2,
        hidden_dim: int = 64,
        lr: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        target_update_freq: int = 100,
        buffer_capacity: int = 20000,
        seed: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device("cpu")  # CPU is robust and fast for 10-dim state

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stability

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)
        self.train_step_count: int = 0
        self.latest_loss: float = 0.0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Epsilon-greedy action selection."""
        if not deterministic and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> Optional[float]:
        """
        Store transition in replay buffer and perform a mini-batch gradient step
        if enough samples exist.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample mini-batch
        b_states, b_actions, b_rewards, b_next_states, b_dones = self.replay_buffer.sample(self.batch_size)
        b_states = b_states.to(self.device)
        b_actions = b_actions.to(self.device)
        b_rewards = b_rewards.to(self.device)
        b_next_states = b_next_states.to(self.device)
        b_dones = b_dones.to(self.device)

        # Q(s, a) from policy net
        q_eval = self.policy_net(b_states).gather(1, b_actions)

        # Max Q(s', a') from target net
        with torch.no_grad():
            q_next = self.target_net(b_next_states).max(1)[0].unsqueeze(1)
            # Terminal mask: if done, target is just reward
            q_target = b_rewards + (1.0 - b_dones) * self.gamma * q_next

        loss = self.criterion(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_step_count += 1
        self.latest_loss = float(loss.item())

        # Periodically update target network
        if self.train_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.latest_loss

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, file_path: str):
        """Save PyTorch weights and configuration."""
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        checkpoint = {
            "algorithm": "dqn",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "train_step_count": self.train_step_count,
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
        }
        torch.save(checkpoint, file_path)

    def load(self, file_path: str):
        """Restore PyTorch weights and configuration."""
        checkpoint = torch.load(file_path, map_location=self.device)
        self.state_dim = checkpoint["state_dim"]
        self.action_dim = checkpoint["action_dim"]
        self.hidden_dim = checkpoint["hidden_dim"]
        self.gamma = checkpoint["gamma"]
        self.epsilon = checkpoint["epsilon"]
        self.epsilon_min = checkpoint["epsilon_min"]
        self.epsilon_decay = checkpoint["epsilon_decay"]
        self.train_step_count = checkpoint.get("train_step_count", 0)

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.target_net.eval()

    def get_summary(self) -> Dict[str, Any]:
        """Network summary for UI display."""
        param_count = sum(p.numel() for p in self.policy_net.parameters())
        return {
            "algorithm": "dqn",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "total_parameters": param_count,
            "train_step_count": self.train_step_count,
            "latest_loss": round(self.latest_loss, 4),
            "buffer_size": len(self.replay_buffer),
            "epsilon": round(self.epsilon, 4),
        }
