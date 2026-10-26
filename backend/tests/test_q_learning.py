"""
Unit tests for Tabular Q-Learning Agent.
"""
import pytest
import numpy as np
import tempfile
import os
from app.agents.q_learning import QLearningAgent

def test_q_table_initialization():
    agent = QLearningAgent(num_states=288, num_actions=2)
    assert agent.q_table.shape == (288, 2)
    assert np.all(agent.q_table == 0.0)

def test_q_learning_update_rule():
    agent = QLearningAgent(num_states=10, num_actions=2, alpha=0.5, gamma=0.9, epsilon=0.0)
    # Step 1: Update with non-terminal transition
    # Q(s=1, a=0) = 0 + 0.5 * (10.0 + 0.9 * max(Q(s=2)) - 0) = 5.0
    td_error = agent.update(state=1, action=0, reward=10.0, next_state=2, done=False)
    assert pytest.approx(agent.q_table[1, 0], 0.01) == 5.0
    assert pytest.approx(td_error, 0.01) == 10.0

def test_q_learning_terminal_state():
    agent = QLearningAgent(num_states=10, num_actions=2, alpha=1.0, gamma=0.9)
    # Set future value in next state
    agent.q_table[3, 0] = 50.0
    agent.q_table[3, 1] = 100.0

    # If done is True, target must ignore future Q(s', a')
    agent.update(state=1, action=1, reward=5.0, next_state=3, done=True)
    # With alpha=1.0, Q(s, a) should equal exactly reward (5.0)
    assert pytest.approx(agent.q_table[1, 1], 0.01) == 5.0

def test_epsilon_greedy_decay():
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == 0.5
    agent.decay_epsilon()
    assert agent.epsilon == 0.25
    agent.decay_epsilon()
    assert agent.epsilon == 0.125
    agent.decay_epsilon()
    assert agent.epsilon == 0.1  # clamped at epsilon_min

def test_save_and_load(tmp_path):
    agent = QLearningAgent(num_states=20, num_actions=2, alpha=0.2)
    agent.q_table[5, 1] = 42.5
    save_path = str(tmp_path / "test_q_table.json")

    agent.save(save_path)
    assert os.path.exists(save_path)

    loaded_agent = QLearningAgent(num_states=20, num_actions=2)
    loaded_agent.load(save_path)
    assert pytest.approx(loaded_agent.q_table[5, 1], 0.001) == 42.5
    assert loaded_agent.alpha == 0.2
