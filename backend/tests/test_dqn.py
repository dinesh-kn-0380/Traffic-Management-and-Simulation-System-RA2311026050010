"""
Unit tests for Deep Q-Network and Experience Replay Buffer.
"""
import pytest
import numpy as np
import torch
import os
from app.agents.dqn import DQNAgent, QNetwork
from app.agents.replay_buffer import ReplayBuffer

def test_q_network_dimensions():
    net = QNetwork(state_dim=10, action_dim=2, hidden_dim=32)
    sample_input = torch.randn(4, 10)  # batch of 4
    output = net(sample_input)
    assert output.shape == (4, 2)

def test_replay_buffer_push_sample():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        s = np.zeros(10)
        s_next = np.ones(10)
        buf.push(s, 1, float(i), s_next, False)

    assert len(buf) == 10
    b_s, b_a, b_r, b_sn, b_d = buf.sample(batch_size=5)
    assert b_s.shape == (5, 10)
    assert b_a.shape == (5, 1)
    assert b_r.shape == (5, 1)
    assert b_sn.shape == (5, 10)
    assert b_d.shape == (5, 1)

def test_dqn_learning_step():
    agent = DQNAgent(state_dim=10, action_dim=2, batch_size=4, buffer_capacity=100)
    for i in range(10):
        s = np.random.randn(10).astype(np.float32)
        s_next = np.random.randn(10).astype(np.float32)
        loss = agent.update(s, 0, 1.0, s_next, False)

    assert agent.train_step_count > 0
    assert isinstance(agent.latest_loss, float)

def test_dqn_save_load(tmp_path):
    agent = DQNAgent(state_dim=10, action_dim=2)
    save_path = str(tmp_path / "test_dqn.pt")
    agent.save(save_path)
    assert os.path.exists(save_path)

    loaded_agent = DQNAgent(state_dim=10, action_dim=2)
    loaded_agent.load(save_path)
    assert loaded_agent.state_dim == 10
