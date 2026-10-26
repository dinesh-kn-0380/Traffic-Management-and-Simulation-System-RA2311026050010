"""
Script to train initial Q-Learning and DQN agents and populate benchmark data.
"""
import os
import sys
import time

# Ensure backend is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.simulation.environment import TrafficEnvironment
from app.agents.q_learning import QLearningAgent
from app.agents.dqn import DQNAgent
from app.training.evaluation import EvaluationEngine
from app.storage.database import init_db, save_experiment, save_evaluation

def train_and_seed():
    print("Initializing Database...")
    init_db()

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Train Q-Learning Agent
    print("\n[1/3] Training Tabular Q-Learning Agent (40 episodes)...")
    env = TrafficEnvironment(scenario_id="heavy_north_south", episode_duration=150.0, seed=42)
    q_agent = QLearningAgent(alpha=0.15, gamma=0.95, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.985, seed=42)

    start_time = time.time()
    for ep in range(1, 41):
        disc_state, cont_state = env.reset(seed=100 + ep)
        done = False
        ep_reward = 0.0

        while not done:
            action = q_agent.select_action(disc_state, deterministic=False)
            next_disc, next_cont, reward, done, info = env.step(action)
            q_agent.update(disc_state, action, reward, next_disc, done)
            disc_state = next_disc
            ep_reward += reward

        q_agent.decay_epsilon()
        if ep % 10 == 0:
            print(f"  Episode {ep}/40 - Reward: {round(ep_reward, 1)} - Epsilon: {round(q_agent.epsilon, 3)}")

    q_duration = time.time() - start_time
    q_path = os.path.join(models_dir, "q_learning_latest.json")
    q_agent.save(q_path)
    print(f"Q-table saved to {q_path}")

    # 2. Evaluate Q-Learning against Baseline
    print("\n[2/3] Evaluating Q-Learning vs Fixed-Time Baseline...")
    eval_engine = EvaluationEngine(env)
    comparison = eval_engine.compare(agent=q_agent, algorithm="q_learning", fixed_green=30.0, num_episodes=3)

    print("  Baseline Mean Waiting Time:", comparison["baseline"]["mean_waiting_time"], "s")
    print("  RL Agent Mean Waiting Time:", comparison["rl"]["mean_waiting_time"], "s")
    print("  Waiting Time Improvement:", comparison["improvements"]["waiting_time_percent"], "%")
    print("  Queue Length Improvement:", comparison["improvements"]["queue_length_percent"], "%")
    print("  Throughput Improvement:", comparison["improvements"]["throughput_percent"], "%")

    # Record in SQLite
    save_experiment(
        experiment_id="EXP_QL_HEAVY_NS",
        name="Tabular Q-Learning (Heavy North-South)",
        algorithm="q_learning",
        scenario="heavy_north_south",
        episodes=40,
        seed=42,
        hyperparameters={"alpha": 0.15, "gamma": 0.95, "epsilon_decay": 0.985},
        training_duration=q_duration,
        mean_reward=comparison["rl"]["mean_reward"],
        mean_waiting_time=comparison["rl"]["mean_waiting_time"],
        mean_queue_length=comparison["rl"]["mean_queue_length"],
        throughput=int(comparison["rl"]["mean_throughput"]),
        switch_frequency=comparison["rl"]["mean_switches"],
        checkpoint_path="models/q_learning_latest.json",
    )

    save_evaluation(
        evaluation_id="EVAL_QL_BENCHMARK",
        algorithm="q_learning",
        scenario="heavy_north_south",
        baseline_wait=comparison["baseline"]["mean_waiting_time"],
        rl_wait=comparison["rl"]["mean_waiting_time"],
        wait_improvement_pct=comparison["improvements"]["waiting_time_percent"],
        baseline_queue=comparison["baseline"]["mean_queue_length"],
        rl_queue=comparison["rl"]["mean_queue_length"],
        queue_improvement_pct=comparison["improvements"]["queue_length_percent"],
        baseline_throughput=comparison["baseline"]["mean_throughput"],
        rl_throughput=comparison["rl"]["mean_throughput"],
        throughput_improvement_pct=comparison["improvements"]["throughput_percent"],
        details=comparison,
    )

    # 3. Quick DQN Agent initialization & checkpoint
    print("\n[3/3] Initializing Deep Q-Network Agent...")
    dqn_agent = DQNAgent(state_dim=10, action_dim=2, seed=42)
    # Quick train for 15 episodes so network weights are trained
    dqn_env = TrafficEnvironment(scenario_id="balanced", episode_duration=100.0, seed=42)
    start_dqn = time.time()
    for ep in range(1, 16):
        disc_state, cont_state = dqn_env.reset(seed=200 + ep)
        done = False
        while not done:
            action = dqn_agent.select_action(cont_state, deterministic=False)
            next_disc, next_cont, reward, done, info = dqn_env.step(action)
            dqn_agent.update(cont_state, action, reward, next_cont, done)
            cont_state = next_cont
        dqn_agent.decay_epsilon()

    dqn_duration = time.time() - start_dqn
    dqn_path = os.path.join(models_dir, "dqn_latest.pt")
    dqn_agent.save(dqn_path)
    print(f"DQN weights saved to {dqn_path}")

    dqn_comp = eval_engine.compare(agent=dqn_agent, algorithm="dqn", fixed_green=30.0, num_episodes=2)
    save_experiment(
        experiment_id="EXP_DQN_BALANCED",
        name="Deep Q-Network (Balanced Demand)",
        algorithm="dqn",
        scenario="balanced",
        episodes=15,
        seed=42,
        hyperparameters={"lr": 0.001, "hidden_dim": 64, "batch_size": 64},
        training_duration=dqn_duration,
        mean_reward=dqn_comp["rl"]["mean_reward"],
        mean_waiting_time=dqn_comp["rl"]["mean_waiting_time"],
        mean_queue_length=dqn_comp["rl"]["mean_queue_length"],
        throughput=int(dqn_comp["rl"]["mean_throughput"]),
        switch_frequency=dqn_comp["rl"]["mean_switches"],
        checkpoint_path="models/dqn_latest.pt",
    )

    print("\nInitial model training and benchmark seeding complete!")

if __name__ == "__main__":
    train_and_seed()
