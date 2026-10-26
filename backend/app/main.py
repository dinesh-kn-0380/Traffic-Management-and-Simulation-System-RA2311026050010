"""
FastAPI application providing complete REST and WebSocket APIs
for Intelligent Traffic Signal Optimization Using Reinforcement Learning.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import csv
import os
import time
import asyncio
from typing import Dict, Any, Optional

from app.schemas.schemas import (
    TrainRequest,
    EvaluateRequest,
    SimulationConfigUpdate,
    SimulationResetRequest,
    SimulationStepRequest,
)
from app.simulation.scenarios import SCENARIOS
from app.simulation.simulation_manager import LiveSimulationManager
from app.training.trainer import TrainingManager
from app.training.evaluation import EvaluationEngine
from app.simulation.environment import TrafficEnvironment
from app.agents.q_learning import QLearningAgent
from app.agents.dqn import DQNAgent
from app.storage.database import (
    init_db,
    save_experiment,
    list_experiments,
    save_evaluation,
    list_evaluations,
)

# Initialize database
init_db()

app = FastAPI(
    title="Intelligent Traffic Signal Optimization API",
    description="Adaptive Traffic Management & Simulation System powered by Tabular Q-Learning and PyTorch Deep Q-Networks",
    version="1.0.0",
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Shared singletons
training_mgr = TrainingManager(models_dir=MODELS_DIR)
sim_mgr = LiveSimulationManager(models_dir=MODELS_DIR)


# -------------------------------------------------------------
# General & Health Endpoints
# -------------------------------------------------------------
@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "backend": "FastAPI",
        "models_available": os.path.exists(MODELS_DIR),
    }

@app.get("/api/algorithms")
def get_algorithms():
    return {
        "algorithms": [
            {
                "id": "q_learning",
                "name": "Tabular Q-Learning",
                "type": "Model-Free Value-Based RL",
                "description": "Learns state-action utility table Q(s,a) with exact Bellman temporal difference updates.",
                "state_type": "Discretized Tuple (288 states)",
                "action_type": "Discrete (0: Keep, 1: Switch)",
            },
            {
                "id": "dqn",
                "name": "Deep Q-Network (DQN)",
                "type": "Deep Reinforcement Learning (PyTorch)",
                "description": "Multi-layer perceptron with Experience Replay and Target Network predicting continuous Q-values.",
                "state_type": "Normalized 10-D Vector",
                "action_type": "Discrete (0: Keep, 1: Switch)",
            },
            {
                "id": "baseline",
                "name": "Fixed-Time Controller",
                "type": "Rule-Based Deterministic",
                "description": "Standard alternating signal cycle with configured green interval and safety clearances.",
                "state_type": "Clock / Elapsed Green",
                "action_type": "Periodic Switch",
            },
        ]
    }

@app.get("/api/scenarios")
def get_scenarios():
    return {
        "scenarios": [
            {
                "id": sc.id,
                "name": sc.name,
                "description": sc.description,
                "base_rates": sc.base_rates,
            }
            for sc in SCENARIOS.values()
        ]
    }


# -------------------------------------------------------------
# Configuration Endpoints
# -------------------------------------------------------------
@app.get("/api/config")
def get_config():
    env = sim_mgr.env
    return {
        "timing": {
            "min_green": env.min_green,
            "max_green": env.max_green,
            "yellow_duration": env.yellow_duration,
            "all_red_duration": env.all_red_duration,
        },
        "rewards": {
            "queue_weight": env.queue_weight,
            "waiting_weight": env.waiting_weight,
            "switch_weight": env.switch_weight,
            "throughput_weight": env.throughput_weight,
        },
        "geometry": {
            "road_length": env.road_length,
            "intersection_length": env.intersection_length,
            "departure_length": env.departure_length,
        }
    }

@app.post("/api/config")
def update_config(cfg: SimulationConfigUpdate):
    env = sim_mgr.env
    if cfg.min_green is not None:
        env.min_green = cfg.min_green
        env.signal_controller.min_green = cfg.min_green
    if cfg.max_green is not None:
        env.max_green = cfg.max_green
        env.signal_controller.max_green = cfg.max_green
    if cfg.yellow_duration is not None:
        env.yellow_duration = cfg.yellow_duration
        env.signal_controller.yellow_duration = cfg.yellow_duration
    if cfg.all_red_duration is not None:
        env.all_red_duration = cfg.all_red_duration
        env.signal_controller.all_red_duration = cfg.all_red_duration
    if cfg.queue_weight is not None:
        env.queue_weight = cfg.queue_weight
    if cfg.waiting_weight is not None:
        env.waiting_weight = cfg.waiting_weight
    if cfg.switch_weight is not None:
        env.switch_weight = cfg.switch_weight
    if cfg.throughput_weight is not None:
        env.throughput_weight = cfg.throughput_weight

    return {"status": "updated", "config": get_config()}


# -------------------------------------------------------------
# Training Control Endpoints
# -------------------------------------------------------------
@app.post("/api/train/start")
def start_training(req: TrainRequest):
    success = training_mgr.start_training(
        algorithm=req.algorithm,
        episodes=req.episodes,
        episode_duration=req.episode_duration,
        scenario_id=req.scenario_id,
        alpha=req.alpha,
        gamma=req.gamma,
        epsilon=req.epsilon,
        epsilon_min=req.epsilon_min,
        epsilon_decay=req.epsilon_decay,
        seed=req.seed,
    )
    if not success:
        raise HTTPException(status_code=400, detail="Training is already running")
    return {"status": "started", "training": training_mgr.get_status()}

@app.get("/api/train/status")
def get_training_status():
    status = training_mgr.get_status()
    # If training recently completed and has history not yet saved, save it to DB
    if not status["is_training"] and status["history_length"] > 0 and not getattr(training_mgr, "_saved_to_db", False):
        try:
            latest = status["latest_metric"] or {}
            exp_id = f"EXP_{int(time.time())}"
            save_experiment(
                experiment_id=exp_id,
                name=f"{status['algorithm'].upper()} on {status['scenario']}",
                algorithm=status["algorithm"],
                scenario=status["scenario"],
                episodes=status["total_episodes"],
                seed=42,
                hyperparameters={
                    "episodes": status["total_episodes"],
                    "algorithm": status["algorithm"],
                },
                training_duration=status["elapsed_seconds"],
                mean_reward=latest.get("total_reward", 0.0),
                mean_waiting_time=latest.get("mean_waiting_time", 0.0),
                mean_queue_length=latest.get("mean_queue_length", 0.0),
                throughput=latest.get("total_departures", 0),
                switch_frequency=latest.get("total_switches", 0),
                checkpoint_path=f"models/{status['algorithm']}_latest.json",
            )
            training_mgr._saved_to_db = True
        except Exception:
            pass

    return status

@app.get("/api/train/history")
def get_training_history():
    return {"history": training_mgr.history}

@app.post("/api/train/pause")
def pause_training():
    training_mgr.pause_training()
    return {"status": "paused", "training": training_mgr.get_status()}

@app.post("/api/train/resume")
def resume_training():
    training_mgr.resume_training()
    return {"status": "resumed", "training": training_mgr.get_status()}

@app.post("/api/train/stop")
def stop_training():
    training_mgr.stop_training()
    return {"status": "stopping", "training": training_mgr.get_status()}


# -------------------------------------------------------------
# Model Management & Laboratory Endpoints
# -------------------------------------------------------------
@app.get("/api/models")
def list_models():
    models = []
    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            fpath = os.path.join(MODELS_DIR, f)
            if f.endswith(".json") or f.endswith(".pt"):
                stat = os.stat(fpath)
                models.append({
                    "filename": f,
                    "path": fpath,
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                    "algorithm": "q_learning" if f.endswith(".json") else "dqn",
                })
    return {"models": models}

@app.get("/api/models/q_table")
def get_q_table():
    """Returns decoded Q-table entries for interactive exploration in Algorithm Lab."""
    # Check if training agent has Q-table or load latest
    agent = None
    if training_mgr.agent and isinstance(training_mgr.agent, QLearningAgent):
        agent = training_mgr.agent
    elif sim_mgr.q_agent:
        agent = sim_mgr.q_agent
    else:
        q_path = os.path.join(MODELS_DIR, "q_learning_latest.json")
        if os.path.exists(q_path):
            agent = QLearningAgent()
            agent.load(q_path)

    if agent is None:
        # Return initialized empty representation
        agent = QLearningAgent()

    return agent.get_q_table_summary(max_entries=60)


# -------------------------------------------------------------
# Evaluation & Benchmarking Endpoints
# -------------------------------------------------------------
@app.post("/api/evaluate")
def run_evaluation(req: EvaluateRequest):
    env = TrafficEnvironment(scenario_id=req.scenario_id, episode_duration=200.0)
    engine = EvaluationEngine(env=env)

    agent = None
    if req.algorithm == "q_learning":
        agent = QLearningAgent()
        model_path = req.model_path or os.path.join(MODELS_DIR, "q_learning_latest.json")
        if os.path.exists(model_path):
            agent.load(model_path)
    elif req.algorithm == "dqn":
        agent = DQNAgent()
        model_path = req.model_path or os.path.join(MODELS_DIR, "dqn_latest.pt")
        if os.path.exists(model_path):
            agent.load(model_path)

    comparison = engine.compare(
        agent=agent,
        algorithm=req.algorithm,
        fixed_green=req.fixed_green,
        num_episodes=req.num_episodes,
    )

    # Save evaluation run to database
    eval_id = f"EVAL_{int(time.time())}"
    save_evaluation(
        evaluation_id=eval_id,
        algorithm=req.algorithm,
        scenario=req.scenario_id,
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

    return comparison

@app.get("/api/evaluations")
def get_evaluations_history():
    return {"evaluations": list_evaluations()}


# -------------------------------------------------------------
# Experiment Management & Export Endpoints
# -------------------------------------------------------------
@app.get("/api/experiments")
def get_experiments():
    return {"experiments": list_experiments()}

@app.get("/api/experiments/export")
def export_experiments_csv():
    experiments = list_experiments()
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "ID", "Experiment ID", "Name", "Algorithm", "Scenario",
        "Episodes", "Seed", "Duration (s)", "Mean Reward",
        "Mean Waiting Time (s)", "Mean Queue Length", "Throughput",
        "Switch Frequency", "Created At"
    ])

    for exp in experiments:
        writer.writerow([
            exp["id"], exp["experiment_id"], exp["name"], exp["algorithm"],
            exp["scenario"], exp["episodes"], exp["seed"], exp["training_duration"],
            exp["mean_reward"], exp["mean_waiting_time"], exp["mean_queue_length"],
            exp["throughput"], exp["switch_frequency"], exp["created_at"]
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=traffic_rl_experiments.csv"}
    )


# -------------------------------------------------------------
# Live Simulation Control Endpoints
# -------------------------------------------------------------
@app.get("/api/simulation/state")
def get_simulation_state():
    return sim_mgr.get_state()

@app.post("/api/simulation/start")
def start_simulation():
    sim_mgr.start()
    return {"status": "started", "state": sim_mgr.get_state()}

@app.post("/api/simulation/pause")
def pause_simulation():
    sim_mgr.pause()
    return {"status": "paused", "state": sim_mgr.get_state()}

@app.post("/api/simulation/step")
def step_simulation(req: SimulationStepRequest):
    state = sim_mgr.step(manual_action=req.action)
    return {"status": "stepped", "state": state}

@app.post("/api/simulation/reset")
def reset_simulation(req: SimulationResetRequest):
    state = sim_mgr.reset(scenario_id=req.scenario_id, seed=req.seed)
    return {"status": "reset", "state": state}

@app.post("/api/simulation/controller")
def set_simulation_controller(controller_type: str = Query(...)):
    sim_mgr.set_controller(controller_type)
    return {"status": "controller_set", "controller": controller_type}

@app.post("/api/simulation/speed")
def set_simulation_speed(speed: float = Query(...)):
    sim_mgr.set_speed(speed)
    return {"status": "speed_set", "speed": speed}


# -------------------------------------------------------------
# Live WebSocket for Real-time Simulation Updates
# -------------------------------------------------------------
@app.websocket("/ws/simulation")
async def websocket_simulation(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Check for incoming client messages (e.g. action or ping)
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=0.01)
                cmd = data.get("command")
                if cmd == "start":
                    sim_mgr.start()
                elif cmd == "pause":
                    sim_mgr.pause()
                elif cmd == "step":
                    action = data.get("action", 0)
                    sim_mgr.step(manual_action=action)
                elif cmd == "reset":
                    sim_mgr.reset(scenario_id=data.get("scenario_id"))
                elif cmd == "controller":
                    sim_mgr.set_controller(data.get("controller_type", "baseline"))
                elif cmd == "speed":
                    sim_mgr.set_speed(float(data.get("speed", 1.0)))
            except asyncio.TimeoutError:
                pass

            # Broadcast current simulation state
            state = sim_mgr.get_state()
            await websocket.send_json(state)
            await asyncio.sleep(0.1)  # 10 Hz broadcast
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
