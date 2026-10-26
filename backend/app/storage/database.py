"""
SQLite database management for experiment logs, evaluation results, and model tracking.
"""
import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "traffic_rl.db")

def get_db_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id TEXT UNIQUE NOT NULL,
        name TEXT,
        algorithm TEXT NOT NULL,
        scenario TEXT NOT NULL,
        episodes INTEGER NOT NULL,
        seed INTEGER,
        hyperparameters TEXT,
        training_duration REAL,
        mean_reward REAL,
        mean_waiting_time REAL,
        mean_queue_length REAL,
        throughput INTEGER,
        switch_frequency REAL,
        checkpoint_path TEXT,
        created_at TEXT NOT NULL
    );
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_id TEXT UNIQUE NOT NULL,
        algorithm TEXT NOT NULL,
        scenario TEXT NOT NULL,
        baseline_wait REAL,
        rl_wait REAL,
        wait_improvement_pct REAL,
        baseline_queue REAL,
        rl_queue REAL,
        queue_improvement_pct REAL,
        baseline_throughput REAL,
        rl_throughput REAL,
        throughput_improvement_pct REAL,
        details_json TEXT,
        created_at TEXT NOT NULL
    );
    """)

    conn.commit()
    conn.close()

def save_experiment(
    experiment_id: str,
    name: str,
    algorithm: str,
    scenario: str,
    episodes: int,
    seed: Optional[int],
    hyperparameters: Dict[str, Any],
    training_duration: float,
    mean_reward: float,
    mean_waiting_time: float,
    mean_queue_length: float,
    throughput: int,
    switch_frequency: float,
    checkpoint_path: Optional[str] = None
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO experiments (
        experiment_id, name, algorithm, scenario, episodes, seed,
        hyperparameters, training_duration, mean_reward, mean_waiting_time,
        mean_queue_length, throughput, switch_frequency, checkpoint_path, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        experiment_id,
        name,
        algorithm,
        scenario,
        episodes,
        seed,
        json.dumps(hyperparameters),
        round(training_duration, 2),
        round(mean_reward, 2),
        round(mean_waiting_time, 2),
        round(mean_queue_length, 2),
        throughput,
        round(switch_frequency, 2),
        checkpoint_path,
        created_at
    ))

    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

def list_experiments() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM experiments ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "experiment_id": r["experiment_id"],
            "name": r["name"],
            "algorithm": r["algorithm"],
            "scenario": r["scenario"],
            "episodes": r["episodes"],
            "seed": r["seed"],
            "hyperparameters": json.loads(r["hyperparameters"]) if r["hyperparameters"] else {},
            "training_duration": r["training_duration"],
            "mean_reward": r["mean_reward"],
            "mean_waiting_time": r["mean_waiting_time"],
            "mean_queue_length": r["mean_queue_length"],
            "throughput": r["throughput"],
            "switch_frequency": r["switch_frequency"],
            "checkpoint_path": r["checkpoint_path"],
            "created_at": r["created_at"],
        })
    return results

def save_evaluation(
    evaluation_id: str,
    algorithm: str,
    scenario: str,
    baseline_wait: float,
    rl_wait: float,
    wait_improvement_pct: float,
    baseline_queue: float,
    rl_queue: float,
    queue_improvement_pct: float,
    baseline_throughput: float,
    rl_throughput: float,
    throughput_improvement_pct: float,
    details: Dict[str, Any],
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO evaluations (
        evaluation_id, algorithm, scenario, baseline_wait, rl_wait, wait_improvement_pct,
        baseline_queue, rl_queue, queue_improvement_pct, baseline_throughput, rl_throughput,
        throughput_improvement_pct, details_json, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        evaluation_id,
        algorithm,
        scenario,
        round(baseline_wait, 2),
        round(rl_wait, 2),
        round(wait_improvement_pct, 2),
        round(baseline_queue, 2),
        round(rl_queue, 2),
        round(queue_improvement_pct, 2),
        round(baseline_throughput, 2),
        round(rl_throughput, 2),
        round(throughput_improvement_pct, 2),
        json.dumps(details),
        created_at
    ))

    conn.commit()
    last_id = cursor.lastrowid
    conn.close()
    return last_id

def list_evaluations() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM evaluations ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "evaluation_id": r["evaluation_id"],
            "algorithm": r["algorithm"],
            "scenario": r["scenario"],
            "baseline_wait": r["baseline_wait"],
            "rl_wait": r["rl_wait"],
            "wait_improvement_pct": r["wait_improvement_pct"],
            "baseline_queue": r["baseline_queue"],
            "rl_queue": r["rl_queue"],
            "queue_improvement_pct": r["queue_improvement_pct"],
            "baseline_throughput": r["baseline_throughput"],
            "rl_throughput": r["rl_throughput"],
            "throughput_improvement_pct": r["throughput_improvement_pct"],
            "details": json.loads(r["details_json"]) if r["details_json"] else {},
            "created_at": r["created_at"],
        })
    return results
