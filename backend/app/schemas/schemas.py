"""
Pydantic schemas for request validation and response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class TrainRequest(BaseModel):
    algorithm: str = Field(default="q_learning", description="Algorithm: 'q_learning' or 'dqn'")
    episodes: int = Field(default=50, ge=1, le=500, description="Total training episodes")
    episode_duration: float = Field(default=200.0, ge=50.0, le=1000.0, description="Seconds per episode")
    scenario_id: str = Field(default="balanced", description="Scenario identifier")
    alpha: float = Field(default=0.1, gt=0.0, le=1.0, description="Learning rate")
    gamma: float = Field(default=0.95, ge=0.0, le=1.0, description="Discount factor")
    epsilon: float = Field(default=1.0, ge=0.0, le=1.0, description="Initial exploration rate")
    epsilon_min: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum exploration rate")
    epsilon_decay: float = Field(default=0.992, ge=0.8, le=1.0, description="Epsilon decay rate")
    seed: int = Field(default=42, description="Random seed")

class EvaluateRequest(BaseModel):
    algorithm: str = Field(default="q_learning", description="'q_learning' or 'dqn'")
    scenario_id: str = Field(default="balanced", description="Evaluation traffic scenario")
    fixed_green: float = Field(default=30.0, ge=10.0, le=120.0, description="Baseline fixed green duration")
    num_episodes: int = Field(default=3, ge=1, le=20, description="Number of evaluation episodes")
    model_path: Optional[str] = Field(default=None, description="Path or name of saved model")

class SimulationConfigUpdate(BaseModel):
    min_green: Optional[float] = Field(default=None, ge=5.0, le=30.0)
    max_green: Optional[float] = Field(default=None, ge=20.0, le=120.0)
    yellow_duration: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    all_red_duration: Optional[float] = Field(default=None, ge=0.5, le=8.0)
    queue_weight: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    waiting_weight: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    switch_weight: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    throughput_weight: Optional[float] = Field(default=None, ge=0.0, le=10.0)

class SimulationResetRequest(BaseModel):
    scenario_id: str = Field(default="balanced")
    seed: Optional[int] = Field(default=None)

class SimulationStepRequest(BaseModel):
    action: int = Field(default=0, ge=0, le=1, description="0=KEEP, 1=SWITCH")
    controller_type: str = Field(default="manual", description="'manual', 'baseline', 'q_learning', 'dqn'")
