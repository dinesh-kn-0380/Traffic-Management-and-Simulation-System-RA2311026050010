export interface Vehicle {
  id: number;
  approach: 'N' | 'S' | 'E' | 'W';
  lane: number;
  position: number;
  speed: number;
  waiting_time: number;
  is_queued: boolean;
  has_departed: boolean;
  color: string;
  length: number;
}

export interface SimulationState {
  sim_time: number;
  current_phase: number;
  phase_name: string;
  phase_elapsed: number;
  is_transitioning: boolean;
  signals: Record<'N' | 'S' | 'E' | 'W', 'GREEN' | 'YELLOW' | 'RED'>;
  queues: Record<'N' | 'S' | 'E' | 'W', number>;
  total_queue: number;
  aggregate_waits: Record<'N' | 'S' | 'E' | 'W', number>;
  vehicles: Vehicle[];
  completed_count: number;
  switch_count: number;
  cumulative_reward: number;
  controller_type?: string;
  is_running?: boolean;
  speed_multiplier?: number;
  geometry: {
    stop_line_pos: number;
    intersection_length: number;
    road_end_pos: number;
  };
}

export interface Scenario {
  id: string;
  name: string;
  description: string;
  base_rates: Record<string, number>;
}

export interface AlgorithmInfo {
  id: string;
  name: string;
  type: string;
  description: string;
  state_type: string;
  action_type: string;
}

export interface TrainingStatus {
  is_training: boolean;
  is_paused: boolean;
  current_episode: number;
  total_episodes: number;
  progress_percent: number;
  algorithm: string;
  scenario: string;
  status_message: string;
  elapsed_seconds: number;
  current_epsilon: number;
  history_length: number;
  latest_metric?: {
    episode: number;
    total_reward: number;
    mean_waiting_time: number;
    mean_queue_length: number;
    total_departures: number;
    total_switches: number;
    epsilon: number;
  };
}

export interface TrainingMetric {
  episode: number;
  total_reward: number;
  mean_waiting_time: number;
  mean_queue_length: number;
  max_queue_length: number;
  total_departures: number;
  total_switches: number;
  epsilon: number;
}

export interface Experiment {
  id: number;
  experiment_id: string;
  name: string;
  algorithm: string;
  scenario: string;
  episodes: number;
  seed: number;
  hyperparameters: Record<string, any>;
  training_duration: number;
  mean_reward: number;
  mean_waiting_time: number;
  mean_queue_length: number;
  throughput: number;
  switch_frequency: number;
  checkpoint_path: string;
  created_at: string;
}

export interface EvaluationComparison {
  scenario: string;
  baseline: {
    controller: string;
    mean_waiting_time: number;
    std_waiting_time: number;
    mean_queue_length: number;
    std_queue_length: number;
    max_queue_length: number;
    mean_throughput: number;
    std_throughput: number;
    mean_switches: number;
    mean_reward: number;
    episodes: any[];
  };
  rl: {
    controller: string;
    mean_waiting_time: number;
    std_waiting_time: number;
    mean_queue_length: number;
    std_queue_length: number;
    max_queue_length: number;
    mean_throughput: number;
    std_throughput: number;
    mean_switches: number;
    mean_reward: number;
    episodes: any[];
  };
  improvements: {
    waiting_time_percent: number;
    queue_length_percent: number;
    max_queue_percent: number;
    throughput_percent: number;
  };
  verdict: string;
}

export interface QTableSummary {
  total_states: number;
  non_zero_states: number;
  mean_q_val: number;
  max_q_val: number;
  min_q_val: number;
  entries: Array<{
    state_id: number;
    phase: string;
    ns_queue: string;
    ew_queue: string;
    green_time: string;
    wait_ratio: string;
    q_keep: number;
    q_switch: number;
    optimal_action: string;
    advantage: number;
  }>;
}

export interface AppConfig {
  timing: {
    min_green: number;
    max_green: number;
    yellow_duration: number;
    all_red_duration: number;
  };
  rewards: {
    queue_weight: number;
    waiting_weight: number;
    switch_weight: number;
    throughput_weight: number;
  };
  geometry: {
    road_length: number;
    intersection_length: number;
    departure_length: number;
  };
}
