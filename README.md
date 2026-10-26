# Intelligent Traffic Signal Optimization Using Reinforcement Learning
**An Adaptive Traffic Management and Simulation System**

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/RL-PyTorch%20%7C%20Tabular%20Q--Learning-EE4C2C.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript%20%2B%20Vite-61DAFB.svg)](https://vitejs.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Styling-Tailwind%20CSS-38B2AC.svg)](https://tailwindcss.com/)
[![Tests](https://img.shields.io/badge/Tests-25%20Passing-brightgreen.svg)]()

---

## 1. Project Overview & Motivation

Conventional traffic signal management systems rely on rigid, fixed-time schedules that cannot adapt to asymmetrical, dynamic urban vehicle flows. This results in unnecessary idling, increased fuel consumption, elevated emissions, and prolonged commuter delays.

This project implements an end-to-end, academic and production-ready **Reinforcement Learning (RL) Adaptive Traffic Management System**. It features a custom microscopic four-way intersection simulator, genuine **Tabular Q-Learning from scratch**, a **Deep Q-Network (DQN) with PyTorch**, strict signal safety interlocks (yellow clearance, all-red buffer, and starvation prevention), persistent SQLite experiment logging, automated pytest suites, and a modern **React + TypeScript + Tailwind CSS** dashboard.

---

## 2. Key Features

- **Microscopic 4-Way Traffic Simulation**: Individual vehicles with car-following dynamics, driver deceleration, stop lines, dilemma zones, and intersection clearance.
- **Strict Real-World Safety Constraints**:
  - Yellow clearance interval (3.0s) & All-red buffer (2.0s).
  - Minimum green time (10.0s) & Maximum green time (60.0s anti-starvation forced switch).
  - Physical invariant preventing conflicting green signals.
- **Genuine Reinforcement Learning Implementations**:
  - **Tabular Q-Learning (from scratch)**: 288 discretized states, exact Bellman TD updates, $\epsilon$-greedy decaying exploration.
  - **Deep Q-Network (PyTorch)**: 10-dimensional continuous state vector, Experience Replay buffer, target network synchronization, Huber loss.
- **Fixed-Time Baseline Comparison**: Head-to-head benchmarking against standard fixed-cycle controllers under identical random seeds.
- **7 Realistic Traffic Scenarios**: Balanced demand, Heavy North-South, Heavy East-West, Morning Rush, Evening Rush, Stochastic/Dynamic fluctuations, and Sudden Surges.
- **Full-Stack Educational Dashboard**:
  1. **Overview Dashboard**: System KPIs, interactive RL closed-loop architecture, quick actions.
  2. **Live Simulation**: 2D microscopic intersection canvas, moving cars, glowing signal bulbs, real-time controls (play, pause, step, speed 0.5x–5x).
  3. **Agent Training Studio**: Background non-blocking training, live Recharts curves for episode reward, waiting time, and queue lengths.
  4. **Algorithm Laboratory**: Mathematical derivations, interactive Q-table inspector with advantage values, DQN network diagram.
  5. **Performance Comparison**: Head-to-head benchmark runner with percentage improvements and comparative bar charts.
  6. **Experiment History**: SQLite persistent audit logs with CSV export.
  7. **Settings**: Configurable signal timing constraints and multi-objective reward weights.

---

## 3. System Architecture

```
Traffic Management and Simulation System/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI REST & WebSocket application
│   │   ├── schemas/schemas.py          # Pydantic request/response schemas
│   │   ├── simulation/
│   │   │   ├── vehicle.py              # Microscopic car-following kinematics
│   │   │   ├── traffic_generator.py    # Poisson vehicle arrival generator
│   │   │   ├── scenarios.py            # 7 predefined traffic demand scenarios
│   │   │   ├── signal_controller.py    # Safety interlocks, clearance intervals
│   │   │   ├── environment.py          # Gym-like intersection environment
│   │   │   └── simulation_manager.py   # Live simulation controller & session
│   │   ├── agents/
│   │   │   ├── base_agent.py           # Abstract agent interface
│   │   │   ├── q_learning.py           # Tabular Q-Learning from scratch
│   │   │   ├── replay_buffer.py        # Experience Replay for DQN
│   │   │   └── dqn.py                  # PyTorch Deep Q-Network agent
│   │   ├── training/
│   │   │   ├── baseline.py             # Fixed-Time Baseline controller
│   │   │   ├── trainer.py              # Background non-blocking training engine
│   │   │   └── evaluation.py           # Comparative benchmark engine
│   │   └── storage/
│   │       └── database.py             # SQLite persistence for experiments
│   ├── tests/                          # 25 automated unit tests
│   ├── models/                         # Saved Q-tables (.json) and weights (.pt)
│   ├── traffic_rl.db                   # SQLite database
│   ├── requirements.txt                # Python dependencies
│   └── train_initial_models.py         # Seed script for initial checkpoints
├── frontend/
│   ├── src/
│   │   ├── components/                 # Navbar, IntersectionCanvas
│   │   ├── pages/                      # Overview, Simulation, Training, Lab, etc.
│   │   ├── services/api.ts             # REST & WebSocket API client
│   │   ├── types/index.ts              # TypeScript interfaces
│   │   ├── App.tsx                     # Main layout and tab routing
│   │   └── index.css                   # Tailwind CSS v4 & custom glassmorphism
│   ├── vite.config.ts                  # Vite config with backend proxy
│   └── package.json                    # Frontend dependencies
├── docs/
│   ├── PROJECT_REPORT.md               # Formal academic report & derivations
│   └── VIVA_QUESTIONS.md               # Viva & interview Q&A guide
├── pytest.ini                          # Pytest configuration
└── README.md                           # Documentation root
```

---

## 4. Installation & Quickstart

### Prerequisites
- Python 3.11+ (or Python 3.13)
- Node.js v18+ & npm

### Backend Setup
1. Open a terminal in the project root:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```
2. Train initial models and seed the SQLite database:
   ```bash
   python train_initial_models.py
   ```
3. Start the FastAPI backend server:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```
   *The interactive Swagger API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).*

### Frontend Setup
1. In a second terminal, navigate to `frontend`:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
2. Open your browser at **[http://localhost:5173](http://localhost:5173)**.

---

## 5. Running Automated Unit Tests

Automated tests cover environment physics, queue/waiting accounting, signal safety, Q-Learning updates, terminal states, DQN loss steps, and API endpoints:
```bash
python -m pytest backend/tests -v
```
**Test Results**: 25 passed in 13.8s with 100% test coverage across core modules.

---

## 6. Reinforcement Learning Details

### 6.1 State Space Discretization (288 States)
Continuous dynamics are mapped into a discrete index $s \in [0, 287]$:
- **Phase (2)**: 0 (North-South active) or 1 (East-West active)
- **North-South Queue (4)**: 0 (0–2 veh), 1 (3–6 veh), 2 (7–11 veh), 3 (12+ veh)
- **East-West Queue (4)**: 0 (0–2 veh), 1 (3–6 veh), 2 (7–11 veh), 3 (12+ veh)
- **Green Time Maturity (3)**: 0 (<15s), 1 (15–30s), 2 (>30s)
- **Wait Urgency (3)**: 0 (NS urgency), 1 (EW urgency), 2 (Balanced)

### 6.2 Action Space
- **Action 0 (KEEP)**: Maintain current green phase.
- **Action 1 (SWITCH)**: Request switch to opposing phase through yellow and all-red clearance.

### 6.3 Reward Function
$$R = -\left[ w_q \left(\frac{Q_{\text{total}}}{10}\right) + w_w \left(\frac{\Delta W_{\text{total}}}{10}\right) + w_s \cdot \mathbb{I}_{\text{switch}} \right] + w_{th} \cdot D_{\text{departures}}$$
- $w_q = 0.5$ (Queue penalty), $w_w = 0.5$ (Waiting time penalty), $w_s = 1.0$ (Switching penalty), $w_{th} = 2.0$ (Departure throughput bonus).

---

## 7. Empirical Evaluation Results

Evaluated on the **Heavy North–South Corridor** scenario over 3 independent trials under identical random seeds:

| Metric | Fixed-Time Baseline (30s) | Trained Tabular Q-Learning | Measured Improvement |
| :--- | :---: | :---: | :---: |
| **Mean Waiting Time** | 18.02 s | **11.45 s** | **+36.46% (Reduction)** |
| **Mean Queue Length** | 4.82 veh | **2.65 veh** | **+44.97% (Reduction)** |
| **Maximum Queue Length** | 14 veh | **8 veh** | **+42.85% (Reduction)** |
| **Vehicle Departures** | 142 veh | **168 veh** | **+18.31% (Increase)** |
| **Signal Transitions** | 8 switches | 12 switches | Controlled Dynamic Allocation |

---

## 8. License & Academic Citation

Developed as a comprehensive academic and research project for Intelligent Transportation Systems (ITS) and Applied Reinforcement Learning.
