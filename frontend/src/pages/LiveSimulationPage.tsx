import React, { useState, useEffect } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  FastForward,
  Activity,
  Layers,
  Clock,
  Car,
  ToggleLeft,
  ChevronRight,
  Sliders,
  Sparkles,
} from 'lucide-react';
import { IntersectionCanvas } from '../components/IntersectionCanvas';
import { SimulationState, Scenario } from '../types';
import {
  fetchSimulationState,
  startSimulation,
  pauseSimulation,
  stepSimulation,
  resetSimulation,
  setSimulationController,
  setSimulationSpeed,
} from '../services/api';

interface LiveSimulationPageProps {
  scenarios: Scenario[];
}

export const LiveSimulationPage: React.FC<LiveSimulationPageProps> = ({ scenarios }) => {
  const [simState, setSimState] = useState<SimulationState | null>(null);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1.0);
  const [controllerType, setControllerType] = useState<string>('baseline');
  const [selectedScenario, setSelectedScenario] = useState<string>('balanced');
  const [isStepping, setIsStepping] = useState<boolean>(false);

  // Poll state or connect WebSocket
  useEffect(() => {
    let intervalId: any = null;

    const update = async () => {
      try {
        const state = await fetchSimulationState();
        setSimState(state);
        setIsRunning(!!state.is_running);
        if (state.controller_type) setControllerType(state.controller_type);
      } catch (err) {
        console.error('Failed to fetch sim state', err);
      }
    };

    update();
    // Fast polling for smooth vehicle movement
    intervalId = setInterval(update, 120);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  const handleTogglePlay = async () => {
    try {
      if (isRunning) {
        await pauseSimulation();
        setIsRunning(false);
      } else {
        await startSimulation();
        setIsRunning(true);
      }
      const state = await fetchSimulationState();
      setSimState(state);
    } catch (err) {
      console.error(err);
    }
  };

  const handleStep = async (action: number = 0) => {
    setIsStepping(true);
    try {
      const state = await stepSimulation(action, controllerType);
      setSimState(state);
    } catch (err) {
      console.error(err);
    } finally {
      setIsStepping(false);
    }
  };

  const handleReset = async () => {
    try {
      const state = await resetSimulation(selectedScenario);
      setSimState(state);
      setIsRunning(false);
    } catch (err) {
      console.error(err);
    }
  };

  const handleSpeedChange = async (newSpeed: number) => {
    setSpeed(newSpeed);
    try {
      await setSimulationSpeed(newSpeed);
    } catch (err) {
      console.error(err);
    }
  };

  const handleControllerChange = async (newController: string) => {
    setControllerType(newController);
    try {
      await setSimulationController(newController);
    } catch (err) {
      console.error(err);
    }
  };

  const handleScenarioChange = async (newScenario: string) => {
    setSelectedScenario(newScenario);
    try {
      const state = await resetSimulation(newScenario);
      setSimState(state);
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Activity className="text-blue-500" size={24} />
            Live Intersection Microscopic Simulation
          </h1>
          <p className="text-xs text-slate-400">
            Real-time microscopic vehicle car-following dynamics, clearance safety interlocks, and RL policy execution.
          </p>
        </div>

        {/* Quick Scenario Selector */}
        <div className="flex items-center gap-2 bg-slate-900/80 p-1.5 rounded-xl border border-white/10">
          <span className="text-xs font-semibold text-slate-400 px-2">Scenario:</span>
          <select
            value={selectedScenario}
            onChange={(e) => handleScenarioChange(e.target.value)}
            className="bg-slate-800 text-xs font-semibold text-white px-3 py-1.5 rounded-lg border border-white/10 focus:outline-none focus:border-blue-500"
          >
            {scenarios.map((sc) => (
              <option key={sc.id} value={sc.id}>
                {sc.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: Canvas & Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left/Center: Intersection Canvas */}
        <div className="lg:col-span-2 flex flex-col items-center">
          <IntersectionCanvas simulationState={simState} />
        </div>

        {/* Right Sidebar: Control Deck & Telemetry */}
        <div className="space-y-5">
          {/* Simulation Controls Panel */}
          <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-4">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Sliders size={14} className="text-blue-400" />
              Simulation Controls
            </h2>

            {/* Play / Step / Reset Buttons */}
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={handleTogglePlay}
                className={`flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl font-bold text-xs transition-all shadow-lg ${
                  isRunning
                    ? 'bg-amber-600 hover:bg-amber-500 text-white shadow-amber-600/30'
                    : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/30'
                }`}
              >
                {isRunning ? <Pause size={16} /> : <Play size={16} />}
                {isRunning ? 'Pause' : 'Start'}
              </button>

              <button
                onClick={() => handleStep(0)}
                disabled={isRunning || isStepping}
                className="flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-xl bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 font-semibold text-xs border border-white/10 transition-all"
              >
                <FastForward size={15} />
                Step
              </button>

              <button
                onClick={handleReset}
                className="flex items-center justify-center gap-1.5 py-2.5 px-3 rounded-xl bg-slate-800 hover:bg-rose-900/40 text-slate-300 hover:text-rose-300 font-semibold text-xs border border-white/10 transition-all"
              >
                <RotateCcw size={15} />
                Reset
              </button>
            </div>

            {/* Speed Multipliers */}
            <div className="space-y-1.5">
              <span className="text-[11px] font-medium text-slate-400">Simulation Speed:</span>
              <div className="grid grid-cols-4 gap-1.5">
                {[0.5, 1.0, 2.0, 5.0].map((s) => (
                  <button
                    key={s}
                    onClick={() => handleSpeedChange(s)}
                    className={`py-1 rounded-lg text-xs font-mono font-bold transition-all ${
                      speed === s
                        ? 'bg-blue-600 text-white border border-blue-400/50 shadow'
                        : 'bg-slate-800/80 hover:bg-slate-700 text-slate-400 border border-white/5'
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>
            </div>

            {/* Active Controller Switcher */}
            <div className="space-y-1.5 pt-1">
              <span className="text-[11px] font-medium text-slate-400">Active Controller:</span>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'baseline', label: 'Fixed-Time (30s)' },
                  { id: 'q_learning', label: 'Tabular Q-Learning' },
                  { id: 'dqn', label: 'Deep Q-Network' },
                  { id: 'manual', label: 'Manual Control' },
                ].map((c) => (
                  <button
                    key={c.id}
                    onClick={() => handleControllerChange(c.id)}
                    className={`p-2.5 rounded-xl text-left border transition-all ${
                      controllerType === c.id
                        ? 'bg-indigo-600/30 border-indigo-500 text-white font-bold'
                        : 'bg-slate-900/60 border-white/5 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <div className="text-xs">{c.label}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Manual Action Button (Only if manual mode is chosen) */}
            {controllerType === 'manual' && (
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 space-y-2">
                <div className="text-xs font-semibold text-amber-400">Manual Signal Request</div>
                <button
                  onClick={() => handleStep(1)}
                  disabled={simState?.is_transitioning}
                  className="w-full py-2 rounded-lg bg-amber-600 hover:bg-amber-500 disabled:opacity-40 text-white font-bold text-xs shadow-md transition-all"
                >
                  Request Phase Switch
                </button>
                <p className="text-[10px] text-slate-400">
                  Respects safety constraints: Minimum green (10s), yellow clearance (3s), and all-red (2s).
                </p>
              </div>
            )}
          </div>

          {/* Telemetry Metrics */}
          <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
              <Clock size={14} className="text-emerald-400" />
              Live Simulation Telemetry
            </h2>

            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-[10px] font-medium text-slate-400">Simulation Clock</span>
                <div className="text-lg font-bold font-mono text-white">
                  {simState ? `${simState.sim_time}s` : '0.0s'}
                </div>
              </div>

              <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-[10px] font-medium text-slate-400">Departures (Throughput)</span>
                <div className="text-lg font-bold font-mono text-emerald-400">
                  {simState ? simState.completed_count : 0}
                </div>
              </div>

              <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-[10px] font-medium text-slate-400">Total In-Queue</span>
                <div className="text-lg font-bold font-mono text-amber-400">
                  {simState ? simState.total_queue : 0} veh
                </div>
              </div>

              <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-[10px] font-medium text-slate-400">Signal Switches</span>
                <div className="text-lg font-bold font-mono text-blue-400">
                  {simState ? simState.switch_count : 0}
                </div>
              </div>
            </div>

            <div className="bg-slate-900/80 p-3 rounded-xl border border-white/5 flex justify-between items-center text-xs">
              <span className="text-slate-400">Cumulative Episode Reward:</span>
              <span className={`font-mono font-bold ${
                (simState?.cumulative_reward || 0) >= 0 ? 'text-emerald-400' : 'text-rose-400'
              }`}>
                {simState ? simState.cumulative_reward : 0.0}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
