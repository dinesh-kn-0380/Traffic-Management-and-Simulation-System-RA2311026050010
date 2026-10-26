import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Play,
  Pause,
  Square,
  RefreshCw,
  TrendingUp,
  Clock,
  Car,
  Activity,
  Layers,
  Sparkles,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Scenario, TrainingStatus, TrainingMetric } from '../types';
import {
  startTraining,
  pauseTraining,
  resumeTraining,
  stopTraining,
  fetchTrainingStatus,
  fetchTrainingHistory,
} from '../services/api';

interface TrainingPageProps {
  scenarios: Scenario[];
}

export const TrainingPage: React.FC<TrainingPageProps> = ({ scenarios }) => {
  const [algorithm, setAlgorithm] = useState<string>('q_learning');
  const [episodes, setEpisodes] = useState<number>(40);
  const [episodeDuration, setEpisodeDuration] = useState<number>(150.0);
  const [scenarioId, setScenarioId] = useState<string>('heavy_north_south');
  const [alpha, setAlpha] = useState<number>(0.15);
  const [gamma, setGamma] = useState<number>(0.95);
  const [epsilon, setEpsilon] = useState<number>(1.0);
  const [epsilonMin, setEpsilonMin] = useState<number>(0.05);
  const [epsilonDecay, setEpsilonDecay] = useState<number>(0.985);
  const [seed, setSeed] = useState<number>(42);

  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [history, setHistory] = useState<TrainingMetric[]>([]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  // Poll training status
  useEffect(() => {
    let intervalId: any = null;

    const poll = async () => {
      try {
        const s = await fetchTrainingStatus();
        setStatus(s);
        if (s.is_training || s.history_length > 0) {
          const h = await fetchTrainingHistory();
          setHistory(h);
        }
      } catch (err) {
        console.error('Failed to poll training status', err);
      }
    };

    poll();
    intervalId = setInterval(poll, 1000);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, []);

  const handleStart = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await startTraining({
        algorithm,
        episodes,
        episode_duration: episodeDuration,
        scenario_id: scenarioId,
        alpha,
        gamma,
        epsilon,
        epsilon_min: epsilonMin,
        epsilon_decay: epsilonDecay,
        seed,
      });
      const s = await fetchTrainingStatus();
      setStatus(s);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handlePause = async () => {
    await pauseTraining();
  };

  const handleResume = async () => {
    await resumeTraining();
  };

  const handleStop = async () => {
    await stopTraining();
  };

  const isTraining = status?.is_training ?? false;
  const isPaused = status?.is_paused ?? false;

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Page Title */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Cpu className="text-indigo-400" size={24} />
            Reinforcement Learning Agent Training Studio
          </h1>
          <p className="text-xs text-slate-400">
            Train Q-Learning and Deep Q-Network agents on custom traffic distributions with real-time loss and reward telemetry.
          </p>
        </div>

        {status && (
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-white/10 text-xs font-mono">
            <span
              className={`w-2.5 h-2.5 rounded-full ${
                isTraining ? (isPaused ? 'bg-amber-400' : 'bg-emerald-400 animate-pulse') : 'bg-slate-500'
              }`}
            ></span>
            <span className="font-semibold text-slate-200">{status.status_message}</span>
          </div>
        )}
      </div>

      {/* Main Grid: Training Setup & Real-time Curves */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Hyperparameters Form */}
        <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-5">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Layers size={16} className="text-indigo-400" />
            Training Configuration
          </h2>

          <form onSubmit={handleStart} className="space-y-4">
            {/* Algorithm Select */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300">RL Algorithm</label>
              <select
                disabled={isTraining}
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value)}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
              >
                <option value="q_learning">Tabular Q-Learning (Bellman TD)</option>
                <option value="dqn">Deep Q-Network (PyTorch MLP)</option>
              </select>
            </div>

            {/* Traffic Scenario */}
            <div className="space-y-1.5">
              <label className="text-xs font-semibold text-slate-300">Traffic Scenario</label>
              <select
                disabled={isTraining}
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 focus:outline-none focus:border-indigo-500 disabled:opacity-50"
              >
                {scenarios.map((sc) => (
                  <option key={sc.id} value={sc.id}>
                    {sc.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Episodes & Duration */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Episodes</label>
                <input
                  type="number"
                  disabled={isTraining}
                  min={5}
                  max={200}
                  value={episodes}
                  onChange={(e) => setEpisodes(Number(e.target.value))}
                  className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 disabled:opacity-50 font-mono"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Duration (s)</label>
                <input
                  type="number"
                  disabled={isTraining}
                  min={50}
                  max={500}
                  value={episodeDuration}
                  onChange={(e) => setEpisodeDuration(Number(e.target.value))}
                  className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 disabled:opacity-50 font-mono"
                />
              </div>
            </div>

            {/* Hyperparameters: Alpha, Gamma */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Learning Rate (α)</label>
                <input
                  type="number"
                  step="0.01"
                  disabled={isTraining}
                  value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 disabled:opacity-50 font-mono"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Discount (γ)</label>
                <input
                  type="number"
                  step="0.01"
                  disabled={isTraining}
                  value={gamma}
                  onChange={(e) => setGamma(Number(e.target.value))}
                  className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 disabled:opacity-50 font-mono"
                />
              </div>
            </div>

            {/* Exploration: Epsilon Decay */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Initial ε</label>
                <input
                  type="number"
                  step="0.1"
                  disabled={isTraining}
                  value={epsilon}
                  onChange={(e) => setEpsilon(Number(e.target.value))}
                  className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 disabled:opacity-50 font-mono"
                />
              </div>

              <div className="space-y-1">
                <label className="text-xs font-semibold text-slate-300">Decay Rate</label>
                <input
                  type="number"
                  step="0.005"
                  disabled={isTraining}
                  value={epsilonDecay}
                  onChange={(e) => setEpsilonDecay(Number(e.target.value))}
                  className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 disabled:opacity-50 font-mono"
                />
              </div>
            </div>

            {/* Random Seed */}
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Deterministic Seed</label>
              <input
                type="number"
                disabled={isTraining}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 disabled:opacity-50 font-mono"
              />
            </div>

            {/* Action Buttons */}
            <div className="pt-2">
              {!isTraining ? (
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-bold text-xs text-white shadow-lg shadow-indigo-600/30 transition-all hover:scale-[1.01]"
                >
                  <Play size={16} />
                  Start Training Pipeline
                </button>
              ) : (
                <div className="grid grid-cols-2 gap-2">
                  {isPaused ? (
                    <button
                      type="button"
                      onClick={handleResume}
                      className="flex items-center justify-center gap-1.5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs transition-all shadow"
                    >
                      <Play size={15} /> Resume
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={handlePause}
                      className="flex items-center justify-center gap-1.5 py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 text-white font-bold text-xs transition-all shadow"
                    >
                      <Pause size={15} /> Pause
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={handleStop}
                    className="flex items-center justify-center gap-1.5 py-2.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-bold text-xs transition-all shadow"
                  >
                    <Square size={15} /> Stop
                  </button>
                </div>
              )}
            </div>
          </form>
        </div>

        {/* Right Columns: Telemetry, Progress, and Live Charts */}
        <div className="lg:col-span-2 space-y-5">
          {/* Progress Card */}
          <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-3">
            <div className="flex justify-between items-center text-xs">
              <span className="font-semibold text-slate-300">Episode Progress:</span>
              <span className="font-mono text-indigo-400 font-bold">
                {status ? `${status.current_episode} / ${status.total_episodes}` : '0 / 0'}
                {' '}({status?.progress_percent || 0}%)
              </span>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-3 bg-slate-900 rounded-full overflow-hidden border border-white/5">
              <div
                className="h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-emerald-400 rounded-full transition-all duration-300"
                style={{ width: `${status?.progress_percent || 0}%` }}
              ></div>
            </div>

            {/* Quick Metrics */}
            <div className="grid grid-cols-3 gap-2 pt-1 text-center">
              <div className="bg-slate-900/60 p-2.5 rounded-xl border border-white/5">
                <div className="text-[10px] text-slate-400">Current Epsilon (ε)</div>
                <div className="text-sm font-bold font-mono text-amber-400">
                  {status?.current_epsilon?.toFixed(3) || '1.000'}
                </div>
              </div>

              <div className="bg-slate-900/60 p-2.5 rounded-xl border border-white/5">
                <div className="text-[10px] text-slate-400">Elapsed Time</div>
                <div className="text-sm font-bold font-mono text-slate-200">
                  {status ? `${status.elapsed_seconds}s` : '0.0s'}
                </div>
              </div>

              <div className="bg-slate-900/60 p-2.5 rounded-xl border border-white/5">
                <div className="text-[10px] text-slate-400">Latest Ep Reward</div>
                <div className="text-sm font-bold font-mono text-emerald-400">
                  {status?.latest_metric?.total_reward ?? 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* Recharts: Episode Reward & Waiting Time */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Chart 1: Reward */}
            <div className="glass-card rounded-2xl p-4 border border-white/10 space-y-2">
              <div className="text-xs font-bold text-slate-300">Episode Cumulative Reward</div>
              <div className="h-48 w-full">
                {history.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="episode" stroke="#64748B" fontSize={10} />
                      <YAxis stroke="#64748B" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: '11px' }} />
                      <Line type="monotone" dataKey="total_reward" stroke="#3B82F6" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-slate-500">
                    Reward curve will display as episodes complete...
                  </div>
                )}
              </div>
            </div>

            {/* Chart 2: Average Waiting Time */}
            <div className="glass-card rounded-2xl p-4 border border-white/10 space-y-2">
              <div className="text-xs font-bold text-slate-300">Mean Vehicle Waiting Time (s)</div>
              <div className="h-48 w-full">
                {history.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                      <XAxis dataKey="episode" stroke="#64748B" fontSize={10} />
                      <YAxis stroke="#64748B" fontSize={10} />
                      <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: '11px' }} />
                      <Line type="monotone" dataKey="mean_waiting_time" stroke="#10B981" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-slate-500">
                    Waiting time curve will display as episodes complete...
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Episode History Table */}
          {history.length > 0 && (
            <div className="glass-card rounded-2xl p-4 border border-white/10 space-y-2">
              <div className="text-xs font-bold text-slate-300">Completed Episodes Log</div>
              <div className="max-h-44 overflow-y-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="sticky top-0 bg-slate-900 text-slate-400 text-[10px] uppercase">
                    <tr>
                      <th className="p-2">Ep</th>
                      <th className="p-2">Reward</th>
                      <th className="p-2">Mean Wait</th>
                      <th className="p-2">Mean Queue</th>
                      <th className="p-2">Throughput</th>
                      <th className="p-2">Epsilon</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-mono text-[11px]">
                    {history.slice(-6).reverse().map((m) => (
                      <tr key={m.episode}>
                        <td className="p-2 font-bold text-white">{m.episode}</td>
                        <td className="p-2 text-blue-400">{m.total_reward}</td>
                        <td className="p-2 text-emerald-400">{m.mean_waiting_time}s</td>
                        <td className="p-2 text-amber-400">{m.mean_queue_length} veh</td>
                        <td className="p-2 text-slate-200">{m.total_departures}</td>
                        <td className="p-2 text-slate-400">{m.epsilon}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
