import React, { useState } from 'react';
import {
  BarChart3,
  TrendingDown,
  TrendingUp,
  Play,
  CheckCircle2,
  AlertCircle,
  Clock,
  Car,
  Activity,
  Zap,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { Scenario, EvaluationComparison } from '../types';
import { runEvaluation } from '../services/api';

interface ComparisonPageProps {
  scenarios: Scenario[];
}

export const ComparisonPage: React.FC<ComparisonPageProps> = ({ scenarios }) => {
  const [algorithm, setAlgorithm] = useState<string>('q_learning');
  const [scenarioId, setScenarioId] = useState<string>('heavy_north_south');
  const [fixedGreen, setFixedGreen] = useState<number>(30.0);
  const [numEpisodes, setNumEpisodes] = useState<number>(3);

  const [comparison, setComparison] = useState<EvaluationComparison | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleRunBenchmark = async () => {
    setIsLoading(true);
    try {
      const res = await runEvaluation({
        algorithm,
        scenario_id: scenarioId,
        fixed_green: fixedGreen,
        num_episodes: numEpisodes,
      });
      setComparison(res);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const chartData = comparison
    ? [
        {
          metric: 'Mean Wait Time (s)',
          Baseline: comparison.baseline.mean_waiting_time,
          RL_Agent: comparison.rl.mean_waiting_time,
        },
        {
          metric: 'Mean Queue (veh)',
          Baseline: comparison.baseline.mean_queue_length,
          RL_Agent: comparison.rl.mean_queue_length,
        },
        {
          metric: 'Throughput (veh)',
          Baseline: comparison.baseline.mean_throughput,
          RL_Agent: comparison.rl.mean_throughput,
        },
      ]
    : [];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <BarChart3 className="text-emerald-400" size={24} />
          Performance Evaluation & Baseline Benchmarking
        </h1>
        <p className="text-xs text-slate-400">
          Scientifically evaluate Reinforcement Learning controllers against conventional Fixed-Time baselines under identical seeds and demand distributions.
        </p>
      </div>

      {/* Benchmark Control Bar */}
      <div className="glass-card rounded-2xl p-5 border border-white/10 flex flex-wrap items-center gap-4">
        <div className="flex-1 min-w-[180px] space-y-1">
          <label className="text-[11px] font-semibold text-slate-400">RL Algorithm</label>
          <select
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10"
          >
            <option value="q_learning">Tabular Q-Learning</option>
            <option value="dqn">Deep Q-Network (DQN)</option>
          </select>
        </div>

        <div className="flex-1 min-w-[200px] space-y-1">
          <label className="text-[11px] font-semibold text-slate-400">Traffic Scenario</label>
          <select
            value={scenarioId}
            onChange={(e) => setScenarioId(e.target.value)}
            className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10"
          >
            {scenarios.map((sc) => (
              <option key={sc.id} value={sc.id}>
                {sc.name}
              </option>
            ))}
          </select>
        </div>

        <div className="w-32 space-y-1">
          <label className="text-[11px] font-semibold text-slate-400">Fixed Green (s)</label>
          <input
            type="number"
            min={10}
            max={90}
            value={fixedGreen}
            onChange={(e) => setFixedGreen(Number(e.target.value))}
            className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
          />
        </div>

        <div className="w-32 space-y-1">
          <label className="text-[11px] font-semibold text-slate-400">Test Trials</label>
          <input
            type="number"
            min={1}
            max={10}
            value={numEpisodes}
            onChange={(e) => setNumEpisodes(Number(e.target.value))}
            className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
          />
        </div>

        <div className="pt-5">
          <button
            onClick={handleRunBenchmark}
            disabled={isLoading}
            className="flex items-center gap-2 px-6 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-bold text-xs shadow-lg shadow-emerald-600/30 transition-all hover:scale-[1.02]"
          >
            <Play size={15} />
            {isLoading ? 'Running Benchmark...' : 'Run Head-to-Head'}
          </button>
        </div>
      </div>

      {/* Comparison Results */}
      {comparison && (
        <div className="space-y-6">
          {/* Scientific Verdict Banner */}
          <div className="p-4 rounded-2xl bg-slate-900 border border-white/10 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400">
                <Zap size={20} />
              </div>
              <div>
                <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">Evaluation Verdict</div>
                <div className="text-sm font-bold text-white">{comparison.verdict}</div>
              </div>
            </div>
            <div className="text-xs text-slate-400 font-mono">
              Evaluated over {comparison.baseline.episodes.length} independent trials
            </div>
          </div>

          {/* Metric Improvement Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Wait Time */}
            <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-2">
              <span className="text-xs font-semibold text-slate-400">Waiting Time Reduction</span>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-2xl font-bold font-mono ${
                    comparison.improvements.waiting_time_percent > 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {comparison.improvements.waiting_time_percent > 0 ? '+' : ''}
                  {comparison.improvements.waiting_time_percent}%
                </span>
                {comparison.improvements.waiting_time_percent > 0 ? (
                  <TrendingDown size={18} className="text-emerald-400" />
                ) : (
                  <TrendingUp size={18} className="text-rose-400" />
                )}
              </div>
              <div className="text-xs text-slate-400 flex justify-between font-mono pt-1">
                <span>Baseline: {comparison.baseline.mean_waiting_time}s</span>
                <span>RL: {comparison.rl.mean_waiting_time}s</span>
              </div>
            </div>

            {/* Queue Length */}
            <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-2">
              <span className="text-xs font-semibold text-slate-400">Queue Length Reduction</span>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-2xl font-bold font-mono ${
                    comparison.improvements.queue_length_percent > 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {comparison.improvements.queue_length_percent > 0 ? '+' : ''}
                  {comparison.improvements.queue_length_percent}%
                </span>
                {comparison.improvements.queue_length_percent > 0 ? (
                  <TrendingDown size={18} className="text-emerald-400" />
                ) : (
                  <TrendingUp size={18} className="text-rose-400" />
                )}
              </div>
              <div className="text-xs text-slate-400 flex justify-between font-mono pt-1">
                <span>Baseline: {comparison.baseline.mean_queue_length}</span>
                <span>RL: {comparison.rl.mean_queue_length}</span>
              </div>
            </div>

            {/* Throughput */}
            <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-2">
              <span className="text-xs font-semibold text-slate-400">Vehicle Throughput Change</span>
              <div className="flex items-baseline gap-2">
                <span
                  className={`text-2xl font-bold font-mono ${
                    comparison.improvements.throughput_percent >= 0 ? 'text-emerald-400' : 'text-rose-400'
                  }`}
                >
                  {comparison.improvements.throughput_percent >= 0 ? '+' : ''}
                  {comparison.improvements.throughput_percent}%
                </span>
                {comparison.improvements.throughput_percent >= 0 ? (
                  <TrendingUp size={18} className="text-emerald-400" />
                ) : (
                  <TrendingDown size={18} className="text-rose-400" />
                )}
              </div>
              <div className="text-xs text-slate-400 flex justify-between font-mono pt-1">
                <span>Baseline: {comparison.baseline.mean_throughput}</span>
                <span>RL: {comparison.rl.mean_throughput}</span>
              </div>
            </div>

            {/* Signal Switches */}
            <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-2">
              <span className="text-xs font-semibold text-slate-400">Signal Switch Frequency</span>
              <div className="text-2xl font-bold font-mono text-white">
                {comparison.rl.mean_switches} switches
              </div>
              <div className="text-xs text-slate-400 flex justify-between font-mono pt-1">
                <span>Baseline: {comparison.baseline.mean_switches}</span>
                <span className="text-slate-400">Safe clearance</span>
              </div>
            </div>
          </div>

          {/* Bar Chart Comparison */}
          <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
            <h2 className="text-sm font-bold text-white uppercase tracking-wider">
              Comparative Metrics Visualization
            </h2>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="metric" stroke="#64748B" fontSize={12} />
                  <YAxis stroke="#64748B" fontSize={11} />
                  <Tooltip contentStyle={{ backgroundColor: '#0F172A', borderColor: '#334155', fontSize: '12px' }} />
                  <Legend />
                  <Bar dataKey="Baseline" fill="#64748B" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="RL_Agent" fill="#10B981" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
