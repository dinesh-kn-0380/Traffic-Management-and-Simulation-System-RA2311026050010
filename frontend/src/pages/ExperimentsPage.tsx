import React, { useState } from 'react';
import {
  History,
  Download,
  Search,
  Filter,
  Layers,
  ChevronRight,
  Clock,
  Car,
  Cpu,
  X,
} from 'lucide-react';
import { Experiment } from '../types';

interface ExperimentsPageProps {
  experiments: Experiment[];
  onRefresh: () => void;
}

export const ExperimentsPage: React.FC<ExperimentsPageProps> = ({ experiments, onRefresh }) => {
  const [filterAlgo, setFilterAlgo] = useState<string>('all');
  const [search, setSearch] = useState<string>('');
  const [selectedExp, setSelectedExp] = useState<Experiment | null>(null);

  const filtered = experiments.filter((exp) => {
    if (filterAlgo !== 'all' && exp.algorithm !== filterAlgo) return false;
    if (search) {
      const q = search.toLowerCase();
      return (
        exp.name.toLowerCase().includes(q) ||
        exp.experiment_id.toLowerCase().includes(q) ||
        exp.scenario.toLowerCase().includes(q)
      );
    }
    return true;
  });

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <History className="text-blue-400" size={24} />
            Experiment Management & Historical Logs
          </h1>
          <p className="text-xs text-slate-400">
            Persistent SQLite audit log of model training iterations, evaluation benchmarks, and checkpoints.
          </p>
        </div>

        {/* Export CSV Button */}
        <a
          href="/api/experiments/export"
          download="traffic_rl_experiments.csv"
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs shadow-md shadow-blue-600/30 transition-all hover:scale-[1.02]"
        >
          <Download size={15} />
          Export to CSV
        </a>
      </div>

      {/* Filter and Search Bar */}
      <div className="glass-card rounded-2xl p-4 border border-white/10 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Filter size={15} className="text-slate-400" />
          <span className="text-xs text-slate-300 font-semibold">Algorithm:</span>
          <select
            value={filterAlgo}
            onChange={(e) => setFilterAlgo(e.target.value)}
            className="bg-slate-800 text-xs text-white px-3 py-1.5 rounded-lg border border-white/10"
          >
            <option value="all">All Algorithms</option>
            <option value="q_learning">Tabular Q-Learning</option>
            <option value="dqn">Deep Q-Network</option>
          </select>
        </div>

        <div className="relative">
          <Search size={14} className="absolute left-3 top-2.5 text-slate-400" />
          <input
            type="text"
            placeholder="Search experiments..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-slate-800 text-xs text-white pl-8 pr-4 py-1.5 rounded-lg border border-white/10 w-64"
          />
        </div>
      </div>

      {/* Experiments Table */}
      <div className="glass-card rounded-2xl p-5 border border-white/10 space-y-3">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-slate-900/80 text-slate-400 text-[10px] uppercase tracking-wider">
              <tr>
                <th className="p-3">ID</th>
                <th className="p-3">Name</th>
                <th className="p-3">Algorithm</th>
                <th className="p-3">Scenario</th>
                <th className="p-3">Episodes</th>
                <th className="p-3">Duration</th>
                <th className="p-3">Mean Wait</th>
                <th className="p-3">Mean Queue</th>
                <th className="p-3">Departures</th>
                <th className="p-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 font-mono text-[11px]">
              {filtered.map((exp) => (
                <tr key={exp.id} className="hover:bg-white/5 transition-colors">
                  <td className="p-3 font-bold text-blue-400">{exp.experiment_id}</td>
                  <td className="p-3 font-semibold text-white font-sans">{exp.name}</td>
                  <td className="p-3 uppercase text-purple-400">{exp.algorithm}</td>
                  <td className="p-3 font-sans">{exp.scenario}</td>
                  <td className="p-3">{exp.episodes}</td>
                  <td className="p-3">{exp.training_duration}s</td>
                  <td className="p-3 text-emerald-400 font-bold">{exp.mean_waiting_time}s</td>
                  <td className="p-3 text-amber-400">{exp.mean_queue_length}</td>
                  <td className="p-3 text-white font-bold">{exp.throughput}</td>
                  <td className="p-3 text-center">
                    <button
                      onClick={() => setSelectedExp(exp)}
                      className="px-2.5 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-sans font-semibold border border-white/10 transition-all"
                    >
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detail Modal */}
      {selectedExp && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4 animate-fadeIn">
          <div className="glass-card bg-[#0F172A] rounded-2xl max-w-xl w-full p-6 border border-white/10 space-y-4 shadow-2xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <div>
                <span className="text-xs font-mono font-bold text-blue-400">{selectedExp.experiment_id}</span>
                <h3 className="text-base font-bold text-white">{selectedExp.name}</h3>
              </div>
              <button
                onClick={() => setSelectedExp(null)}
                className="p-1 rounded-lg hover:bg-white/10 text-slate-400"
              >
                <X size={18} />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-slate-900 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase">Algorithm</span>
                <div className="font-bold text-white uppercase">{selectedExp.algorithm}</div>
              </div>
              <div className="bg-slate-900 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase">Scenario</span>
                <div className="font-bold text-white">{selectedExp.scenario}</div>
              </div>
              <div className="bg-slate-900 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase">Mean Waiting Time</span>
                <div className="font-bold text-emerald-400 font-mono">{selectedExp.mean_waiting_time}s</div>
              </div>
              <div className="bg-slate-900 p-3 rounded-xl border border-white/5 space-y-1">
                <span className="text-slate-400 text-[10px] uppercase">Vehicle Throughput</span>
                <div className="font-bold text-white font-mono">{selectedExp.throughput} departures</div>
              </div>
            </div>

            <div className="space-y-1.5">
              <span className="text-xs font-semibold text-slate-300">Hyperparameters & Config</span>
              <pre className="p-3 bg-slate-950 rounded-xl border border-white/5 text-[11px] font-mono text-slate-300 overflow-x-auto">
                {JSON.stringify(selectedExp.hyperparameters, null, 2)}
              </pre>
            </div>

            <div className="text-[11px] text-slate-400">
              Checkpoint: <span className="font-mono text-slate-300">{selectedExp.checkpoint_path || 'None'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
