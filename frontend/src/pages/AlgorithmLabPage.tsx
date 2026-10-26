import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Cpu,
  Layers,
  Sparkles,
  ShieldCheck,
  CheckCircle,
  HelpCircle,
  ArrowRight,
  Database,
  Search,
} from 'lucide-react';
import { QTableSummary } from '../types';
import { fetchQTable } from '../services/api';

export const AlgorithmLabPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'q_learning' | 'dqn' | 'state_space' | 'reward_safety'>('q_learning');
  const [qTableData, setQTableData] = useState<QTableSummary | null>(null);
  const [filterPhase, setFilterPhase] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchQTable()
      .then((data) => setQTableData(data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  const filteredEntries = qTableData?.entries.filter((entry) => {
    if (filterPhase !== 'all' && !entry.phase.toLowerCase().includes(filterPhase)) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      return (
        entry.ns_queue.toLowerCase().includes(q) ||
        entry.ew_queue.toLowerCase().includes(q) ||
        entry.green_time.toLowerCase().includes(q) ||
        entry.wait_ratio.toLowerCase().includes(q)
      );
    }
    return true;
  }) || [];

  return (
    <div className="space-y-6 animate-fadeIn">
      {/* Page Title */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <BookOpen className="text-purple-400" size={24} />
          Algorithm & Reinforcement Learning Laboratory
        </h1>
        <p className="text-xs text-slate-400">
          Mathematical derivations, state-action utility inspection, experience replay mechanisms, and signal safety guarantees.
        </p>
      </div>

      {/* Lab Tabs */}
      <div className="flex border-b border-white/10 gap-2">
        {[
          { id: 'q_learning', label: 'Tabular Q-Learning' },
          { id: 'dqn', label: 'Deep Q-Network (DQN)' },
          { id: 'state_space', label: 'State Space Discretization' },
          { id: 'reward_safety', label: 'Reward & Safety Guarantees' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`pb-3 px-3 text-xs font-semibold border-b-2 transition-all ${
              activeTab === tab.id
                ? 'border-purple-500 text-purple-400 font-bold'
                : 'border-transparent text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab 1: Tabular Q-Learning & Q-Table Heatmap */}
      {activeTab === 'q_learning' && (
        <div className="space-y-6">
          {/* Theory Card */}
          <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
            <h2 className="text-base font-bold text-white">Tabular Q-Learning Mathematical Formulation</h2>
            <p className="text-xs text-slate-300 leading-relaxed">
              Tabular Q-Learning is a model-free, off-policy Temporal Difference (TD) reinforcement learning algorithm.
              The agent maintains a discrete lookup matrix Q(s, a) of dimension 288 x 2 representing the expected
              discounted cumulative reward of executing action a in state s.
            </p>

            <div className="bg-slate-900/90 p-4 rounded-xl border border-white/5 font-mono text-xs text-purple-300 space-y-1">
              <div className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Bellman TD Update Equation:</div>
              <div className="text-sm font-bold text-white pt-1">
                Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') - Q(s, a) ]
              </div>
              <div className="text-[11px] text-slate-400 pt-1">
                Where α is the learning rate, γ is the discount factor, and terminal states set future value to 0.
              </div>
            </div>
          </div>

          {/* Live Q-Table Inspector */}
          <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-3">
              <div>
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <Database size={16} className="text-purple-400" />
                  Learned Q-Table Inspector & Policy Advantage
                </h2>
                <p className="text-xs text-slate-400">
                  Showing top state-action utility estimates learned by the agent.
                </p>
              </div>

              {/* Filter controls */}
              <div className="flex items-center gap-2">
                <select
                  value={filterPhase}
                  onChange={(e) => setFilterPhase(e.target.value)}
                  className="bg-slate-800 text-xs text-white px-2.5 py-1.5 rounded-lg border border-white/10"
                >
                  <option value="all">All Phases</option>
                  <option value="north-south">North-South Green</option>
                  <option value="east-west">East-West Green</option>
                </select>

                <div className="relative">
                  <Search size={13} className="absolute left-2.5 top-2 text-slate-400" />
                  <input
                    type="text"
                    placeholder="Search queues/urgency..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="bg-slate-800 text-xs text-white pl-7 pr-3 py-1.5 rounded-lg border border-white/10 w-44"
                  />
                </div>
              </div>
            </div>

            {loading ? (
              <div className="text-center py-10 text-xs text-slate-400">Loading learned Q-table...</div>
            ) : filteredEntries.length === 0 ? (
              <div className="text-center py-10 text-xs text-slate-400">No states matching current filter.</div>
            ) : (
              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="sticky top-0 bg-slate-900/90 backdrop-blur-md text-slate-400 text-[10px] uppercase">
                    <tr>
                      <th className="p-2.5">State ID</th>
                      <th className="p-2.5">Current Phase</th>
                      <th className="p-2.5">NS Queue</th>
                      <th className="p-2.5">EW Queue</th>
                      <th className="p-2.5">Green Time</th>
                      <th className="p-2.5">Urgency</th>
                      <th className="p-2.5 text-right">Q(Keep)</th>
                      <th className="p-2.5 text-right">Q(Switch)</th>
                      <th className="p-2.5 text-center">Optimal Action</th>
                      <th className="p-2.5 text-right">Advantage</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5 font-mono text-[11px]">
                    {filteredEntries.map((row) => (
                      <tr key={row.state_id} className="hover:bg-white/5 transition-colors">
                        <td className="p-2.5 font-bold text-slate-400">#{row.state_id}</td>
                        <td className="p-2.5 text-slate-200">{row.phase}</td>
                        <td className="p-2.5 text-amber-300">{row.ns_queue}</td>
                        <td className="p-2.5 text-indigo-300">{row.ew_queue}</td>
                        <td className="p-2.5 text-slate-300">{row.green_time}</td>
                        <td className="p-2.5 text-slate-400">{row.wait_ratio}</td>
                        <td className="p-2.5 text-right font-bold text-blue-400">{row.q_keep.toFixed(2)}</td>
                        <td className="p-2.5 text-right font-bold text-purple-400">{row.q_switch.toFixed(2)}</td>
                        <td className="p-2.5 text-center">
                          <span
                            className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              row.optimal_action === 'KEEP'
                                ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
                                : 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                            }`}
                          >
                            {row.optimal_action}
                          </span>
                        </td>
                        <td className="p-2.5 text-right text-emerald-400 font-bold">{row.advantage.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Tab 2: Deep Q-Network (DQN) */}
      {activeTab === 'dqn' && (
        <div className="space-y-6">
          <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
            <h2 className="text-base font-bold text-white">Deep Q-Network (PyTorch MLP) Architecture</h2>
            <p className="text-xs text-slate-300 leading-relaxed">
              DQN replaces tabular storage with a Multi-Layer Perceptron (MLP) function approximator
              $Q(s, a; \theta) \approx Q^*(s, a)$. It addresses continuous traffic observation vectors and
              breaks temporal correlation via Experience Replay and a separate Target Network $\theta^-$.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
              <div className="bg-slate-900/80 p-4 rounded-xl border border-white/5 space-y-2">
                <div className="text-[10px] font-bold text-blue-400 uppercase">Input Layer (10 Dimensions)</div>
                <div className="text-xs font-semibold text-white">Normalized Traffic State Vector</div>
                <p className="text-xs text-slate-400">
                  Normalized approach queues $[N, S, E, W]$, waiting times $[N, S, E, W]$, phase ID, and elapsed green time.
                </p>
              </div>

              <div className="bg-slate-900/80 p-4 rounded-xl border border-white/5 space-y-2">
                <div className="text-[10px] font-bold text-indigo-400 uppercase">Hidden Layers (64 x 64)</div>
                <div className="text-xs font-semibold text-white">Dense Layers + ReLU Activations</div>
                <p className="text-xs text-slate-400">
                  Fully connected layers with gradient clipping (norm 1.0) and Adam optimization (lr = 0.001).
                </p>
              </div>

              <div className="bg-slate-900/80 p-4 rounded-xl border border-white/5 space-y-2">
                <div className="text-[10px] font-bold text-emerald-400 uppercase">Output Layer (2 Actions)</div>
                <div className="text-xs font-semibold text-white">Q-Value Prediction</div>
                <p className="text-xs text-slate-400">
                  Direct estimates for Q(s, keep) and Q(s, switch) evaluated against Huber loss.
                </p>
              </div>
            </div>

            <div className="bg-slate-900/90 p-4 rounded-xl border border-white/5 font-mono text-xs text-indigo-300 space-y-1">
              <div className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Experience Replay Bellman Loss:</div>
              <div className="text-sm font-bold text-white pt-1">
                L(θ) = E [ ( r + γ · max_a' Q(s', a'; θ_target) - Q(s, a; θ) )² ]
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Tab 3: State Space Discretization */}
      {activeTab === 'state_space' && (
        <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-5">
          <h2 className="text-base font-bold text-white">State Space Discretization & Binning Logic</h2>
          <p className="text-xs text-slate-300 leading-relaxed">
            To guarantee fast and robust convergence for Tabular Q-Learning without state explosion, continuous
            traffic dynamics are discretized into a compact product space of 288 states:
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 pt-2">
            <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-2">
              <div className="text-[10px] font-bold text-blue-400 uppercase">Phase ID (2)</div>
              <div className="text-xs font-bold text-white">Active Corridor</div>
              <ul className="text-xs text-slate-400 space-y-1">
                <li>• 0: North–South Active</li>
                <li>• 1: East–West Active</li>
              </ul>
            </div>

            <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-2">
              <div className="text-[10px] font-bold text-amber-400 uppercase">NS Queue Bin (4)</div>
              <div className="text-xs font-bold text-white">North-South Queues</div>
              <ul className="text-xs text-slate-400 space-y-1">
                <li>• 0: 0–2 vehicles (Light)</li>
                <li>• 1: 3–6 vehicles (Moderate)</li>
                <li>• 2: 7–11 vehicles (Heavy)</li>
                <li>• 3: 12+ vehicles (Saturated)</li>
              </ul>
            </div>

            <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-2">
              <div className="text-[10px] font-bold text-amber-400 uppercase">EW Queue Bin (4)</div>
              <div className="text-xs font-bold text-white">East-West Queues</div>
              <ul className="text-xs text-slate-400 space-y-1">
                <li>• 0: 0–2 vehicles (Light)</li>
                <li>• 1: 3–6 vehicles (Moderate)</li>
                <li>• 2: 7–11 vehicles (Heavy)</li>
                <li>• 3: 12+ vehicles (Saturated)</li>
              </ul>
            </div>

            <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-2">
              <div className="text-[10px] font-bold text-emerald-400 uppercase">Green Elapsed (3)</div>
              <div className="text-xs font-bold text-white">Phase Maturity</div>
              <ul className="text-xs text-slate-400 space-y-1">
                <li>• 0: &lt; 15s (Recent switch)</li>
                <li>• 1: 15s–30s (Mid green)</li>
                <li>• 2: &gt; 30s (Mature green)</li>
              </ul>
            </div>

            <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-2">
              <div className="text-[10px] font-bold text-purple-400 uppercase">Wait Urgency (3)</div>
              <div className="text-xs font-bold text-white">Wait Ratio</div>
              <ul className="text-xs text-slate-400 space-y-1">
                <li>• 0: NS Wait &gt; EW Wait</li>
                <li>• 1: EW Wait &gt; NS Wait</li>
                <li>• 2: Balanced</li>
              </ul>
            </div>
          </div>

          <div className="p-4 rounded-xl bg-slate-900 border border-white/5 flex items-center justify-between text-xs text-slate-300">
            <span>Total State Space Cardinality:</span>
            <span className="font-mono text-emerald-400 font-bold text-sm">
              2 × 4 × 4 × 3 × 3 = 288 States
            </span>
          </div>
        </div>
      )}

      {/* Tab 4: Reward Formulation & Safety */}
      {activeTab === 'reward_safety' && (
        <div className="space-y-6">
          <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
            <h2 className="text-base font-bold text-white">Reward Function Design</h2>
            <p className="text-xs text-slate-300 leading-relaxed">
              The scalar reward function provides the essential feedback signal driving policy convergence.
              It penalizes queue lengths, incremental vehicle delay, and signal thrashing, while rewarding departures:
            </p>

            <div className="bg-slate-900/90 p-4 rounded-xl border border-white/5 font-mono text-xs text-rose-300 space-y-2">
              <div className="text-slate-400 text-[10px] uppercase font-bold tracking-wider">Reward Formulation:</div>
              <div className="text-sm font-bold text-white">
                R = - [ w_q · (Q_total / 10) + w_w · (ΔW_total / 10) + w_s · I_switch ] + w_th · Departures
              </div>
            </div>
          </div>

          <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <ShieldCheck size={18} className="text-emerald-400" />
              Signal Controller Safety Interlocks
            </h2>
            <p className="text-xs text-slate-300 leading-relaxed">
              Reinforcement Learning actions operate strictly through safety interlocks. The agent can never
              violate clearance regulations or create dangerous vehicle conflicts:
            </p>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-3 pt-2">
              <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-1">
                <div className="text-[10px] text-amber-400 font-bold uppercase">Yellow Clearance</div>
                <div className="text-xs font-bold text-white">3.0s Standard Interval</div>
                <p className="text-xs text-slate-400">Allows approaching vehicles in dilemma zone to clear intersection safely.</p>
              </div>

              <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-1">
                <div className="text-[10px] text-rose-400 font-bold uppercase">All-Red Buffer</div>
                <div className="text-xs font-bold text-white">2.0s Safety Window</div>
                <p className="text-xs text-slate-400">Ensures intersection box is completely vacant before opposing green activates.</p>
              </div>

              <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-1">
                <div className="text-[10px] text-blue-400 font-bold uppercase">Minimum Green</div>
                <div className="text-xs font-bold text-white">10.0s Min Duration</div>
                <p className="text-xs text-slate-400">Prevents rapid signal fluttering and guarantees driver expectation safety.</p>
              </div>

              <div className="bg-slate-900/80 p-3.5 rounded-xl border border-white/5 space-y-1">
                <div className="text-[10px] text-purple-400 font-bold uppercase">Maximum Green</div>
                <div className="text-xs font-bold text-white">60.0s Anti-Starvation</div>
                <p className="text-xs text-slate-400">Forces phase switch if one corridor holds green too long, preventing gridlock.</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
