import React, { useState, useEffect } from 'react';
import {
  Sliders,
  Save,
  RotateCcw,
  ShieldCheck,
  Award,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { AppConfig } from '../types';
import { fetchConfig, updateConfig } from '../services/api';

export const SettingsPage: React.FC = () => {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [minGreen, setMinGreen] = useState<number>(10.0);
  const [maxGreen, setMaxGreen] = useState<number>(60.0);
  const [yellowDuration, setYellowDuration] = useState<number>(3.0);
  const [allRedDuration, setAllRedDuration] = useState<number>(2.0);

  const [queueWeight, setQueueWeight] = useState<number>(0.5);
  const [waitingWeight, setWaitingWeight] = useState<number>(0.5);
  const [switchWeight, setSwitchWeight] = useState<number>(1.0);
  const [throughputWeight, setThroughputWeight] = useState<number>(2.0);

  const [isSaved, setIsSaved] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetchConfig()
      .then((cfg) => {
        setConfig(cfg);
        setMinGreen(cfg.timing.min_green);
        setMaxGreen(cfg.timing.max_green);
        setYellowDuration(cfg.timing.yellow_duration);
        setAllRedDuration(cfg.timing.all_red_duration);

        setQueueWeight(cfg.rewards.queue_weight);
        setWaitingWeight(cfg.rewards.waiting_weight);
        setSwitchWeight(cfg.rewards.switch_weight);
        setThroughputWeight(cfg.rewards.throughput_weight);
      })
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await updateConfig({
        min_green: minGreen,
        max_green: maxGreen,
        yellow_duration: yellowDuration,
        all_red_duration: allRedDuration,
        queue_weight: queueWeight,
        waiting_weight: waitingWeight,
        switch_weight: switchWeight,
        throughput_weight: throughputWeight,
      });
      setIsSaved(true);
      setTimeout(() => setIsSaved(false), 3000);
    } catch (err) {
      console.error(err);
    }
  };

  const handleResetDefaults = () => {
    setMinGreen(10.0);
    setMaxGreen(60.0);
    setYellowDuration(3.0);
    allRedDuration !== 2.0 && setAllRedDuration(2.0);
    setQueueWeight(0.5);
    setWaitingWeight(0.5);
    setSwitchWeight(1.0);
    setThroughputWeight(2.0);
  };

  return (
    <div className="space-y-6 max-w-4xl mx-auto animate-fadeIn">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Sliders className="text-amber-400" size={24} />
          System Simulation & Reward Settings
        </h1>
        <p className="text-xs text-slate-400">
          Tune microscopic traffic parameters, signal timing constraints, and reward weighting functions.
        </p>
      </div>

      {isSaved && (
        <div className="p-3 rounded-xl bg-emerald-500/20 border border-emerald-500/30 text-emerald-400 text-xs font-semibold flex items-center gap-2 animate-fadeIn">
          <CheckCircle2 size={16} />
          Settings successfully updated and applied across simulation engines.
        </div>
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Signal Timing Interlocks */}
        <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <ShieldCheck size={16} className="text-emerald-400" />
            Signal Phase Timing & Safety Constraints
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Minimum Green Duration (seconds)</label>
              <input
                type="number"
                step="1"
                min={5}
                max={30}
                value={minGreen}
                onChange={(e) => setMinGreen(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Minimum time a green phase must remain before a switch is permitted.</p>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Maximum Green Duration (seconds)</label>
              <input
                type="number"
                step="5"
                min={20}
                max={120}
                value={maxGreen}
                onChange={(e) => setMaxGreen(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Forced termination interval to prevent approach starvation.</p>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Yellow Clearance Interval (seconds)</label>
              <input
                type="number"
                step="0.5"
                min={1}
                max={6}
                value={yellowDuration}
                onChange={(e) => setYellowDuration(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Clearance window for vehicles approaching the intersection.</p>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">All-Red Buffer (seconds)</label>
              <input
                type="number"
                step="0.5"
                min={1}
                max={5}
                value={allRedDuration}
                onChange={(e) => setAllRedDuration(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Safety clearance ensuring intersection box is completely clear.</p>
            </div>
          </div>
        </div>

        {/* Reward Formulation Weights */}
        <div className="glass-card rounded-2xl p-6 border border-white/10 space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-white flex items-center gap-2">
            <Award size={16} className="text-amber-400" />
            Reward Function Parameter Weights
          </h2>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Queue Length Weight (w_q)</label>
              <input
                type="number"
                step="0.1"
                min={0}
                max={5}
                value={queueWeight}
                onChange={(e) => setQueueWeight(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Penalizes total number of queued vehicles.</p>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Waiting Time Weight (w_w)</label>
              <input
                type="number"
                step="0.1"
                min={0}
                max={5}
                value={waitingWeight}
                onChange={(e) => setWaitingWeight(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Penalizes incremental seconds vehicles spend idling.</p>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Signal Switch Penalty (w_s)</label>
              <input
                type="number"
                step="0.1"
                min={0}
                max={5}
                value={switchWeight}
                onChange={(e) => setSwitchWeight(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Penalizes rapid and unnecessary phase oscillations.</p>
            </div>

            <div className="space-y-1">
              <label className="text-xs font-semibold text-slate-300">Throughput Reward (w_th)</label>
              <input
                type="number"
                step="0.1"
                min={0}
                max={10}
                value={throughputWeight}
                onChange={(e) => setThroughputWeight(Number(e.target.value))}
                className="w-full bg-slate-800 text-xs text-white px-3 py-2 rounded-xl border border-white/10 font-mono"
              />
              <p className="text-[10px] text-slate-500">Incentivizes successful completion and departure of vehicles.</p>
            </div>
          </div>
        </div>

        {/* Buttons */}
        <div className="flex justify-end gap-3">
          <button
            type="button"
            onClick={handleResetDefaults}
            className="flex items-center gap-1.5 px-4 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold border border-white/10 transition-all"
          >
            <RotateCcw size={15} />
            Reset Defaults
          </button>

          <button
            type="submit"
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-xs font-bold shadow-lg shadow-blue-600/30 transition-all hover:scale-[1.02]"
          >
            <Save size={15} />
            Save Configuration
          </button>
        </div>
      </form>
    </div>
  );
};
