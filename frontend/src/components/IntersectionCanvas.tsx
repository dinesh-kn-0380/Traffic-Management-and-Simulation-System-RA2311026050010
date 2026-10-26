import React, { useRef, useEffect, useState } from 'react';
import { SimulationState, Vehicle } from '../types';

interface IntersectionCanvasProps {
  simulationState: SimulationState | null;
}

export const IntersectionCanvas: React.FC<IntersectionCanvasProps> = ({ simulationState }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hoveredVehicle, setHoveredVehicle] = useState<Vehicle | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Fixed canvas size for crisp rendering
    const width = 800;
    const height = 800;
    const center = 400;
    const roadWidth = 140;
    const halfRoad = roadWidth / 2; // 70
    const laneOffset = 35; // Center of 70px lane

    // Background terrain
    ctx.fillStyle = '#0F172A';
    ctx.fillRect(0, 0, width, height);

    // Subtle terrain grid pattern
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Asphalt roads
    ctx.fillStyle = '#1E293B';
    // North-South Road
    ctx.fillRect(center - halfRoad, 0, roadWidth, height);
    // East-West Road
    ctx.fillRect(0, center - halfRoad, width, roadWidth);

    // Curbs & Road borders
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 3;
    // Top-Left corner
    ctx.strokeRect(0, 0, center - halfRoad, center - halfRoad);
    // Top-Right corner
    ctx.strokeRect(center + halfRoad, 0, center - halfRoad, center - halfRoad);
    // Bottom-Left corner
    ctx.strokeRect(0, center + halfRoad, center - halfRoad, center - halfRoad);
    // Bottom-Right corner
    ctx.strokeRect(center + halfRoad, center + halfRoad, center - halfRoad, center - halfRoad);

    // Crosswalk zebra stripes
    ctx.fillStyle = 'rgba(255, 255, 255, 0.35)';
    const zebraW = 20;
    const stripeCount = 7;
    // North Crosswalk (y: center - halfRoad - zebraW to center - halfRoad)
    for (let i = 0; i < stripeCount; i++) {
      ctx.fillRect(center - halfRoad + 6 + i * 18, center - halfRoad - zebraW, 10, zebraW);
    }
    // South Crosswalk
    for (let i = 0; i < stripeCount; i++) {
      ctx.fillRect(center - halfRoad + 6 + i * 18, center + halfRoad, 10, zebraW);
    }
    // West Crosswalk
    for (let i = 0; i < stripeCount; i++) {
      ctx.fillRect(center - halfRoad - zebraW, center - halfRoad + 6 + i * 18, zebraW, 10);
    }
    // East Crosswalk
    for (let i = 0; i < stripeCount; i++) {
      ctx.fillRect(center + halfRoad, center - halfRoad + 6 + i * 18, zebraW, 10);
    }

    // Stop Lines (Solid White)
    ctx.fillStyle = '#F8FAFC';
    // North Stop Line (Inbound is on left side x: center - halfRoad to center)
    ctx.fillRect(center - halfRoad, center - halfRoad - 2, halfRoad, 4);
    // South Stop Line (Inbound is on right side x: center to center + halfRoad)
    ctx.fillRect(center, center + halfRoad - 2, halfRoad, 4);
    // West Stop Line (Inbound is on bottom side y: center to center + halfRoad)
    ctx.fillRect(center - halfRoad - 2, center, 4, halfRoad);
    // East Stop Line (Inbound is on top side y: center - halfRoad to center)
    ctx.fillRect(center + halfRoad - 2, center - halfRoad, 4, halfRoad);

    // Double Yellow Center Dividers
    ctx.strokeStyle = '#FBBF24';
    ctx.lineWidth = 2;
    // North Road Divider
    ctx.beginPath();
    ctx.moveTo(center - 2, 0);
    ctx.lineTo(center - 2, center - halfRoad - zebraW);
    ctx.moveTo(center + 2, 0);
    ctx.lineTo(center + 2, center - halfRoad - zebraW);
    ctx.stroke();

    // South Road Divider
    ctx.beginPath();
    ctx.moveTo(center - 2, center + halfRoad + zebraW);
    ctx.lineTo(center - 2, height);
    ctx.moveTo(center + 2, center + halfRoad + zebraW);
    ctx.lineTo(center + 2, height);
    ctx.stroke();

    // West Road Divider
    ctx.beginPath();
    ctx.moveTo(0, center - 2);
    ctx.lineTo(center - halfRoad - zebraW, center - 2);
    ctx.moveTo(0, center + 2);
    ctx.lineTo(center - halfRoad - zebraW, center + 2);
    ctx.stroke();

    // East Road Divider
    ctx.beginPath();
    ctx.moveTo(center + halfRoad + zebraW, center - 2);
    ctx.lineTo(width, center - 2);
    ctx.moveTo(center + halfRoad + zebraW, center + 2);
    ctx.lineTo(width, center + 2);
    ctx.stroke();

    // Intersection Center Markings
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.setLineDash([6, 6]);
    ctx.strokeRect(center - halfRoad, center - halfRoad, roadWidth, roadWidth);
    ctx.setLineDash([]);

    if (!simulationState) return;

    // Scale calculation:
    // 0 to 100 meters maps to 0 to (center - halfRoad) = 330px.
    // Scale factor = 330 / 100 = 3.3 px/meter.
    const pxPerMeter = (center - halfRoad) / simulationState.geometry.stop_line_pos;

    // Draw Traffic Lights
    const drawSignalHead = (x: number, y: number, state: 'RED' | 'YELLOW' | 'GREEN', label: string) => {
      // Housing
      ctx.fillStyle = '#0F172A';
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.roundRect(x - 16, y - 44, 32, 88, 8);
      ctx.fill();
      ctx.stroke();

      // Label
      ctx.fillStyle = '#94A3B8';
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(label, x, y - 48);

      const lights: Array<{ color: 'RED' | 'YELLOW' | 'GREEN'; hex: string; activeHex: string; cy: number }> = [
        { color: 'RED', hex: '#450A0A', activeHex: '#EF4444', cy: y - 26 },
        { color: 'YELLOW', hex: '#451A03', activeHex: '#F59E0B', cy: y },
        { color: 'GREEN', hex: '#022C22', activeHex: '#10B981', cy: y + 26 },
      ];

      for (const light of lights) {
        const isActive = state === light.color;
        ctx.beginPath();
        ctx.arc(x, light.cy, 10, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? light.activeHex : light.hex;
        ctx.fill();

        if (isActive) {
          // Glow bloom
          ctx.save();
          ctx.shadowColor = light.activeHex;
          ctx.shadowBlur = 18;
          ctx.beginPath();
          ctx.arc(x, light.cy, 10, 0, Math.PI * 2);
          ctx.fillStyle = light.activeHex;
          ctx.fill();
          ctx.restore();
        }
      }
    };

    const sigs = simulationState.signals;
    drawSignalHead(center - halfRoad - 28, center - halfRoad - 50, sigs.N, 'NORTH');
    drawSignalHead(center + halfRoad + 28, center + halfRoad + 50, sigs.S, 'SOUTH');
    drawSignalHead(center + halfRoad + 50, center - halfRoad - 28, sigs.E, 'EAST');
    drawSignalHead(center - halfRoad - 50, center + halfRoad + 28, sigs.W, 'WEST');

    // Draw Vehicles
    for (const v of simulationState.vehicles) {
      if (v.has_departed) continue;

      let vx = 0;
      let vy = 0;
      let angle = 0;

      // Position in pixels relative to entry
      const p = v.position * pxPerMeter;

      if (v.approach === 'N') {
        // Heading South along left side of road
        vx = center - laneOffset;
        vy = p;
        angle = Math.PI / 2; // facing down
      } else if (v.approach === 'S') {
        // Heading North along right side of road
        vx = center + laneOffset;
        vy = height - p;
        angle = -Math.PI / 2; // facing up
      } else if (v.approach === 'E') {
        // Heading West along top side of road
        vx = width - p;
        vy = center - laneOffset;
        angle = Math.PI; // facing left
      } else if (v.approach === 'W') {
        // Heading East along bottom side of road
        vx = p;
        vy = center + laneOffset;
        angle = 0; // facing right
      }

      ctx.save();
      ctx.translate(vx, vy);
      ctx.rotate(angle);

      // Car dimensions
      const carLen = 24;
      const carWid = 14;

      // Shadow
      ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
      ctx.beginPath();
      ctx.roundRect(-carLen / 2 + 2, -carWid / 2 + 2, carLen, carWid, 4);
      ctx.fill();

      // Car Body
      ctx.fillStyle = v.color;
      ctx.beginPath();
      ctx.roundRect(-carLen / 2, -carWid / 2, carLen, carWid, 4);
      ctx.fill();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Windshield & Roof
      ctx.fillStyle = 'rgba(15, 23, 42, 0.8)';
      ctx.fillRect(-carLen / 4, -carWid / 2 + 2, carLen / 2, carWid - 4);

      // Headlights (Front: +x)
      ctx.fillStyle = '#FEF08A';
      ctx.fillRect(carLen / 2 - 2, -carWid / 2 + 1, 2, 3);
      ctx.fillRect(carLen / 2 - 2, carWid / 2 - 4, 2, 3);

      // Taillights (Rear: -x)
      ctx.fillStyle = v.is_queued ? '#EF4444' : '#991B1B';
      ctx.fillRect(-carLen / 2, -carWid / 2 + 1, 2, 3);
      ctx.fillRect(-carLen / 2, carWid / 2 - 4, 2, 3);

      // Brake light intense glow when queued
      if (v.is_queued) {
        ctx.shadowColor = '#EF4444';
        ctx.shadowBlur = 8;
        ctx.fillRect(-carLen / 2 - 1, -carWid / 2 + 1, 2, 3);
        ctx.fillRect(-carLen / 2 - 1, carWid / 2 - 4, 2, 3);
      }

      ctx.restore();
    }
  }, [simulationState]);

  return (
    <div className="relative w-full max-w-[800px] aspect-square mx-auto rounded-2xl overflow-hidden shadow-2xl border border-white/10 bg-[#0F172A]">
      <canvas
        ref={canvasRef}
        width={800}
        height={800}
        className="w-full h-full block"
      />

      {/* Floating Approach HUD Badges */}
      {simulationState && (
        <>
          {/* North HUD */}
          <div className="absolute top-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 shadow-lg text-xs">
            <span className="w-2 h-2 rounded-full bg-blue-400"></span>
            <span className="font-semibold text-slate-200">North:</span>
            <span className="text-amber-400 font-mono font-bold">Q: {simulationState.queues.N}</span>
            <span className="text-slate-400 font-mono">({simulationState.aggregate_waits.N}s)</span>
          </div>

          {/* South HUD */}
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 bg-slate-900/85 backdrop-blur-md px-3 py-1.5 rounded-full border border-white/10 shadow-lg text-xs">
            <span className="w-2 h-2 rounded-full bg-blue-400"></span>
            <span className="font-semibold text-slate-200">South:</span>
            <span className="text-amber-400 font-mono font-bold">Q: {simulationState.queues.S}</span>
            <span className="text-slate-400 font-mono">({simulationState.aggregate_waits.S}s)</span>
          </div>

          {/* West HUD */}
          <div className="absolute left-4 top-1/2 -translate-y-1/2 flex flex-col items-center gap-1 bg-slate-900/85 backdrop-blur-md px-2.5 py-2 rounded-xl border border-white/10 shadow-lg text-xs">
            <span className="font-semibold text-slate-200">West</span>
            <span className="text-amber-400 font-mono font-bold">Q: {simulationState.queues.W}</span>
            <span className="text-slate-400 font-mono">{simulationState.aggregate_waits.W}s</span>
          </div>

          {/* East HUD */}
          <div className="absolute right-4 top-1/2 -translate-y-1/2 flex flex-col items-center gap-1 bg-slate-900/85 backdrop-blur-md px-2.5 py-2 rounded-xl border border-white/10 shadow-lg text-xs">
            <span className="font-semibold text-slate-200">East</span>
            <span className="text-amber-400 font-mono font-bold">Q: {simulationState.queues.E}</span>
            <span className="text-slate-400 font-mono">{simulationState.aggregate_waits.E}s</span>
          </div>

          {/* Center Intersection Phase Indicator */}
          <div className="absolute top-4 left-4 flex flex-col gap-1 bg-slate-900/90 backdrop-blur-md p-3 rounded-xl border border-white/10 shadow-xl max-w-[210px]">
            <div className="text-[10px] font-bold tracking-wider text-slate-400 uppercase">Current Phase</div>
            <div className={`text-xs font-bold px-2 py-0.5 rounded text-center ${
              simulationState.is_transitioning ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40' :
              simulationState.current_phase === 0 ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' :
              'bg-blue-500/20 text-blue-400 border border-blue-500/40'
            }`}>
              {simulationState.phase_name}
            </div>
            <div className="flex justify-between items-center text-[11px] text-slate-300 mt-1">
              <span>Elapsed:</span>
              <span className="font-mono font-bold">{simulationState.phase_elapsed}s</span>
            </div>
            <div className="flex justify-between items-center text-[11px] text-slate-300">
              <span>Departures:</span>
              <span className="font-mono text-emerald-400 font-bold">{simulationState.completed_count}</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
