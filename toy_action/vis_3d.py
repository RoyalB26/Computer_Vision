"""Interactive 3D Attention Volume HTML Exporter for Toy Action Inference.

Generates a standalone, zero-dependency offline HTML file visualizing the
spatiotemporal slot attention volume `(T, S, H, W)` as colored 3D voxels and
3D centroid trajectories with interactive orbit/zoom, per-slot toggles,
threshold filtering, and frame-by-frame playback.
"""

import base64
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Interactive 3D Slot Attention Volume</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    height: 100vh;
    display: flex;
    overflow: hidden;
  }
  #sidebar {
    width: 340px;
    min-width: 300px;
    background: #1e293b;
    border-right: 1px solid #334155;
    padding: 14px 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  h1 { font-size: 15px; font-weight: 700; color: #f8fafc; letter-spacing: 0.2px; }
  h2 {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #94a3b8;
    margin-bottom: 6px;
  }
  .card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 10px 12px;
  }
  .meta-line { font-size: 12px; line-height: 1.5; color: #cbd5e1; word-break: break-word; }
  .meta-line strong { color: #38bdf8; }
  .gt-line strong { color: #4ade80; }
  .btn-row { display: flex; flex-wrap: wrap; gap: 6px; }
  button {
    background: #334155;
    color: #f1f5f9;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  button:hover { background: #475569; }
  button.primary { background: #0284c7; border-color: #0ea5e9; }
  button.primary:hover { background: #0369a1; }
  .slider-row { margin-bottom: 8px; }
  .slider-header {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    margin-bottom: 3px;
    color: #cbd5e1;
  }
  input[type="range"] { width: 100%; accent-color: #38bdf8; cursor: pointer; }
  .slot-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 6px;
    border-radius: 6px;
    font-size: 12px;
    cursor: pointer;
    user-select: none;
  }
  .slot-item:hover { background: #1e293b; }
  .swatch {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid rgba(255,255,255,0.4);
    flex-shrink: 0;
  }
  .slot-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .slot-badge {
    font-size: 10px;
    padding: 1px 5px;
    border-radius: 4px;
    background: #1e293b;
    color: #94a3b8;
  }
  .check-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #cbd5e1;
    margin-top: 6px;
    cursor: pointer;
  }
  #frame-preview-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 8px;
  }
  #frame-preview {
    width: 148px;
    height: auto;
    max-height: 128px;
    border-radius: 6px;
    border: 1px solid #475569;
    background: #020617;
    object-fit: contain;
  }
  #viewport {
    flex: 1;
    position: relative;
    overflow: hidden;
    background: radial-gradient(circle at center, #172033 0%, #090e1a 100%);
  }
  canvas { display: block; width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #hud {
    position: absolute;
    top: 12px;
    left: 14px;
    background: rgba(15, 23, 42, 0.78);
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 11px;
    color: #94a3b8;
    pointer-events: none;
    backdrop-filter: blur(4px);
  }
  #tooltip {
    position: absolute;
    display: none;
    pointer-events: none;
    background: rgba(2, 6, 23, 0.92);
    border: 1px solid #38bdf8;
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 11px;
    color: #f8fafc;
    white-space: nowrap;
    z-index: 20;
  }
  .top-prob-row {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #cbd5e1;
    padding: 2px 0;
  }
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>3D Slot Attention Explorer</h1>
    <div style="font-size:11px;color:#64748b;margin-top:2px;">(X, Y, Time) Spatiotemporal Color Cube</div>
  </div>

  <div class="card">
    <h2>Predictions</h2>
    <div class="meta-line"><strong>Pred:</strong> <span id="pred-str"></span></div>
    <div class="meta-line gt-line" id="gt-wrap" style="display:none;"><strong>GT:</strong> <span id="gt-str"></span></div>
    <div id="top-probs" style="margin-top:6px;border-top:1px solid #1e293b;padding-top:4px;"></div>
  </div>

  <div class="card">
    <h2>Camera Views</h2>
    <div class="btn-row">
      <button class="primary" onclick="setView(-0.68, 0.48)">Iso 3D</button>
      <button onclick="setView(0.0, 0.0)">Front (X-Y)</button>
      <button onclick="setView(0.0, 1.56)">Top (X-T)</button>
      <button onclick="setView(-1.57, 0.0)">Side (T-Y)</button>
      <button onclick="resetCamera()">Reset</button>
    </div>
  </div>

  <div class="card">
    <h2>Slot Colors</h2>
    <div class="btn-row" style="margin-bottom:6px;">
      <button onclick="setAllSlots(true)">All</button>
      <button onclick="selectActiveSlots()">Active Only</button>
      <button onclick="setAllSlots(false)">None</button>
    </div>
    <div id="slot-list"></div>
  </div>

  <div class="card">
    <h2>Volume Controls</h2>
    <div class="slider-row">
      <div class="slider-header"><span>Attention Threshold</span><span id="val-thresh">0.12</span></div>
      <input id="sl-thresh" type="range" min="0.03" max="0.85" step="0.01" value="0.12" />
    </div>
    <div class="slider-row">
      <div class="slider-header"><span>Voxel Size</span><span id="val-size">0.82</span></div>
      <input id="sl-size" type="range" min="0.35" max="1.00" step="0.05" value="0.82" />
    </div>
    <div class="slider-row">
      <div class="slider-header"><span>Opacity</span><span id="val-alpha">0.85</span></div>
      <input id="sl-alpha" type="range" min="0.20" max="1.00" step="0.05" value="0.85" />
    </div>
    <label class="check-row"><input id="chk-scale" type="checkbox" checked /> Scale Size &amp; Brightness by Attn</label>
    <label class="check-row"><input id="chk-autorange" type="checkbox" checked /> Auto-Normalize Peak Range</label>
    <label class="check-row"><input id="chk-traj" type="checkbox" checked /> Show 3D Centroid Trajectories</label>
    <label class="check-row"><input id="chk-winner" type="checkbox" /> Winner-Takes-All per Voxel</label>
    <div style="margin-top:8px;font-size:11px;color:#94a3b8;">
      <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
        <span>Low (<span id="range-min">0.12</span>)</span>
        <span>High (<span id="range-max">1.00</span>)</span>
      </div>
      <div style="height:8px;border-radius:4px;background:linear-gradient(90deg,#1e293b 0%,#38bdf8 60%,#f8fafc 100%);border:1px solid #334155;"></div>
    </div>
  </div>

  <div class="card">
    <h2>Time Slice (Frame t)</h2>
    <div class="slider-row">
      <div class="slider-header"><span>Frame</span><span id="val-frame">All (0..15)</span></div>
      <input id="sl-frame" type="range" min="-1" max="15" step="1" value="-1" />
    </div>
    <div class="btn-row">
      <button class="primary" id="btn-play" onclick="togglePlay()">Play</button>
      <button onclick="setFrame(-1)">Show All</button>
    </div>
    <label class="check-row"><input id="chk-isolate" type="checkbox" /> Isolate Selected Frame Only</label>
    <div id="frame-preview-wrap" style="display:none;">
      <img id="frame-preview" alt="Frame preview" />
      <div style="font-size:11px;color:#94a3b8;" id="frame-preview-caption"></div>
    </div>
  </div>
</div>

<div id="viewport">
  <canvas id="canvas"></canvas>
  <div id="hud">Drag: Rotate 360° &nbsp;|&nbsp; Wheel: Zoom &nbsp;|&nbsp; Right-Drag / Shift+Drag: Pan &nbsp;|&nbsp; Hover voxel for details</div>
  <div id="tooltip"></div>
</div>

<script id="vol-data" type="application/json">__VOLUME_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById("vol-data").textContent);
const T = DATA.T, H = DATA.H, W = DATA.W;

// State
const state = {
  yaw: -0.68,
  pitch: 0.48,
  zoom: 1.0,
  panX: 0,
  panY: 0,
  thresh: 0.12,
  voxelScale: 0.82,
  opacity: 0.85,
  showTraj: true,
  winnerOnly: false,
  scaleByAttn: true,
  autoNormalize: true,
  selectedFrame: -1,
  isolateFrame: false,
  playing: false,
  playTimer: null,
  enabledSlots: new Set(DATA.slots.filter(s => s.active).map(s => s.id)),
  renderedBoxes: [],
};

// Populate metadata & controls
document.getElementById("pred-str").textContent = DATA.pred_labels || "(None above threshold)";
if (DATA.gt_label) {
  document.getElementById("gt-wrap").style.display = "block";
  document.getElementById("gt-str").textContent = DATA.gt_label;
}
const topProbsEl = document.getElementById("top-probs");
(DATA.top_preds || []).slice(0, 5).forEach(([name, pct]) => {
  const row = document.createElement("div");
  row.className = "top-prob-row";
  row.innerHTML = `<span>${name}</span><span>${pct.toFixed(1)}%</span>`;
  topProbsEl.appendChild(row);
});

const slFrame = document.getElementById("sl-frame");
slFrame.max = String(T - 1);

const slotListEl = document.getElementById("slot-list");
const slotCheckboxes = new Map();
DATA.slots.forEach(slot => {
  const label = document.createElement("label");
  label.className = "slot-item";
  const cb = document.createElement("input");
  cb.type = "checkbox";
  cb.checked = state.enabledSlots.has(slot.id);
  cb.onchange = () => {
    if (cb.checked) state.enabledSlots.add(slot.id);
    else state.enabledSlots.delete(slot.id);
    render();
  };
  slotCheckboxes.set(slot.id, cb);

  const sw = document.createElement("span");
  sw.className = "swatch";
  sw.style.background = `rgb(${slot.color[0]},${slot.color[1]},${slot.color[2]})`;

  const txt = document.createElement("span");
  txt.className = "slot-text";
  txt.textContent = `S${String(slot.id).padStart(2, "0")} [${slot.symbol}]: ${slot.label}`;
  txt.title = txt.textContent;

  const badge = document.createElement("span");
  badge.className = "slot-badge";
  badge.textContent = `${slot.voxels.length}`;

  label.appendChild(cb);
  label.appendChild(sw);
  label.appendChild(txt);
  label.appendChild(badge);
  slotListEl.appendChild(label);
});

function setAllSlots(on) {
  DATA.slots.forEach(s => {
    if (on) state.enabledSlots.add(s.id);
    else state.enabledSlots.delete(s.id);
    slotCheckboxes.get(s.id).checked = on;
  });
  render();
}
function selectActiveSlots() {
  DATA.slots.forEach(s => {
    if (s.active) state.enabledSlots.add(s.id);
    else state.enabledSlots.delete(s.id);
    slotCheckboxes.get(s.id).checked = s.active;
  });
  render();
}

// Slider bindings
document.getElementById("sl-thresh").oninput = e => {
  state.thresh = parseFloat(e.target.value);
  document.getElementById("val-thresh").textContent = state.thresh.toFixed(2);
  render();
};
document.getElementById("sl-size").oninput = e => {
  state.voxelScale = parseFloat(e.target.value);
  document.getElementById("val-size").textContent = state.voxelScale.toFixed(2);
  render();
};
document.getElementById("sl-alpha").oninput = e => {
  state.opacity = parseFloat(e.target.value);
  document.getElementById("val-alpha").textContent = state.opacity.toFixed(2);
  render();
};
document.getElementById("chk-scale").onchange = e => { state.scaleByAttn = e.target.checked; render(); };
document.getElementById("chk-autorange").onchange = e => { state.autoNormalize = e.target.checked; render(); };
document.getElementById("chk-traj").onchange = e => { state.showTraj = e.target.checked; render(); };
document.getElementById("chk-winner").onchange = e => { state.winnerOnly = e.target.checked; render(); };
document.getElementById("chk-isolate").onchange = e => { state.isolateFrame = e.target.checked; render(); };
slFrame.oninput = e => setFrame(parseInt(e.target.value, 10));

function setFrame(f) {
  state.selectedFrame = f;
  slFrame.value = String(f);
  document.getElementById("val-frame").textContent = f < 0 ? `All (0..${T - 1})` : `F${String(f).padStart(2, "0")}`;
  const wrap = document.getElementById("frame-preview-wrap");
  if (f >= 0 && DATA.frames_b64 && DATA.frames_b64[f]) {
    wrap.style.display = "flex";
    document.getElementById("frame-preview").src = DATA.frames_b64[f];
    document.getElementById("frame-preview-caption").textContent = `Frame F${String(f).padStart(2, "0")} / ${T - 1}`;
  } else {
    wrap.style.display = "none";
  }
  render();
}

function togglePlay() {
  state.playing = !state.playing;
  const btn = document.getElementById("btn-play");
  if (state.playing) {
    btn.textContent = "Pause";
    if (state.selectedFrame < 0) setFrame(0);
    state.playTimer = setInterval(() => {
      setFrame((state.selectedFrame + 1) % T);
    }, 220);
  } else {
    btn.textContent = "Play";
    clearInterval(state.playTimer);
  }
}

function setView(yaw, pitch) { state.yaw = yaw; state.pitch = pitch; render(); }
function resetCamera() { state.yaw = -0.68; state.pitch = 0.48; state.zoom = 1.0; state.panX = 0; state.panY = 0; render(); }

// Canvas & 3D Math
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const tooltip = document.getElementById("tooltip");

function resize() {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}
window.addEventListener("resize", resize);

// Normalize volume coordinates to [-1, 1] box preserving aspect
const maxDim = Math.max(W, H, T);
const sx = W / maxDim, sy = H / maxDim, st = T / maxDim;

function gridToWorld(x, y, t) {
  return [
    ((x + 0.5) / W - 0.5) * 2.0 * sx,
    ((y + 0.5) / H - 0.5) * 2.0 * sy,
    ((t + 0.5) / T - 0.5) * 2.0 * st,
  ];
}
function cornerToWorld(x, y, t) {
  return [
    (x / W - 0.5) * 2.0 * sx,
    (y / H - 0.5) * 2.0 * sy,
    (t / T - 0.5) * 2.0 * st,
  ];
}

function project(wx, wy, wz, cw, ch) {
  const cy = Math.cos(state.yaw), sy_ = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  // Rotate around Y-axis (yaw), then X-axis (pitch)
  const x1 = wx * cy + wz * sy_;
  const z1 = -wx * sy_ + wz * cy;
  const y2 = wy * cp - z1 * sp;
  const z2 = wy * sp + z1 * cp;

  const camDist = 4.2;
  const persp = camDist / Math.max(0.6, camDist - z2);
  const baseScale = Math.min(cw, ch) * 0.34 * state.zoom;
  return {
    sx: cw * 0.5 + state.panX + x1 * baseScale * persp,
    sy: ch * 0.5 + state.panY + y2 * baseScale * persp,
    z: z2,
    persp: persp,
  };
}

function shadeColor(rgb, factor, normVal, alpha) {
  let rBase = rgb[0], gBase = rgb[1], bBase = rgb[2];
  if (state.scaleByAttn) {
    // Low attn: muted dark tint; high attn: vivid bright core
    const sat = 0.25 + 0.75 * normVal;
    const boost = Math.max(0.0, normVal - 0.72) / 0.28 * 0.25;
    rBase = rgb[0] * sat + (255 - rgb[0] * sat) * boost;
    gBase = rgb[1] * sat + (255 - rgb[1] * sat) * boost;
    bBase = rgb[2] * sat + (255 - rgb[2] * sat) * boost;
  }
  const r = Math.min(255, Math.round(rBase * factor));
  const g = Math.min(255, Math.round(gBase * factor));
  const b = Math.min(255, Math.round(bBase * factor));
  return `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
}

// Cube face definitions (vertex indices into 8 corners)
const CUBE_FACES = [
  { idx: [0, 1, 3, 2], norm: [0, 0, -1], shade: 0.76 }, // -Z (back)
  { idx: [4, 5, 7, 6], norm: [0, 0, 1], shade: 0.98 },  // +Z (front)
  { idx: [0, 4, 6, 2], norm: [-1, 0, 0], shade: 0.70 }, // -X (left)
  { idx: [1, 5, 7, 3], norm: [1, 0, 0], shade: 0.88 },  // +X (right)
  { idx: [0, 1, 5, 4], norm: [0, -1, 0], shade: 1.05 }, // -Y (top)
  { idx: [2, 3, 7, 6], norm: [0, 1, 0], shade: 0.62 },  // +Y (bottom)
];

function isFaceVisible(norm) {
  const cy = Math.cos(state.yaw), sy_ = Math.sin(state.yaw);
  const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
  const nx1 = norm[0] * cy + norm[2] * sy_;
  const nz1 = -norm[0] * sy_ + norm[2] * cy;
  const nz2 = norm[1] * sp + nz1 * cp;
  return nz2 > 0.0; // Camera looks toward -Z from +Z
}

function render() {
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (cw <= 0 || ch <= 0) return;
  ctx.clearRect(0, 0, cw, ch);

  // 1. Draw 3D Bounding Box + Axis Labels
  const boxCorners = [
    cornerToWorld(0, 0, 0), cornerToWorld(W, 0, 0),
    cornerToWorld(0, H, 0), cornerToWorld(W, H, 0),
    cornerToWorld(0, 0, T), cornerToWorld(W, 0, T),
    cornerToWorld(0, H, T), cornerToWorld(W, H, T),
  ].map(p => project(p[0], p[1], p[2], cw, ch));

  const edges = [
    [0,1],[1,3],[3,2],[2,0],
    [4,5],[5,7],[7,6],[6,4],
    [0,4],[1,5],[2,6],[3,7],
  ];
  ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  edges.forEach(([a, b]) => {
    ctx.moveTo(boxCorners[a].sx, boxCorners[a].sy);
    ctx.lineTo(boxCorners[b].sx, boxCorners[b].sy);
  });
  ctx.stroke();

  // Highlighted axis lines from origin (0, 0, 0)
  const drawAxis = (p0, p1, color, label) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(p0.sx, p0.sy);
    ctx.lineTo(p1.sx, p1.sy);
    ctx.stroke();
    ctx.fillStyle = color;
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(label, p1.sx + 5, p1.sy + 4);
  };
  drawAxis(boxCorners[0], boxCorners[1], "#38bdf8", `X (0..${W - 1})`);
  drawAxis(boxCorners[0], boxCorners[2], "#fbbf24", `Y (0..${H - 1})`);
  drawAxis(boxCorners[0], boxCorners[4], "#c084fc", `T (0..${T - 1})`);

  // Draw active time slice plane if a frame is selected
  if (state.selectedFrame >= 0) {
    const tf = state.selectedFrame + 0.5;
    const sp = [
      cornerToWorld(0, 0, tf), cornerToWorld(W, 0, tf),
      cornerToWorld(W, H, tf), cornerToWorld(0, H, tf),
    ].map(p => project(p[0], p[1], p[2], cw, ch));
    ctx.fillStyle = "rgba(56, 189, 248, 0.07)";
    ctx.strokeStyle = "rgba(56, 189, 248, 0.8)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(sp[0].sx, sp[0].sy);
    for (let i = 1; i < 4; i++) ctx.lineTo(sp[i].sx, sp[i].sy);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }

  // 2. Collect visible voxels
  const rawItems = [];
  const winnerMap = state.winnerOnly ? new Map() : null;

  DATA.slots.forEach(slot => {
    if (!state.enabledSlots.has(slot.id)) return;
    for (let i = 0; i < slot.voxels.length; i++) {
      const [t, y, x, val] = slot.voxels[i];
      if (val < state.thresh) continue;
      if (state.isolateFrame && state.selectedFrame >= 0 && t !== state.selectedFrame) continue;
      if (winnerMap) {
        const key = `${t},${y},${x}`;
        const prev = winnerMap.get(key);
        if (!prev || val > prev.val) {
          winnerMap.set(key, { slot, t, y, x, val });
        }
      } else {
        rawItems.push({ slot, t, y, x, val });
      }
    }
  });
  const items = winnerMap ? Array.from(winnerMap.values()) : rawItems;

  // Depth sort back-to-front
  const dxHalf = (1.0 / W) * sx * state.voxelScale;
  const dyHalf = (1.0 / H) * sy * state.voxelScale;
  const dtHalf = (1.0 / T) * st * state.voxelScale;
  const visibleFaces = CUBE_FACES.filter(f => isFaceVisible(f.norm));

  const drawList = items.map(it => {
    const [wx, wy, wz] = gridToWorld(it.x, it.y, it.t);
    const centerP = project(wx, wy, wz, cw, ch);
    return { it, wx, wy, wz, z: centerP.z };
  });
  drawList.sort((a, b) => a.z - b.z);

  state.renderedBoxes = [];
  let maxVal = 1.0;
  if (state.autoNormalize && items.length > 0) {
    maxVal = 0.0;
    for (let i = 0; i < items.length; i++) {
      if (items[i].val > maxVal) maxVal = items[i].val;
    }
    maxVal = Math.max(state.thresh + 0.05, maxVal);
  }
  document.getElementById("range-min").textContent = state.thresh.toFixed(2);
  document.getElementById("range-max").textContent = maxVal.toFixed(2);
  const denom = Math.max(0.05, maxVal - state.thresh);

  for (let i = 0; i < drawList.length; i++) {
    const { it, wx, wy, wz } = drawList[i];
    const normVal = Math.min(1.0, Math.max(0.0, (it.val - state.thresh) / denom));
    const intensity = Math.pow(normVal, 0.85);
    let alpha = state.scaleByAttn
      ? (0.06 + 0.90 * intensity) * state.opacity
      : (0.25 + 0.70 * normVal) * state.opacity;
    const isCurrentSlice = state.selectedFrame >= 0 && it.t === state.selectedFrame;
    if (state.selectedFrame >= 0 && !isCurrentSlice) {
      alpha *= 0.22;
    }

    const sizeMult = state.scaleByAttn ? (0.26 + 0.74 * intensity) : 1.0;
    const vx = dxHalf * sizeMult;
    const vy = dyHalf * sizeMult;
    const vt = dtHalf * sizeMult;

    const corners = [
      [wx - vx, wy - vy, wz - vt],
      [wx + vx, wy - vy, wz - vt],
      [wx - vx, wy + vy, wz - vt],
      [wx + vx, wy + vy, wz - vt],
      [wx - vx, wy - vy, wz + vt],
      [wx + vx, wy - vy, wz + vt],
      [wx - vx, wy + vy, wz + vt],
      [wx + vx, wy + vy, wz + vt],
    ].map(c => project(c[0], c[1], c[2], cw, ch));

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let k = 0; k < 8; k++) {
      if (corners[k].sx < minX) minX = corners[k].sx;
      if (corners[k].sx > maxX) maxX = corners[k].sx;
      if (corners[k].sy < minY) minY = corners[k].sy;
      if (corners[k].sy > maxY) maxY = corners[k].sy;
    }
    state.renderedBoxes.push({ minX, maxX, minY, maxY, it });

    for (let f = 0; f < visibleFaces.length; f++) {
      const face = visibleFaces[f];
      const p0 = corners[face.idx[0]];
      const p1 = corners[face.idx[1]];
      const p2 = corners[face.idx[2]];
      const p3 = corners[face.idx[3]];
      ctx.fillStyle = shadeColor(it.slot.color, face.shade, intensity, alpha);
      ctx.beginPath();
      ctx.moveTo(p0.sx, p0.sy);
      ctx.lineTo(p1.sx, p1.sy);
      ctx.lineTo(p2.sx, p2.sy);
      ctx.lineTo(p3.sx, p3.sy);
      ctx.closePath();
      ctx.fill();
      if (isCurrentSlice || normVal > 0.55) {
        ctx.strokeStyle = isCurrentSlice ? "rgba(255,255,255,0.65)" : "rgba(15,23,42,0.35)";
        ctx.lineWidth = isCurrentSlice ? 1.1 : 0.6;
        ctx.stroke();
      }
    }
  }

  // 3. Draw 3D Centroid Trajectories
  if (state.showTraj) {
    DATA.slots.forEach(slot => {
      if (!state.enabledSlots.has(slot.id)) return;
      const pts = [];
      (slot.centroids || []).forEach(([t, cx, cy, peak]) => {
        if (peak >= state.thresh) {
          const [wx, wy, wz] = gridToWorld(cx, cy, t);
          pts.push({ t, p: project(wx, wy, wz, cw, ch) });
        }
      });
      if (pts.length >= 2) {
        ctx.strokeStyle = `rgb(${slot.color[0]},${slot.color[1]},${slot.color[2]})`;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.moveTo(pts[0].p.sx, pts[0].p.sy);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i].p.sx, pts[i].p.sy);
        }
        ctx.stroke();
      }
      pts.forEach((pt, idx) => {
        ctx.fillStyle = idx === pts.length - 1 ? "#ffffff" : `rgb(${slot.color[0]},${slot.color[1]},${slot.color[2]})`;
        ctx.beginPath();
        ctx.arc(pt.p.sx, pt.p.sy, idx === pts.length - 1 ? 4.2 : 2.5, 0, Math.PI * 2);
        ctx.fill();
      });
    });
  }
}

// Mouse interaction: Orbit, Pan, Zoom, Hover
let dragMode = null, lastX = 0, lastY = 0;
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("mousedown", e => {
  lastX = e.clientX;
  lastY = e.clientY;
  dragMode = (e.button === 2 || e.shiftKey) ? "pan" : "orbit";
});
window.addEventListener("mouseup", () => { dragMode = null; });
window.addEventListener("mousemove", e => {
  if (dragMode) {
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragMode === "orbit") {
      state.yaw += dx * 0.008;
      state.pitch = Math.max(-1.55, Math.min(1.55, state.pitch + dy * 0.008));
    } else {
      state.panX += dx;
      state.panY += dy;
    }
    tooltip.style.display = "none";
    render();
    return;
  }
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  if (mx < 0 || my < 0 || mx > rect.width || my > rect.height) {
    tooltip.style.display = "none";
    return;
  }
  let hit = null;
  for (let i = state.renderedBoxes.length - 1; i >= 0; i--) {
    const b = state.renderedBoxes[i];
    if (mx >= b.minX && mx <= b.maxX && my >= b.minY && my <= b.maxY) {
      hit = b.it;
      break;
    }
  }
  if (hit) {
    tooltip.style.display = "block";
    tooltip.style.left = `${mx + 14}px`;
    tooltip.style.top = `${my + 14}px`;
    tooltip.innerHTML =
      `<span style="color:rgb(${hit.slot.color.join(",")});font-weight:700;">S${String(hit.slot.id).padStart(2,"0")} [${hit.slot.symbol}]</span>` +
      ` &nbsp;|&nbsp; F${String(hit.t).padStart(2,"0")}` +
      ` &nbsp;|&nbsp; (x=${hit.x}, y=${hit.y})` +
      ` &nbsp;|&nbsp; attn=${hit.val.toFixed(3)}`;
  } else {
    tooltip.style.display = "none";
  }
});
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  state.zoom = Math.max(0.35, Math.min(3.5, state.zoom * factor));
  render();
}, { passive: false });

resize();
</script>
</body>
</html>
"""


def _encode_frames_b64(
    frames_rgb: Optional[List[np.ndarray]],
    thumb_size: Tuple[int, int] = (128, 128),
) -> List[str]:
    """Encode RGB frames as compact data-URI PNG strings for interactive time scrubbing."""
    if not frames_rgb:
        return []
    encoded: List[str] = []
    for frame in frames_rgb:
        thumb = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        if ok:
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            encoded.append(f"data:image/png;base64,{b64}")
        else:
            encoded.append("")
    return encoded


def export_3d_attention_html(
    attn: np.ndarray,
    num_obj_slots: int,
    active_slot_indices: List[int],
    slot_label_names: Dict[int, str],
    slot_palette: List[Tuple[int, int, int]],
    output_html_path: str,
    overlaid_frames: Optional[List[np.ndarray]] = None,
    bg_slot_attn: Optional[np.ndarray] = None,
    pred_labels: str = "",
    gt_label: Optional[str] = None,
    top_preds: Optional[List[Tuple[str, float]]] = None,
    symbols: Optional[List[str]] = None,
    min_voxel_val: float = 0.04,
    thumb_size: Tuple[int, int] = (128, 128),
) -> str:
    """Export an interactive offline 3D HTML visualizer for slot attention maps.

    Args:
        attn: Attention array of shape `(T, total_slots, H, W)`.
        num_obj_slots: Number of foreground object slots (excluding `bg_slot`).
        active_slot_indices: Slot indices highlighted as active predictions.
        slot_label_names: Mapping `slot_idx -> readable prediction label`.
        slot_palette: List of RGB color tuples for slot visualization.
        output_html_path: Destination `.html` file path.
        overlaid_frames: Optional 16 overlaid RGB frames for time-slice thumbnail preview.
        bg_slot_attn: Optional `(T, H, W)` background slot attention map for soft filtering.
        pred_labels: Predicted label string summary.
        gt_label: Optional ground-truth label string.
        top_preds: Optional list of `(class_name, confidence_pct)`.
        symbols: Optional symbol names for slots when `num_obj_slots == len(symbols)`.
        min_voxel_val: Minimum attention value stored in JSON to keep HTML compact.
    """
    t_len, _, h, w = attn.shape
    active_set = set(active_slot_indices)
    fg_mask = np.clip(1.0 - bg_slot_attn, 0.0, 1.0) if bg_slot_attn is not None else None

    slots_payload = []
    for s_idx in range(num_obj_slots):
        s_attn = attn[:, s_idx, :, :].astype(np.float32)
        if fg_mask is not None:
            # Blend raw slot attn and foreground-weighted attn so active slots stay crisp
            s_attn = 0.5 * s_attn + 0.5 * (s_attn * fg_mask)

        # Sparse voxels above min_voxel_val
        t_coords, y_coords, x_coords = np.where(s_attn >= min_voxel_val)
        voxels = [
            [int(t), int(y), int(x), round(float(s_attn[t, y, x]), 4)]
            for t, y, x in zip(t_coords, y_coords, x_coords)
        ]

        # Squared-attention weighted centroid per frame for 3D trajectory polyline
        centroids = []
        grid_y, grid_x = np.indices((h, w), dtype=np.float32)
        for t in range(t_len):
            frame_a = s_attn[t]
            peak = float(frame_a.max())
            w_sq = frame_a ** 2
            w_sum = float(w_sq.sum())
            if w_sum > 1e-6:
                prob = w_sq / w_sum
                cx = float((prob * grid_x).sum())
                cy = float((prob * grid_y).sum())
            else:
                cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
            centroids.append([t, round(cx, 3), round(cy, 3), round(peak, 4)])

        sym = symbols[s_idx] if (symbols is not None and s_idx < len(symbols)) else f"S{s_idx}"
        color = list(slot_palette[s_idx % len(slot_palette)])
        slots_payload.append(
            {
                "id": s_idx,
                "symbol": sym,
                "label": slot_label_names.get(s_idx, sym),
                "color": color,
                "active": s_idx in active_set,
                "voxels": voxels,
                "centroids": centroids,
            }
        )

    payload = {
        "T": int(t_len),
        "H": int(h),
        "W": int(w),
        "pred_labels": pred_labels,
        "gt_label": gt_label,
        "top_preds": [[name, round(float(pct), 2)] for name, pct in (top_preds or [])],
        "slots": slots_payload,
        "frames_b64": _encode_frames_b64(overlaid_frames, thumb_size=thumb_size),
    }

    safe_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html_content = _HTML_TEMPLATE.replace("__VOLUME_JSON__", safe_json)

    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_html_path
