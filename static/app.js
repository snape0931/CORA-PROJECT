"use strict";

/* =====================================================================
   Cora GCN Explorer
   Endpoints: GET /health, GET /info, POST /predict, POST /predict/cora_node
   Optional : GET /cora/neighbors/{node_id}   (powers the citation graph)
   ===================================================================== */

const CORA_NODES = 2708;
const MAX_CUSTOM_NODES = 20;
const MAX_NEIGHBORS_SHOWN = 14;

// If the page is opened with VS Code Live Server (port 5500/5501), talk to FastAPI on 8000.
// If FastAPI serves the page itself, use same-origin requests.
const API_BASE = ["5500", "5501"].includes(location.port) ? "http://127.0.0.1:8000" : "";

const TOPIC_CONFIG = {
  Neural_Networks: { color: "#6366f1", icon: "fa-brain" },
  Probabilistic_Methods: { color: "#06b6d4", icon: "fa-chart-column" },
  Reinforcement_Learning: { color: "#10b981", icon: "fa-robot" },
  Theory: { color: "#f59e0b", icon: "fa-book-open" },
  Genetic_Algorithms: { color: "#ec4899", icon: "fa-dna" },
  Rule_Learning: { color: "#8b5cf6", icon: "fa-scale-balanced" },
  Case_Based: { color: "#f43f5e", icon: "fa-briefcase" },
};
const FALLBACK_COLORS = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#ec4899", "#8b5cf6", "#f43f5e"];

const PRESETS = {
  preset_a: [10, 25, 42, 100, 250, 500, 800, 1200],
  preset_b: [5, 15, 30, 90, 300, 600, 900, 1400],
};

const state = {
  classNames: {
    0: "Case_Based", 1: "Genetic_Algorithms", 2: "Neural_Networks", 3: "Probabilistic_Methods",
    4: "Reinforcement_Learning", 5: "Rule_Learning", 6: "Theory",
  },
  featureDim: 1433,
  cora: { data: null, selected: 0 },
  graphToken: 0,
};

/* ---------- helpers ---------- */
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pretty = (s) => String(s).replace(/_/g, " ");
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const nameOf = (id) => state.classNames[id] ?? `Class_${id}`;
const cfgOf = (id) =>
  TOPIC_CONFIG[nameOf(id)] || { color: FALLBACK_COLORS[Math.abs(id) % FALLBACK_COLORS.length], icon: "fa-graduation-cap" };
const probsOf = (pred) => pred.probabilities || pred.probabilites || [];

async function fetchJSON(path, options) {
  const res = await fetch(API_BASE + path, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) {
    const d = body && body.detail;
    const err = new Error(d ? (typeof d === "string" ? d : JSON.stringify(d)) : `HTTP ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return body;
}
const post = (path, body) =>
  fetchJSON(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function toast(message, kind = "error") {
  const stack = $("#toast-stack");
  if (!stack) return console.error(message);
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = `<i class="fa-solid ${kind === "error" ? "fa-triangle-exclamation" : "fa-circle-check"}"></i><span></span>`;
  $("span", el).textContent = message;
  stack.appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 300);
  }, 6000);
}

function setBusy(btn, busy, html) {
  if (busy) {
    btn.dataset.html = btn.innerHTML;
    btn.innerHTML = html;
    btn.disabled = true;
  } else {
    if (btn.dataset.html) btn.innerHTML = btn.dataset.html;
    btn.disabled = false;
  }
}
const spinner = (text) => `<i class="fa-solid fa-spinner fa-spin"></i> ${text}`;

/* ---------- start-up ---------- */
document.addEventListener("DOMContentLoaded", async () => {
  $("#docs-link").href = API_BASE + "/docs";

  initTabs();
  initCoraExplorer();
  initCustomPredictor();
  initBenchmark();
  drawEmptyGraph($("#citation-canvas"), "Classify a paper to see its citation neighbourhood");

  const healthy = await initHealthCheck();
  const info = await loadModelInfo();
  if (info) refreshPayload();
  if (healthy) runCora($("#node-id-input").value);
});

async function initHealthCheck() {
  const badge = $("#status-badge"), dot = $(".status-dot", badge), text = $("#status-text");
  try {
    const data = await fetchJSON("/health");
    if (data.status === "healthy") {
      const prov = (data.providers || []).join(", ");
      text.textContent = `ONNX model online${prov ? " · " + prov : ""}`;
      badge.classList.add("ok");
      dot.classList.remove("pulsing");
      $("#spec-provider").textContent = prov || "-";
      return true;
    }
    text.textContent = "Model offline";
  } catch {
    text.textContent = "API disconnected";
  }
  badge.classList.add("err");
  dot.classList.remove("pulsing");
  return false;
}

function initTabs() {
  const btns = $$(".tab-btn"), panes = $$(".tab-content");
  btns.forEach((btn) =>
    btn.addEventListener("click", () => {
      btns.forEach((b) => b.classList.toggle("active", b === btn));
      panes.forEach((p) => p.classList.toggle("active", p.id === btn.dataset.tab));
      // canvases have zero width while hidden, so redraw when their tab opens
      if (btn.dataset.tab === "tab-cora" && state.cora.data) renderCoraDetail();
    })
  );
}

/* ---------- canvas helpers ---------- */
function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 450, h = canvas.clientHeight || 260;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawNode(ctx, x, y, r, color, label) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 14;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 2 * Math.PI);
  ctx.fill();
  ctx.restore();
  ctx.fillStyle = "#ffffff";
  ctx.font = `bold ${r > 20 ? 12 : 10}px Inter, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y);
}

function drawEmptyGraph(canvas, msg) {
  const { ctx, w, h } = setupCanvas(canvas);
  ctx.fillStyle = "#6b7280";
  ctx.font = "13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(msg, w / 2, h / 2);
}

function drawFeatureMap(canvas, vec) {
  const dim = state.featureDim, cols = 48, cell = 6, rows = Math.ceil(dim / cols);
  canvas.width = cols * cell;
  canvas.height = rows * cell;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#070a12";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let max = 0;
  for (let i = 0; i < dim; i++) max = Math.max(max, Math.abs(vec[i] || 0));
  max = max || 1;

  for (let i = 0; i < dim; i++) {
    const x = (i % cols) * cell, y = Math.floor(i / cols) * cell;
    ctx.fillStyle = "#121a2b";
    ctx.fillRect(x, y, cell - 1, cell - 1);
    if (vec[i]) {
      ctx.fillStyle = `rgba(129, 140, 248, ${0.4 + 0.6 * Math.min(1, Math.abs(vec[i]) / max)})`;
      ctx.fillRect(x, y, cell - 1, cell - 1);
    }
  }
}

// Circle layout used by the custom-graph panel
function drawGraph(canvas, preds, edges) {
  const { ctx, w, h } = setupCanvas(canvas);
  const n = preds.length;
  const cx = w / 2, cy = h / 2, radius = Math.max(20, Math.min(w, h) / 2 - 40);
  const pos = preds.map((_, i) => {
    const a = (2 * Math.PI * i) / n - Math.PI / 2;
    return n === 1 ? [cx, cy] : [cx + radius * Math.cos(a), cy + radius * Math.sin(a)];
  });

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(99, 102, 241, 0.35)";
  if (edges) {
    for (let k = 0; k < edges[0].length; k++) {
      const s = edges[0][k], d = edges[1][k];
      if (s === d || !pos[s] || !pos[d]) continue;
      ctx.beginPath();
      ctx.moveTo(pos[s][0], pos[s][1]);
      ctx.lineTo(pos[d][0], pos[d][1]);
      ctx.stroke();
    }
  }
  preds.forEach((p, i) => drawNode(ctx, pos[i][0], pos[i][1], 18, cfgOf(p.predicted_class_id).color, `#${p.node_index}`));
}

/* ---------- Local citation graph (Tab 1) ----------
   Needs GET /cora/neighbors/{id} -> { "neighbors": [ ...node ids... ] }.
   Without that endpoint it still draws the selected paper and explains what is missing. */
async function drawLocalGraph(pred) {
  const canvas = $("#citation-canvas");
  const nodeId = pred.node_index;
  const token = ++state.graphToken;
  $("#graph-node-count").textContent = `Node #${nodeId} context`;

  let neighbors = null;
  try {
    const r = await fetchJSON(`/cora/neighbors/${nodeId}`);
    neighbors = [...new Set((r.neighbors || []).filter((n) => n !== nodeId))];
  } catch (e) {
    neighbors = null;
  }
  if (token !== state.graphToken) return;

  const total = neighbors ? neighbors.length : 0;
  const shown = neighbors ? neighbors.slice(0, MAX_NEIGHBORS_SHOWN) : [];

  // Classify the neighbours so each node gets its topic colour
  let nbPreds = shown.map((id) => ({ node_index: id, predicted_class_id: -1 }));
  if (shown.length) {
    try {
      const d = await post("/predict/cora_node", { node_indices: shown });
      nbPreds = d.predictions;
    } catch { /* keep grey nodes */ }
    if (token !== state.graphToken) return;
  }

  const { ctx, w, h } = setupCanvas(canvas);
  const cx = w / 2, cy = h / 2;
  const radius = Math.max(40, Math.min(w, h) / 2 - 34);
  const pos = nbPreds.map((_, i) => {
    const a = (2 * Math.PI * i) / nbPreds.length - Math.PI / 2;
    return [cx + radius * Math.cos(a), cy + radius * Math.sin(a)];
  });

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "rgba(99, 102, 241, 0.4)";
  pos.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
  });

  nbPreds.forEach((p, i) => {
    const color = p.predicted_class_id >= 0 ? cfgOf(p.predicted_class_id).color : "#4b5563";
    drawNode(ctx, pos[i][0], pos[i][1], 15, color, `${p.node_index}`);
  });
  drawNode(ctx, cx, cy, 24, cfgOf(pred.predicted_class_id).color, `#${nodeId}`);

  ctx.fillStyle = "#9ca3af";
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  if (neighbors === null) {
    ctx.fillText("Add GET /cora/neighbors/{id} to main.py to see cited papers", 10, 10);
  } else if (total === 0) {
    ctx.fillText("This paper has no citations in the graph", 10, 10);
  } else {
    ctx.fillText(
      total > shown.length ? `Showing ${shown.length} of ${total} neighbours` : `${total} neighbour${total === 1 ? "" : "s"}`,
      10, 10
    );
  }

  const present = [...new Set([pred.predicted_class_id, ...nbPreds.map((p) => p.predicted_class_id)])].filter((i) => i >= 0);
  $("#citation-legend").innerHTML = present
    .map((id) => `<span><i style="background:${cfgOf(id).color}"></i>${esc(pretty(nameOf(id)))}</span>`)
    .join("");
}

/* =====================================================================
   TAB 1 - Cora benchmark explorer
   ===================================================================== */
function parseIndices(text) {
  const out = [];
  for (const part of text.split(/[,\s;]+/).filter(Boolean)) {
    const m = part.match(/^(\d+)(?:-(\d+))?$/);
    if (!m) throw new Error(`"${part}" is not a valid index or range.`);
    const a = Number(m[1]);
    const b = m[2] !== undefined ? Number(m[2]) : a;
    if (b < a) throw new Error(`The range ${part} runs backwards.`);
    for (let i = a; i <= b; i++) {
      out.push(i);
      if (out.length > 200) throw new Error("Please request at most 200 papers at a time.");
    }
  }
  if (!out.length) throw new Error("Enter at least one paper index.");
  const bad = out.find((i) => i >= CORA_NODES);
  if (bad !== undefined) throw new Error(`Index ${bad} is out of range (valid: 0-${CORA_NODES - 1}).`);
  return [...new Set(out)];
}

function initCoraExplorer() {
  const input = $("#node-id-input");

  $("#btn-predict-cora").addEventListener("click", () => runCora(input.value));
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") runCora(input.value); });

  $$(".chip[data-nodes]").forEach((chip) =>
    chip.addEventListener("click", () => {
      input.value = chip.dataset.nodes;
      runCora(input.value);
    })
  );

  $("#btn-random-node").addEventListener("click", () => {
    const picks = new Set();
    while (picks.size < 6) picks.add(Math.floor(Math.random() * CORA_NODES));
    input.value = [...picks].sort((a, b) => a - b).join(", ");
    runCora(input.value);
  });

  $("#toggle-logits").addEventListener("change", (e) => document.body.classList.toggle("show-logits", e.target.checked));
}

async function runCora(text) {
  let indices;
  try { indices = parseIndices(text); } catch (e) { return toast(e.message); }

  const btn = $("#btn-predict-cora");
  setBusy(btn, true, spinner("Classifying…"));
  try {
    const data = await post("/predict/cora_node", { node_indices: indices });
    state.cora = { data, selected: 0 };
    renderCora();
  } catch (e) {
    toast(`Prediction failed: ${e.message}`);
  } finally {
    setBusy(btn, false);
  }
}

function renderCora() {
  const preds = state.cora.data.predictions;
  $("#cora-empty-state").classList.add("hidden");
  $("#cora-results-container").classList.remove("hidden");
  $("#cora-overview").classList.remove("hidden");
  $("#overview-count").textContent = `${preds.length} paper${preds.length === 1 ? "" : "s"}`;

  const list = $("#overview-list");
  list.innerHTML = preds
    .map((p, i) => {
      const id = p.predicted_class_id;
      const conf = Math.max(...probsOf(p));
      return `
        <button class="ov-item" data-i="${i}" style="--topic:${cfgOf(id).color}">
          <span class="ov-dot"></span>
          <span class="ov-id">#${p.node_index}</span>
          <span class="ov-topic">${esc(pretty(nameOf(id)))}</span>
          <span class="ov-conf">${(conf * 100).toFixed(1)}%</span>
        </button>`;
    })
    .join("");
  $$(".ov-item", list).forEach((btn) =>
    btn.addEventListener("click", () => {
      state.cora.selected = Number(btn.dataset.i);
      renderCoraDetail();
    })
  );

  const dist = $("#topic-dist");
  dist.classList.toggle("hidden", preds.length < 2);
  if (preds.length >= 2) {
    const counts = {};
    preds.forEach((p) => { counts[p.predicted_class_id] = (counts[p.predicted_class_id] || 0) + 1; });
    const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
    const max = entries[0][1];
    dist.innerHTML =
      `<div class="dist-title">Topic mix</div>` +
      entries
        .map(([id, c]) => {
          const cfg = cfgOf(Number(id));
          return `
            <div class="dist-row">
              <span><i class="fa-solid ${cfg.icon}" style="color:${cfg.color}"></i> ${esc(pretty(nameOf(Number(id))))}</span>
              <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${(c / max) * 100}%;background:${cfg.color}"></div></div>
              <b>${c}</b>
            </div>`;
        })
        .join("");
  }

  renderCoraDetail();
}

function renderCoraDetail() {
  const { data, selected } = state.cora;
  const pred = data.predictions[selected];
  const id = pred.predicted_class_id, cfg = cfgOf(id), probs = probsOf(pred);
  const maxProb = Math.max(...probs);

  $$(".ov-item").forEach((b, i) => b.classList.toggle("active", i === selected));

  $("#pred-paper-label").textContent = `Paper #${pred.node_index}`;
  $("#prediction-hero").style.setProperty("--topic", cfg.color);
  $("#pred-topic-title i").className = `fa-solid ${cfg.icon}`;
  $("#pred-topic-name").textContent = pretty(nameOf(id));
  $("#pred-confidence-pct").textContent = `${(maxProb * 100).toFixed(1)}%`;

  const order = probs.map((_, i) => i).sort((a, b) => probs[b] - probs[a]);
  const list = $("#prob-bars-list");
  list.innerHTML = order
    .map((i) => {
      const c = cfgOf(i);
      const logit = pred.logits ? pred.logits[i].toFixed(2) : "";
      return `
        <div class="prob-item ${i === id ? "top" : ""}">
          <div class="prob-info">
            <span class="prob-name"><i class="fa-solid ${c.icon}" style="color:${c.color}"></i> ${esc(pretty(nameOf(i)))}</span>
            <span class="prob-right">
              <span class="logit">${logit}</span>
              <span class="prob-pct">${(probs[i] * 100).toFixed(1)}%</span>
            </span>
          </div>
          <div class="prob-bar-track">
            <div class="prob-bar-fill" data-w="${(probs[i] * 100).toFixed(1)}" style="background:${c.color}"></div>
          </div>
        </div>`;
    })
    .join("");
  requestAnimationFrame(() =>
    requestAnimationFrame(() => $$(".prob-bar-fill", list).forEach((el) => { el.style.width = `${el.dataset.w}%`; }))
  );

  if (pred.logits) {
    const logits = Object.fromEntries(pred.logits.map((v, i) => [nameOf(i), Number(v.toFixed(4))]));
    $("#raw-logits-code").textContent = JSON.stringify(logits, null, 2);
  }

  drawLocalGraph(pred);
}

/* =====================================================================
   TAB 2 - custom paper & graph predictor
   ===================================================================== */
function randomIndices(count, dim) {
  const s = new Set();
  while (s.size < Math.min(count, dim)) s.add(Math.floor(Math.random() * dim));
  return [...s];
}

function makePayload(n, mode, topology) {
  const dim = state.featureDim;
  const base = PRESETS[mode];
  const vectors = Array.from({ length: n }, () => {
    const v = new Array(dim).fill(0);
    const idx = base ? [...base, ...randomIndices(3, dim)] : randomIndices(18, dim);
    idx.forEach((i) => { if (i < dim) v[i] = 1; });
    return v;
  });

  const pairs = new Set();
  const link = (a, b) => { if (a !== b) { pairs.add(`${a},${b}`); pairs.add(`${b},${a}`); } };
  if (n > 1) {
    if (topology === "ring") for (let i = 0; i < n; i++) link(i, (i + 1) % n);
    else if (topology === "chain") for (let i = 0; i < n - 1; i++) link(i, i + 1);
    else if (topology === "random") {
      for (let i = 0; i < n; i++) {
        let c = 0;
        while (c < Math.min(2, n - 1)) {
          const j = Math.floor(Math.random() * n);
          if (j !== i) { link(i, j); c++; }
        }
      }
    }
  }

  const payload = { node_features: vectors };
  if (pairs.size) {
    const src = [], dst = [];
    pairs.forEach((k) => { const [a, b] = k.split(",").map(Number); src.push(a); dst.push(b); });
    payload.edge_indices = [src, dst];
  }
  return payload;
}

function formatPayload(p) {
  let s = `{\n  "node_features": [\n${p.node_features.map((v) => "    " + JSON.stringify(v)).join(",\n")}\n  ]`;
  if (p.edge_indices) {
    s += `,\n  "edge_indices": [\n    ${JSON.stringify(p.edge_indices[0])},\n    ${JSON.stringify(p.edge_indices[1])}\n  ]`;
  }
  return s + "\n}";
}

function currentControls() {
  const n = clamp(parseInt($("#num-custom-nodes").value, 10) || 2, 1, MAX_CUSTOM_NODES);
  $("#num-custom-nodes").value = n;
  return {
    n,
    mode: $("input[name='feature-mode']:checked").value,
    topology: $("#custom-topology").value,
  };
}

function refreshPayload() {
  const { n, mode, topology } = currentControls();
  $("#custom-payload-editor").value = formatPayload(makePayload(n, mode, topology));
  updatePayloadStats();
}

function updatePayloadStats() {
  const box = $("#payload-stats");
  try {
    const p = JSON.parse($("#custom-payload-editor").value);
    const nodes = p.node_features.length;
    const edges = p.edge_indices && p.edge_indices[0] ? p.edge_indices[0].length : 0;
    const active = nodes
      ? Math.round(p.node_features.reduce((a, v) => a + v.filter(Boolean).length, 0) / nodes)
      : 0;
    box.innerHTML = `
      <span class="badge">${nodes} paper${nodes === 1 ? "" : "s"}</span>
      <span class="badge cyan">${edges ? `${edges} directed edges` : "self-loops only"}</span>
      <span class="badge">≈ ${active} active words / paper</span>`;
  } catch {
    box.innerHTML = `<span class="badge bad">Payload is not valid</span>`;
  }
}

function initCustomPredictor() {
  $("#num-custom-nodes").addEventListener("change", refreshPayload);
  $("#custom-topology").addEventListener("change", refreshPayload);
  $$("input[name='feature-mode']").forEach((r) => r.addEventListener("change", refreshPayload));
  $("#btn-regenerate").addEventListener("click", refreshPayload);
  $("#custom-payload-editor").addEventListener("input", updatePayloadStats);
  $("#btn-predict-custom").addEventListener("click", runCustom);
  refreshPayload();
}

async function runCustom() {
  let payload;
  try { payload = JSON.parse($("#custom-payload-editor").value); }
  catch { return toast("The payload is not valid JSON."); }

  if (!Array.isArray(payload.node_features) || !payload.node_features.length)
    return toast("node_features must be a non-empty list of vectors.");
  const n = payload.node_features.length;
  if (payload.edge_indices) {
    const [s, d] = payload.edge_indices;
    const bad = [...(s || []), ...(d || [])].find((i) => i < 0 || i >= n);
    if (bad !== undefined) return toast(`Edge refers to paper ${bad}, but the graph only has papers 0-${n - 1}.`);
  }

  const btn = $("#btn-predict-custom");
  setBusy(btn, true, spinner("Running…"));
  try {
    const data = await post("/predict", payload);
    renderCustomResults(data, payload);
  } catch (e) {
    toast(`Custom prediction failed: ${e.message}`);
  } finally {
    setBusy(btn, false);
  }
}

function customCard(p, vec) {
  const id = p.predicted_class_id, cfg = cfgOf(id), probs = probsOf(p);
  const conf = Math.max(...probs);

  const top3 = probs.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const bars = top3
    .map(([v, i]) => {
      const c = cfgOf(i);
      return `
        <div class="prob-item ${i === id ? "top" : ""}">
          <div class="prob-info">
            <span class="prob-name">${esc(pretty(nameOf(i)))}</span>
            <span class="prob-pct">${(v * 100).toFixed(1)}%</span>
          </div>
          <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${(v * 100).toFixed(1)}%;background:${c.color}"></div></div>
        </div>`;
    })
    .join("");

  const active = [];
  (vec || []).forEach((x, i) => { if (x) active.push(i); });
  const chips =
    active.slice(0, 10).map((i) => `<span class="chip-mini">#${i}</span>`).join("") +
    (active.length > 10 ? `<span class="chip-mini more">+${active.length - 10} more</span>` : "");

  return `
    <div class="card glass-card paper-card" style="--topic:${cfg.color}">
      <div class="card-header">
        <h3><i class="fa-solid ${cfg.icon}" style="color:${cfg.color}"></i> Paper #${p.node_index}</h3>
      </div>
      <div class="card-body">
        <div class="paper-topic" style="color:${cfg.color}">${esc(pretty(nameOf(id)))}</div>
        <div class="paper-conf">Confidence: <strong>${(conf * 100).toFixed(1)}%</strong></div>
        <div class="bars-mini">${bars}</div>
        <div class="feat-block">
          <div class="feat-head"><span>Input words</span><span>${active.length} / ${state.featureDim} active</span></div>
          <canvas class="fmap" data-node="${p.node_index}"></canvas>
          <div class="chips-mini">${chips}</div>
        </div>
      </div>
    </div>`;
}

function renderCustomResults(data, payload) {
  const wrap = $("#custom-results-wrapper");
  wrap.classList.remove("hidden");

  const preds = data.predictions;
  const grid = $("#custom-nodes-results-grid");
  grid.innerHTML = preds.map((p) => customCard(p, payload.node_features[p.node_index])).join("");
  $$("canvas.fmap", grid).forEach((cv) => drawFeatureMap(cv, payload.node_features[Number(cv.dataset.node)]));

  const nEdges = data.num_edges ?? (payload.edge_indices ? payload.edge_indices[0].length : 0);
  $("#custom-graph-meta").textContent = `${data.num_nodes ?? preds.length} nodes · ${nEdges} edges`;
  drawGraph($("#custom-graph-canvas"), preds, payload.edge_indices || null);

  const present = [...new Set(preds.map((p) => p.predicted_class_id))];
  $("#custom-graph-legend").innerHTML = present
    .map((id) => `<span><i style="background:${cfgOf(id).color}"></i>${esc(pretty(nameOf(id)))}</span>`)
    .join("");

  wrap.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* =====================================================================
   TAB 3 - model specs & latency benchmark
   ===================================================================== */
async function loadModelInfo() {
  try {
    const info = await fetchJSON("/info");

    if (info.class_mapping) {
      state.classNames = {};
      Object.entries(info.class_mapping).forEach(([k, v]) => { state.classNames[Number(k)] = v; });
    }
    if (info.feature_dimension) state.featureDim = info.feature_dimension;

    $("#metric-features").textContent = Number(state.featureDim).toLocaleString();
    $("#metric-classes").textContent = info.num_classes ?? Object.keys(state.classNames).length;

    const fmt = (t) => `${t.name}: [${(t.shape || []).join(", ")}] (${t.type})`;
    const modelName = info["Model Name"] || info.model_name || "SimpleGCN";
    $("#spec-file").textContent = `${modelName} · simple_gcn_cora.onnx`;
    if (info.inputs && info.inputs[0]) $("#spec-input-x").textContent = fmt(info.inputs[0]);
    if (info.inputs && info.inputs[1]) $("#spec-input-edge").textContent = fmt(info.inputs[1]);
    if (info.outputs && info.outputs[0]) $("#spec-output").textContent = fmt(info.outputs[0]);

    $("#class-chips").innerHTML = Object.keys(state.classNames)
      .map((id) => {
        const c = cfgOf(Number(id));
        return `<span class="class-chip" style="--topic:${c.color}"><i class="fa-solid ${c.icon}"></i>${esc(pretty(nameOf(Number(id))))}<b>${esc(id)}</b></span>`;
      })
      .join("");
    return info;
  } catch (e) {
    console.error("Failed to load model info:", e);
    return null;
  }
}

function initBenchmark() {
  const btn = $("#btn-run-benchmark");
  btn.addEventListener("click", async () => {
    const target = $("#bench-target").value;
    const req =
      target === "cora"
        ? { url: "/predict/cora_node", body: { node_indices: [0, 10, 42, 100, 500] } }
        : { url: "/predict", body: makePayload(6, "random", "ring") };

    setBusy(btn, true, spinner("Running benchmark…"));
    $("#bench-ms").textContent = "…";
    $("#bench-runs").innerHTML = "";

    const timings = [];
    try {
      for (let i = 0; i < 5; i++) {
        const t0 = performance.now();
        await post(req.url, req.body);
        timings.push(performance.now() - t0);
      }
      const avg = timings.reduce((a, b) => a + b, 0) / timings.length;
      const min = Math.min(...timings), max = Math.max(...timings);

      $("#bench-ms").textContent = `${avg.toFixed(1)} ms`;
      $("#bench-desc").textContent = `Average over ${timings.length} requests to ${req.url}`;
      $("#bench-runs").innerHTML = timings
        .map((t) => `<div class="bench-bar" style="height:${Math.max(14, (t / max) * 100)}%" title="${t.toFixed(1)} ms"></div>`)
        .join("");
      $("#bench-details").innerHTML = `
        <p><i class="fa-solid fa-circle-check" style="color:#10b981"></i>
        Fastest <strong>${min.toFixed(1)} ms</strong> · slowest <strong>${max.toFixed(1)} ms</strong>.
        These are end-to-end timings from your browser, so they include network and JSON handling,
        not only the ONNX forward pass.</p>`;
    } catch (e) {
      $("#bench-ms").textContent = "Error";
      toast(`Benchmark failed: ${e.message}`);
    } finally {
      setBusy(btn, false);
    }
  });
}
