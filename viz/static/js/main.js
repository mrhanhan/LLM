import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { getModel, getCell, connectWS } from './api.js';
import { Shelf } from './shelf.js';
import { Links, Flow, layerOfMatrixName } from './links.js';
import { cellFromIntersection } from './matrix.js';
import { Panel } from './panel.js';
import { KvBars } from './kv.js';
import { TopLabels } from './toplabels.js';
import { showCloud, resizeCloud, disposeCloud } from './cloud.js';
import { loadArchList, showArch, getArchSpec } from './arch.js';

const panel = new Panel(document.getElementById('drawer'));
window.__panel = panel;

export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a12);
const canvas = document.getElementById('scene');
export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
export const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
camera.position.set(0, 0, 18);
export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.mouseButtons = {
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.PAN,
  RIGHT: THREE.MOUSE.PAN,
};
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const key = new THREE.DirectionalLight(0xbcd2ff, 1.1); key.position.set(6, 10, 8); scene.add(key);

export const modelGroup = new THREE.Group();
scene.add(modelGroup);

const cloudGroup = new THREE.Group();
scene.add(cloudGroup);
panel.cloudGroup = cloudGroup;

const flowGroup = new THREE.Group();
scene.add(flowGroup);

const kvGroup = new THREE.Group();
scene.add(kvGroup);
const kvBars = new KvBars(kvGroup);

const topGroup = new THREE.Group();
scene.add(topGroup);
const topLabels = new TopLabels(topGroup);
window.__kv = kvBars;
window.__top = topLabels;

const archGroup = new THREE.Group();
archGroup.visible = false;
scene.add(archGroup);

// 架构层栈很高，按总高重新摆相机。
let defaultPose = null;
window.__frameArch = (height) => {
  const fov = (camera.fov * Math.PI) / 180;
  const dist = Math.max(14, ((height / 2) / Math.tan(fov / 2)) * 1.2);
  camera.position.set(0, 0, dist);
  controls.target.set(0, 0, 0);
  controls.update();
  defaultPose = { pos: camera.position.clone(), target: controls.target.clone() };
};

const PAN_SPEED = 6;
const _panOffset = new THREE.Vector3();
const _panV = new THREE.Vector3();

function panCamera(deltaX, deltaY) {
  const el = renderer.domElement;
  if (!camera.isPerspectiveCamera || !el.clientHeight) return;
  _panOffset.set(0, 0, 0);
  _panV.copy(camera.position).sub(controls.target);
  const targetDistance = _panV.length() * Math.tan(((camera.fov / 2) * Math.PI) / 180);
  _panV
    .setFromMatrixColumn(camera.matrix, 0)
    .multiplyScalar((-2 * deltaX * targetDistance) / el.clientHeight);
  _panOffset.add(_panV);
  if (controls.screenSpacePanning) {
    _panV.setFromMatrixColumn(camera.matrix, 1);
  } else {
    _panV.setFromMatrixColumn(camera.matrix, 0).crossVectors(camera.up, _panV);
  }
  _panV.multiplyScalar((2 * deltaY * targetDistance) / el.clientHeight);
  _panOffset.add(_panV);
  camera.position.add(_panOffset);
  controls.target.add(_panOffset);
  controls.update();
}

function resetView() {
  const pose = defaultPose || {
    pos: new THREE.Vector3(0, 0, 18),
    target: new THREE.Vector3(0, 0, 0),
  };
  camera.position.copy(pose.pos);
  controls.target.copy(pose.target);
  controls.update();
}

const kbPan = new Set();
const hudPan = new Set();
const KEY_DIR = {
  arrowleft: 'left', arrowright: 'right', arrowup: 'up', arrowdown: 'down',
  a: 'left', d: 'right', w: 'up', s: 'down',
};

function isTyping() {
  const el = document.activeElement;
  return !!el && /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName);
}

addEventListener('keydown', (e) => {
  if (isTyping()) return;
  const dir = KEY_DIR[e.key.toLowerCase()];
  if (!dir) return;
  kbPan.add(dir);
  e.preventDefault();
});
addEventListener('keyup', (e) => {
  const dir = KEY_DIR[e.key.toLowerCase()];
  if (dir) kbPan.delete(dir);
});
addEventListener('blur', () => kbPan.clear());

function bindHud() {
  const hud = document.getElementById('navhud');
  if (!hud) return;
  for (const btn of hud.querySelectorAll('button[data-pan]')) {
    const dir = btn.dataset.pan;
    if (dir === 'reset') {
      btn.addEventListener('click', resetView);
      continue;
    }
    const start = (e) => {
      hudPan.add(dir);
      e.preventDefault();
    };
    const stop = () => hudPan.delete(dir);
    btn.addEventListener('pointerdown', start);
    btn.addEventListener('pointerup', stop);
    btn.addEventListener('pointerleave', stop);
    btn.addEventListener('pointercancel', stop);
    btn.addEventListener('contextmenu', (e) => e.preventDefault());
  }
}
bindHud();

function applyPanInput() {
  let dx = 0;
  let dy = 0;
  for (const set of [kbPan, hudPan]) {
    if (set.has('left')) dx += 1;
    if (set.has('right')) dx -= 1;
    if (set.has('up')) dy += 1;
    if (set.has('down')) dy -= 1;
  }
  if (dx || dy) panCamera(dx * PAN_SPEED, dy * PAN_SPEED);
}


const CLOUD_BTN_STYLE =
  'margin-top:10px;padding:5px 12px;border-radius:8px;cursor:pointer;' +
  'font-size:12px;border:1px solid #22345a;background:#0e1730;color:#9fb2d8;';
let cloudOn = false;
let cloudMatrix = null;
let cloudSeq = 0;

function refreshCloudBtn() {
  const btn = document.getElementById('cloudBtn');
  if (!btn || btn.disabled) return;
  btn.textContent = cloudOn ? '关闭点云' : '点云';
}

function cloudOff() {
  if (!cloudOn) return;
  cloudOn = false;
  cloudSeq += 1;
  disposeCloud(cloudGroup);
  const btn = document.getElementById('cloudBtn');
  if (btn) btn.disabled = false;
  refreshCloudBtn();
}
window.__cloudOff = cloudOff;

async function onCloudClick() {
  const btn = document.getElementById('cloudBtn');
  if (!cloudMatrix) return;
  if (cloudOn) {
    cloudOff();
    return;
  }
  cloudOn = true;
  const mySeq = ++cloudSeq;
  window.__panel?.linksOff?.();
  if (btn) {
    btn.disabled = true;
    btn.textContent = '加载中…';
  }
  try {
    const rendered = await showCloud(
      cloudGroup,
      cloudMatrix,
      source,
      160,
      () => mySeq !== cloudSeq
    );
    if (mySeq !== cloudSeq) return;
    if (!rendered) cloudOn = false;
  } catch (e) {
    if (mySeq !== cloudSeq) return;
    cloudOn = false;
    console.error('点云加载失败', e);
  } finally {
    if (mySeq === cloudSeq && btn) {
      btn.disabled = false;
      refreshCloudBtn();
    }
  }
}

function mountCloudControl(spec) {
  cloudMatrix = spec.name;
  cloudOn = false;
  cloudSeq += 1;
  disposeCloud(cloudGroup);
  const body = panel.body;
  let btn = document.getElementById('cloudBtn');
  if (!btn) {
    btn = document.createElement('button');
    btn.id = 'cloudBtn';
    btn.textContent = '点云';
    btn.style.cssText = CLOUD_BTN_STYLE;
    btn.onclick = onCloudClick;
    body.append(btn);
  }
  btn.disabled = false;
  refreshCloudBtn();
}

// ---------------------------------------------------------------------------
// 实时训练：左栏开始/停止，WS tick 驱动权重配色与滚动 loss 曲线
// ---------------------------------------------------------------------------
const lossCanvas = document.getElementById('loss');
const trainStatsEl = document.getElementById('train-stats');
const startBtn = document.getElementById('btn-train-start');
const stopBtn = document.getElementById('btn-train-stop');
const trainTargetSel = document.getElementById('train-target');
const leftEl = document.getElementById('left');
const trainPanel = document.getElementById('train-panel') || leftEl;
const archPanel = document.getElementById('arch-panel');
const archSelect = document.getElementById('arch-model');
const archMeta = document.getElementById('arch-meta');
const LOSS_MAX = 160;
const lossHistory = [];
let training = false;

function drawLoss() {
  if (!lossCanvas) return;
  const ctx = lossCanvas.getContext('2d');
  const w = lossCanvas.width;
  const h = lossCanvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a1020';
  ctx.fillRect(0, 0, w, h);
  if (lossHistory.length < 2) return;
  let mn = lossHistory[0];
  let mx = lossHistory[0];
  for (const v of lossHistory) {
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const span = mx - mn;
  ctx.strokeStyle = '#4f8cff';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const denom = Math.max(lossHistory.length - 1, 1);
  for (let k = 0; k < lossHistory.length; k++) {
    const x = (k / denom) * w;
    const t = span < 1e-9 ? 0.5 : (lossHistory[k] - mn) / span;
    const y = h - 4 - t * (h - 10);
    if (k === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function updateTrainStats(step, loss, lr) {
  if (!trainStatsEl) return;
  const parts = [];
  if (step != null) parts.push(`step ${step}`);
  if (loss != null && Number.isFinite(Number(loss))) {
    parts.push(`loss ${Number(loss).toFixed(4)}`);
  }
  if (lr != null && Number.isFinite(Number(lr))) {
    parts.push(`lr ${Number(lr).toExponential(1)}`);
  }
  trainStatsEl.textContent = parts.length ? parts.join(' · ') : '未开始训练';
}

function resetLoss() {
  lossHistory.length = 0;
  drawLoss();
  updateTrainStats(null, null, null);
}

function pushLoss(loss) {
  const v = Number(loss);
  if (!Number.isFinite(v)) return;
  lossHistory.push(v);
  if (lossHistory.length > LOSS_MAX) lossHistory.shift();
  drawLoss();
}

function setTrainPanel(visible) {
  if (trainPanel) trainPanel.style.display = visible ? 'block' : 'none';
  if (archPanel) archPanel.style.display = 'none';
  if (leftEl) leftEl.style.display = visible ? 'block' : 'none';
}

function setTraining(on) {
  training = !!on;
  if (startBtn) {
    startBtn.disabled = training;
    startBtn.textContent = training ? '训练中…' : '开始训练';
  }
  if (stopBtn) stopBtn.disabled = !training;
  if (trainTargetSel) trainTargetSel.disabled = training;
}

function markModeButtons(target) {
  for (const b of modeButtons) {
    b.classList.toggle('active', b.dataset.mode === target);
  }
}

async function ensureSourceForTraining(target) {
  if (target === 'ckpt' && !ckptAvailable) return false;
  if (source === target) return true;
  const ok = await reload(target);
  if (ok) {
    viewMode = target;
    showWeightGroups();
  }
  return ok;
}

async function trainStart() {
  if (training) return;
  const target = (trainTargetSel && trainTargetSel.value) || 'live';
  if (!(await ensureSourceForTraining(target))) return;
  markModeButtons(target);
  const prev = training;
  setTraining(true);
  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (e) {
    console.error('开始训练失败', e);
    setTraining(prev);
  }
}

async function trainStop() {
  try {
    const res = await fetch('/api/train/stop', { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
  } catch (e) {
    console.error('停止训练失败', e);
  }
}

startBtn?.addEventListener('click', trainStart);
stopBtn?.addEventListener('click', trainStop);
drawLoss();

let shelf = null;
let links = null;
let flow = null;
let source = 'live';
let viewMode = 'live';
let archList = null;
let archSeq = 0;
let archSelectReady = false;
let savedCamera = null;
window.__shelf = null;
window.__links = null;
window.__flow = null;
let autoExpanded = false;

function maybeAutoExpand(values) {
  if (autoExpanded || !shelf || shelf.expanded) return;
  let bestName = null;
  let best = -1;
  for (const [name, v] of Object.entries(values || {})) {
    const d = v && typeof v === 'object' ? v.delta : null;
    if (typeof d === 'number' && Number.isFinite(d) && d > best) {
      best = d;
      bestName = name;
    }
  }
  const layer = layerOfMatrixName(bestName);
  if (!layer) return;
  autoExpanded = true;
  shelf.expand(layer).then(() => links?.rebuild(shelf)).catch(console.error);
}

const badgeModel = document.getElementById('badge-model');
const badgeDevice = document.getElementById('badge-device');
let serverSource = null;
let ckptAvailable = true;
const ckptBtn = document.querySelector('.viewswitch button[data-mode="ckpt"]');

function setCkptAvailable(has) {
  ckptAvailable = !!has;
  if (ckptBtn) {
    ckptBtn.disabled = !ckptAvailable;
    ckptBtn.title = ckptAvailable
      ? '检视 --ckpt 加载的真实模型'
      : '服务端未加载 checkpoint：请用 --ckpt <path> 启动后重试';
  }
  if (trainTargetSel) {
    const opt = trainTargetSel.querySelector('option[value="ckpt"]');
    if (opt) {
      opt.disabled = !ckptAvailable;
      opt.textContent = ckptAvailable
        ? '192M 微调（--ckpt 权重）'
        : '192M 微调（需 --ckpt）';
    }
    if (!ckptAvailable && trainTargetSel.value === 'ckpt') trainTargetSel.value = 'live';
  }
}
setCkptAvailable(true);
let badgeSource = document.getElementById('badge-source');
if (!badgeSource && badgeDevice && badgeDevice.parentElement) {
  badgeSource = document.createElement('span');
  badgeSource.className = 'badge';
  badgeSource.id = 'badge-source';
  badgeSource.textContent = '–';
  badgeDevice.parentElement.insertBefore(badgeSource, badgeDevice);
}

function fmtCount(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v <= 0) return '–';
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
  return String(v);
}

function setBadges(graph) {
  const model = (graph && graph.model) || {};
  if (badgeModel) {
    const label = model.name || (graph && graph.source) || '–';
    badgeModel.textContent = model.params ? `${label} · ${fmtCount(model.params)}` : label;
  }
  if (badgeDevice) badgeDevice.textContent = (graph && graph.device) || '–';
}

// 切换真实 checkpoint / 实时小模型：重建 Shelf + Links + Flow。
// 成功返回 true；失败返回 false 且保持现有场景/数据源不变（仅记录并提示）。
async function reload(src) {
  const next = ['live', 'ckpt', 'scratch'].includes(src) ? src : 'live';
  try {
    const graph = await getModel(next);
    cloudOff();
    window.__panel?.close?.();
    if (links) links.dispose();
    if (flow) flow.dispose();
    shelf?.dispose();
    modelGroup.clear();
    flowGroup.clear();
    source = next;
    autoExpanded = false;
    kvBars.dispose();
    topLabels.dispose();
    shelf = new Shelf(modelGroup, graph, next);
    window.__shelf = shelf;
    links = new Links(modelGroup, graph);
    links.resizeLinks(innerWidth, innerHeight);
    links.rebuild(shelf);
    window.__links = links;
    flow = new Flow(flowGroup, graph, shelf);
    window.__flow = flow;
    setBadges(graph);
    const layers = (graph.layers || []).length;
    const height = Math.max(1, layers - 1) * 1.25 + 1;
    const fov = (camera.fov * Math.PI) / 180;
    defaultPose = {
      pos: new THREE.Vector3(0, 0, Math.max(14, (height / 2 / Math.tan(fov / 2)) * 1.2)),
      target: new THREE.Vector3(0, 0, 0),
    };
    resetLoss();
    updateKvHud(null);
    setFooterTopN(null);
    setTrainPanel(true);
    return true;
  } catch (e) {
    console.error(`加载模型图失败（${next}）`, e);
    if (badgeModel) badgeModel.textContent = `加载失败 · ${next}`;
    return false;
  }
}

await reload('live');

// ---------------------------------------------------------------------------
// 推理播放器：▶/⏭ 触发 /api/infer，WS token 消息驱动流带/注意力/文本
// ---------------------------------------------------------------------------
const playerEl = document.getElementById('player');
const promptEl = document.getElementById('prompt');
let promptInput = document.getElementById('prompt-input');
if (playerEl && !promptInput) {
  promptInput = document.createElement('input');
  promptInput.id = 'prompt-input';
  promptInput.type = 'text';
  promptInput.value = '人工智能';
  promptInput.style.width = '130px';
  playerEl.insertBefore(promptInput, document.getElementById('btn-step'));
}

const attnCanvas = document.getElementById('attn');

function clearAttn() {
  if (!attnCanvas) return;
  const ctx = attnCanvas.getContext('2d');
  ctx.clearRect(0, 0, attnCanvas.width, attnCanvas.height);
}

function drawAttn(attn) {
  if (!attnCanvas) return;
  const ctx = attnCanvas.getContext('2d');
  const w = attnCanvas.width;
  const h = attnCanvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a1020';
  ctx.fillRect(0, 0, w, h);
  if (!attn || !attn.length) return;
  const bw = w / attn.length;
  for (let i = 0; i < attn.length; i++) {
    const v = Math.max(0, Math.min(1, Number(attn[i]) || 0));
    const bh = v * (h - 4);
    ctx.fillStyle = `rgb(${Math.round(80 + 150 * v)},${Math.round(140 + 70 * v)},${Math.round(220 - 60 * v)})`;
    ctx.fillRect(i * bw + 0.5, h - bh, Math.max(1, bw - 1), bh);
  }
}

function runInfer(maxNewTokens) {
  const prompt = (promptInput && promptInput.value.trim()) || '人工智能';
  flow?.reset();
  clearAttn();
  panel.openInference(prompt);
  const params = panel.getInferParams ? panel.getInferParams() : {};
  if (maxNewTokens != null) params.max_new_tokens = maxNewTokens;
  return fetch('/api/infer', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(Object.assign({ source, prompt }, params)),
  }).catch((e) => console.error('推理请求失败', e));
}

document.getElementById('btn-play')?.addEventListener('click', () => runInfer());
document.getElementById('btn-step')?.addEventListener('click', () => runInfer(1));

connectWS((m) => {
  if (!m) return;
  if (m.type === 'init') {
    serverSource = m.source === 'ckpt' ? 'ckpt' : 'live';
    setCkptAvailable(m.has_ckpt !== false);
    if (badgeSource) {
      badgeSource.textContent = serverSource === 'ckpt' ? 'checkpoint' : '实时训练';
      badgeSource.title = `服务端数据源：${serverSource}`;
    }
    return;
  }
  if (m.type === 'tick') {
    shelf?.updateValues(m.values);
    links?.update(m.values);
    maybeAutoExpand(m.values);
    pushLoss(m.loss);
    updateTrainStats(m.step, m.loss, m.lr);
    return;
  }
  if (m.type === 'status') {
    setTraining(!!m.training);
    if (m.training && m.target && trainTargetSel) trainTargetSel.value = m.target;
    return;
  }
  if (m.type !== 'token') return;
  if (promptEl) promptEl.textContent = m.text ?? '';
  flow?.setToken(m.values);
  links?.update(m.values);
  drawAttn(m.attn);
  kvBars.update(m.kv, shelf, (shelf && shelf.graph && shelf.graph.model && shelf.graph.model.ctx_len) || 64);
  topLabels.update(m.topn, outputAnchor(), m.id);
  updateKvHud(m.kv);
  setFooterTopN(m.topn, m.id);
  panel.setInference({
    text: m.text, topn: m.topn, kv: m.kv, chosenId: m.id,
    n: panel.infer ? (panel.infer.n || 0) + 1 : 1,
  });
});

function outputAnchor() {
  if (!shelf || !shelf.trays) return null;
  const finalTray = shelf.trays.get('final');
  if (finalTray) {
    return new THREE.Vector3(finalTray.position.x - 3.2, finalTray.position.y + 0.2, 1.0);
  }
  const ids = Array.from(shelf.trays.keys());
  const last = ids.length ? shelf.trays.get(ids[ids.length - 1]) : null;
  return last ? new THREE.Vector3(last.position.x - 3.2, last.position.y, 1.0) : null;
}

function updateKvHud(kv) {
  const hud = document.getElementById('kvhud');
  if (!hud) return;
  const body = hud.querySelector('.kvhud-body');
  const cv = hud.querySelector('canvas');
  if (!kv || !kv.layers || !kv.layers.length) {
    hud.classList.add('hidden');
    if (body) body.textContent = '';
    return;
  }
  hud.classList.remove('hidden');
  const bytes = kv.total_bytes >= (1 << 20)
    ? `${(kv.total_bytes / (1 << 20)).toFixed(2)} MB`
    : `${(kv.total_bytes / (1 << 10)).toFixed(1)} KB`;
  if (body) body.textContent = `KV Cache · ${kv.n_layer} 层 · ${kv.total_len} tok · ${bytes}`;
  if (!cv) return;
  const layers = kv.layers;
  const ctx = cv.getContext('2d');
  const w = cv.width;
  const h = cv.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0a1020';
  ctx.fillRect(0, 0, w, h);
  const maxLen = Math.max(1, ...layers.map((L) => L.len || 0));
  const bw = w / layers.length;
  for (let i = 0; i < layers.length; i++) {
    const bh = ((layers[i].len || 0) / maxLen) * (h - 4);
    ctx.fillStyle = '#4f8cff';
    ctx.fillRect(i * bw + 1, h - bh, Math.max(1, bw - 2), bh);
  }
}

function setFooterTopN(topn, chosenId) {
  const node = document.getElementById('topn');
  if (!node) return;
  if (!topn || !topn.length) {
    node.textContent = '';
    return;
  }
  node.textContent = topn.slice(0, 5).map((t) => {
    const tok = t.token === '\n' ? '⏎' : (t.token || '·');
    const mark = t.id === chosenId ? '▸' : '';
    return `${mark}${tok}:${(Number(t.prob) * 100).toFixed(0)}%`;
  }).join('  ');
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const hoverEl = document.getElementById('hover');
let pointerDown = null;
let hoverTimer = null;
let hoverSeq = 0;

function setPointer(e) {
  pointer.x = (e.clientX / innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / innerHeight) * 2 + 1;
}

function visibleMatrices() {
  return shelf.meshes.filter(
    (m) => m.userData && m.userData.kind === 'matrix' && m.visible
  );
}

renderer.domElement.addEventListener('pointermove', (e) => {
  if (!shelf || !hoverEl) return;
  if (viewMode === 'arch') {
    hoverEl.textContent = '';
    return;
  }
  setPointer(e);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(visibleMatrices(), false);
  if (!hits.length) {
    if (hoverTimer) clearTimeout(hoverTimer);
    hoverTimer = null;
    hoverSeq += 1;
    hoverEl.textContent = '';
    return;
  }
  const hit = hits[0];
  const { i, j } = cellFromIntersection(hit);
  const name = hit.object.userData.name;
  if (hoverTimer) clearTimeout(hoverTimer);
  const seq = ++hoverSeq;
  hoverTimer = setTimeout(async () => {
    hoverTimer = null;
    try {
      const cell = await getCell(name, i, j, source);
      if (seq !== hoverSeq) return;
      hoverEl.textContent = `i=${i} j=${j} value=${Number(cell.value).toFixed(4)}`;
    } catch (err) {
      if (seq === hoverSeq) hoverEl.textContent = `i=${i} j=${j} value=?`;
    }
  }, 120);
});

renderer.domElement.addEventListener('pointerdown', (e) => {
  pointerDown = { x: e.clientX, y: e.clientY };
});

renderer.domElement.addEventListener('pointerup', (e) => {
  if (!pointerDown) return;
  const moved = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y);
  pointerDown = null;
  if (moved > 5) return;
  setPointer(e);
  raycaster.setFromCamera(pointer, camera);
  if (viewMode === 'arch') {
    const hits = raycaster.intersectObjects(archGroup.children, true);
    const hit = hits.find(
      (h) => h.object.userData && h.object.userData.pickable
    );
    if (hit) panel.showArchLayer(getArchSpec(), hit.object.userData);
    return;
  }
  if (!shelf) return;
  const candidates = shelf.meshes.filter((m) => m.visible);
  const hits = raycaster.intersectObjects(candidates, false);
  if (!hits.length) return;
  const ud = hits[0].object.userData || {};
  if (ud.kind === 'matrix') {
    window.__panel?.show?.(ud.spec, source, shelf.graph.global_absmax);
    mountCloudControl(ud.spec);
  } else if (ud.role === 'slab') {
    shelf.expand(ud.layerId).then(() => links?.rebuild(shelf)).catch(console.error);
  }
});

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  resizeCloud(innerWidth, innerHeight);
  links?.resizeLinks(innerWidth, innerHeight);
  flow?.resize(innerWidth, innerHeight);
}
addEventListener('resize', resize); resize();

// 顶栏模式切换：live/ckpt 重建权重书架；arch 只切换结构视图（不取权重）。
function populateArchSelect() {
  if (!archSelect || !archList) return;
  if (archSelectReady) return;
  archSelectReady = true;
  archSelect.replaceChildren();
  for (const m of archList) {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.name || m.id;
    archSelect.appendChild(opt);
  }
  archSelect.addEventListener('change', () => renderArch(archSelect.value));
}

async function renderArch(modelId) {
  if (!modelId) return;
  const seq = ++archSeq;
  try {
    await showArch(archGroup, panel, modelId);
  } catch (e) {
    if (seq === archSeq) console.error('加载架构失败', modelId, e);
    return;
  }
  if (seq !== archSeq) return;
  const spec = getArchSpec();
  if (!spec) return;
  const p = spec.params || {};
  if (archMeta) {
    archMeta.textContent = [
      spec.vendor, spec.family, spec.released,
      p.total, p.layers ? `${p.layers} 层` : '',
    ].filter(Boolean).join(' · ');
  }
  if (badgeModel) badgeModel.textContent = spec.name || modelId;
  if (badgeDevice) badgeDevice.textContent = '结构视图';
}

async function enterArch() {
  if (!archList) {
    try {
      archList = await loadArchList();
    } catch (e) {
      console.error('加载架构列表失败', e);
      return false;
    }
  }
  if (viewMode !== 'arch') {
    savedCamera = { pos: camera.position.clone(), target: controls.target.clone() };
  }
  viewMode = 'arch';
  cloudOff();
  window.__panel?.close?.();
  modelGroup.visible = false;
  flowGroup.visible = false;
  cloudGroup.visible = false;
  kvGroup.visible = false;
  topGroup.visible = false;
  archGroup.visible = true;
  if (trainPanel) trainPanel.style.display = 'none';
  if (archPanel) archPanel.style.display = 'block';
  if (leftEl) leftEl.style.display = 'block';
  populateArchSelect();
  await renderArch(archSelect?.value || (archList[0] && archList[0].id));
  return true;
}

function showWeightGroups() {
  archGroup.visible = false;
  modelGroup.visible = true;
  flowGroup.visible = true;
  cloudGroup.visible = true;
  kvGroup.visible = true;
  topGroup.visible = true;
  if (archPanel) archPanel.style.display = 'none';
}

function exitArch() {
  if (viewMode !== 'arch') return;
  showWeightGroups();
  if (savedCamera) {
    camera.position.copy(savedCamera.pos);
    controls.target.copy(savedCamera.target);
    controls.update();
    savedCamera = null;
  }
  if (leftEl) leftEl.style.display = 'block';
  if (trainPanel) trainPanel.style.display = 'block';
  if (shelf) setBadges(shelf.graph);
  viewMode = source;
}

async function switchMode(mode) {
  if (mode === 'arch') return enterArch();
  if (viewMode === 'arch') exitArch();
  if (mode === source && !archGroup.visible) {
    viewMode = mode;
    showWeightGroups();
    return true;
  }
  const ok = await reload(mode);
  if (ok) {
    viewMode = mode;
    showWeightGroups();
  }
  return ok;
}

const modeButtons = Array.from(document.querySelectorAll('.viewswitch button'));
for (const btn of modeButtons) {
  btn.addEventListener('click', async () => {
    const mode = btn.dataset.mode;
    const prev = viewMode;
    const ok = await switchMode(mode);
    if (!ok) viewMode = prev;
    for (const b of modeButtons) {
      b.classList.toggle('active', b.dataset.mode === (ok ? mode : viewMode));
    }
    if (!ok) console.error(`切换到 ${mode} 失败`);
  });
}

function lerpTrays() {
  if (!shelf) return;
  for (const tray of shelf.trays.values()) {
    const slab = tray.userData.slab;
    if (!slab) continue;
    const target = tray.userData.targetOpacity ?? 0.7;
    slab.material.opacity += (target - slab.material.opacity) * 0.15;
  }
}

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  applyPanInput();
  lerpTrays();
  flow?.tick();
  renderer.render(scene, camera);
}
tick();
