import * as THREE from 'three';

// 架构浏览器：仅结构 / 公式 / 源码，绝不加载权重。
const BOARD_W = 5.6;
const BOARD_H = 0.5;
const BOARD_D = 1.4;
const GAP = 1.15;
const MODULE_COLOR = 0x4a5a78;

let currentSpec = null;
let currentRows = [];

function typeColor(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) >>> 0;
  return new THREE.Color().setHSL((h % 360) / 360, 0.62, 0.55);
}

function makeLabel(text) {
  const pad = 16;
  const fs = 44;
  const cv = document.createElement('canvas');
  const measure = cv.getContext('2d');
  measure.font = `${fs}px sans-serif`;
  cv.width = Math.ceil(measure.measureText(text).width) + pad * 2;
  cv.height = fs + pad * 2;
  const ctx = cv.getContext('2d');
  ctx.font = `${fs}px sans-serif`;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#dbe6ff';
  ctx.fillText(text, pad, cv.height / 2);
  const tex = new THREE.CanvasTexture(cv);
  tex.needsUpdate = true;
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false })
  );
  const scale = 0.011;
  sprite.scale.set(cv.width * scale, cv.height * scale, 1);
  return sprite;
}

export async function loadArchList() {
  const res = await fetch('/api/arch/list');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export async function loadArch(modelId) {
  const res = await fetch(`/api/arch/${encodeURIComponent(modelId)}`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function getArchSpec() {
  return currentSpec;
}

export function getArchRows() {
  return currentRows;
}

function boardLabel(kind, board) {
  if (kind === 'layer') return `层 ${board.idx}`;
  return board.label || '模块';
}

function disposeBoard(obj) {
  if (obj.geometry) obj.geometry.dispose();
  const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
  for (const mat of mats) {
    if (!mat) continue;
    if (mat.map) mat.map.dispose();
    mat.dispose();
  }
}

// 构建竖直层栈：每行一块层板 + 首尾模块板（词嵌入 / extra_specs / 输出头）。
export async function showArch(group, panel, modelId) {
  const spec = await loadArch(modelId);
  currentSpec = spec;
  currentRows = spec.rows || [];
  for (const child of group.children) disposeBoard(child);
  group.clear();
  panel?.close?.();

  const boards = [];
  for (const row of currentRows) {
    boards.push({
      kind: 'layer',
      idx: row.idx,
      label: `层 ${row.idx}`,
      color: typeColor(`${row.attn}|${row.ffn}`),
      attn: row.attn,
      ffn: row.ffn,
    });
  }
  boards.push({ kind: 'module', label: '词嵌入 tok_emb', color: MODULE_COLOR, mod: null });
  for (const e of spec.extra_specs || []) {
    boards.push({ kind: 'module', label: e.label || '模块', color: MODULE_COLOR, mod: e });
  }
  boards.push({ kind: 'module', label: '输出头 lm_head', color: MODULE_COLOR, mod: null });

  const n = boards.length;
  boards.forEach((board, i) => {
    const material = new THREE.MeshStandardMaterial({
      color: board.color,
      roughness: 0.5,
      metalness: 0.15,
      transparent: true,
      opacity: 0.92,
    });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(BOARD_W, BOARD_H, BOARD_D), material);
    mesh.position.y = ((n - 1) / 2 - i) * GAP;
    mesh.userData = {
      pickable: true,
      kind: board.kind,
      idx: board.idx,
      label: board.label,
      mod: board.mod,
      attn: board.attn,
      ffn: board.ffn,
    };
    group.add(mesh);

    const label = makeLabel(boardLabel(board.kind, board));
    label.position.set(
      -BOARD_W / 2 - label.scale.x / 2 - 0.2,
      mesh.position.y,
      0
    );
    group.add(label);
  });

  const height = (n - 1) * GAP + BOARD_H;
  window.__frameArch?.(height);
  return spec;
}
