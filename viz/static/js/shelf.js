import * as THREE from 'three';
import { buildPlane, buildCubes, refreshMatrixTexture } from './matrix.js';
import { divergingRGB } from './colors.js';

const GAP = 1.25;
const SLAB_W = 4.2;
const SLAB_H = 0.35;
const SLAB_D = 1.2;
const BASE_OPACITY = 0.7;
const DIM_OPACITY = 0.25;
const SLAB_FADE = 0.12; // 展开层薄板淡出，给矩阵板让位（仍可拾取）
const MATRIX_H = 0.9;
const MATRIX_PITCH = MATRIX_H + 0.15;
const MATRIX_Z = SLAB_D / 2 + 0.05;

// 权重统计值（norm/delta）在矩阵之间做 min-max 归一化，避免不同矩阵量级差异
function bounds(arr) {
  if (!arr.length) return { min: 0, span: 0 };
  let min = arr[0];
  let max = arr[0];
  for (const v of arr) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return { min, span: max - min };
}

function norm01(v, b) {
  return b.span < 1e-9 ? 0.5 : (v - b.min) / b.span;
}

// 对矩阵 mesh 做可视化调制：norm 决定冷暖色调，delta 决定自发光强度。
// 纹理面片保留权重纹理，仅叠加明暗；实例化立方体直接改基础色。
function applyValueVisual(mesh, t, dt, withOpacity = true) {
  const mat = mesh.material;
  if (!mat || Array.isArray(mat)) return;
  const [r, g, b] = divergingRGB(t * 2 - 1);
  if (mat.color && mat.color.setRGB) {
    if (mat.map) mat.color.setRGB(0.55 + 0.45 * r, 0.55 + 0.45 * g, 0.55 + 0.45 * b);
    else mat.color.setRGB(r, g, b);
  }
  if (mat.emissive && mat.emissive.setRGB) {
    const e = 0.15 + 0.5 * dt;
    mat.emissive.setRGB(r * e, g * e, b * e);
  }
  if (withOpacity && mat.transparent && typeof mat.opacity === 'number') {
    mat.opacity = 0.55 + 0.45 * t;
  }
}

function makeLabel(text) {
  const pad = 16;
  const fs = 48;
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
  const scale = 0.012;
  sprite.scale.set(cv.width * scale, cv.height * scale, 1);
  return sprite;
}

export class Shelf {
  constructor(group, graph, source = 'live') {
    this.group = group;
    this.graph = graph;
    this.source = source;
    this.byLayer = new Map();
    this.trays = new Map();
    this.meshes = [];
    this.expanded = null;
    this._build();
  }

  _build() {
    const layers = this.graph.layers || [];
    layers.forEach((L) => {
      const tray = new THREE.Group();
      tray.userData.layerId = L.id;
      tray.userData.targetOpacity = BASE_OPACITY;

      const material = new THREE.MeshStandardMaterial({
        color: 0x2f63d8,
        transparent: true,
        opacity: BASE_OPACITY,
        roughness: 0.55,
        metalness: 0.1,
      });
      const slab = new THREE.Mesh(new THREE.BoxGeometry(SLAB_W, SLAB_H, SLAB_D), material);
      slab.userData.layerId = L.id;
      slab.userData.role = 'slab';
      tray.add(slab);

      const label = makeLabel(L.label);
      label.position.x = -SLAB_W / 2 - label.scale.x / 2 - 0.25;
      tray.add(label);

      tray.userData.slab = slab;
      tray.userData.label = label;
      this.group.add(tray);
      this.trays.set(L.id, tray);
      this.meshes.push(slab);
      this.byLayer.set(
        L.id,
        (this.graph.matrices || []).filter((m) => m.layer === L.id)
      );
    });
    this.layout();
  }

  layout() {
    const layers = this.graph.layers || [];
    const n = layers.length;
    layers.forEach((L, idx) => {
      const tray = this.trays.get(L.id);
      if (tray) tray.position.y = ((n - 1) / 2 - idx) * GAP;
    });
  }

  async expand(layerId) {
    this.expanded = this.expanded === layerId ? null : layerId;
    if (this.expanded) {
      try {
        await this._buildMatrices(this.expanded);
      } catch (e) {
        console.error('构建矩阵失败', this.expanded, e);
      }
    }
    for (const [id, tray] of this.trays) {
      const isExpanded = id === this.expanded;
      tray.userData.targetOpacity = isExpanded
        ? SLAB_FADE
        : this.expanded
          ? DIM_OPACITY
          : BASE_OPACITY;
      for (const mesh of tray.userData.matrixMeshes || []) mesh.visible = isExpanded;
    }
  }

  async _buildMatrices(layerId) {
    const tray = this.trays.get(layerId);
    if (!tray || tray.userData.matrixMeshes || tray.userData.building) return;
    const specs = this.byLayer.get(layerId) || [];
    const n = specs.length;
    const built = [];
    tray.userData.building = true; // 仅作并发去重，成功后不保留
    try {
      for (let i = 0; i < n; i++) {
        const spec = specs[i];
        const mesh = spec.small
          ? await buildCubes(spec, this.source, MATRIX_H)
          : buildPlane(spec, this.source, 64, MATRIX_H);
        mesh.position.x = (i - (n - 1) / 2) * MATRIX_PITCH;
        mesh.position.z = MATRIX_Z;
        mesh.visible = false;
        tray.add(mesh);
        built.push(mesh);
        this.meshes.push(mesh);
      }
    } catch (e) {
      // 失败回滚：移除已加入的部分网格，且不缓存，以便重试
      for (const mesh of built) {
        tray.remove(mesh);
        const idx = this.meshes.indexOf(mesh);
        if (idx >= 0) this.meshes.splice(idx, 1);
      }
      throw e;
    } finally {
      tray.userData.building = false;
    }
    tray.userData.matrixMeshes = built;
    for (const mesh of built) mesh.visible = this.expanded === layerId;
  }

  // 训练中仅刷新展开层的矩阵纹理；用 _refreshing 防止请求叠加。
  async refreshTextures(step) {
    if (this.expanded == null || this._refreshing) return;
    const tray = this.trays.get(this.expanded);
    const list = tray && tray.userData.matrixMeshes;
    if (!list || !list.length) return;
    this._refreshing = true;
    try {
      await Promise.all(list.map((m) =>
        refreshMatrixTexture(m, this.source, step).catch(() => {})));
    } finally {
      this._refreshing = false;
    }
  }

  pickables() {
    return this.meshes.filter((m) => m.userData && m.userData.role === 'slab');
  }

  // 释放 GPU 资源：矩阵/薄板几何、材质（含 map 纹理）与标签精灵纹理，然后清空分组。
  dispose() {
    const disposeObj = (obj) => {
      if (!obj) return;
      if (obj.geometry) obj.geometry.dispose();
      const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of mats) {
        if (!mat) continue;
        if (mat.map) mat.map.dispose();
        mat.dispose();
      }
    };
    for (const mesh of this.meshes) disposeObj(mesh);
    for (const tray of this.trays.values()) {
      disposeObj(tray.userData.label);
    }
    this.group.clear();
    this.meshes.length = 0;
    this.trays.clear();
    this.byLayer.clear();
    this.expanded = null;
  }

  // 由 WS tick 的 values（按矩阵名）驱动矩阵板配色。
  // 认 norm（预处理归一化值），delta 存在时额外驱动自发光。
  updateValues(values) {
    if (!values) return;
    const items = [];
    const layerAgg = new Map();
    for (const mesh of this.meshes) {
      const ud = mesh.userData || {};
      const name = ud.name;
      if (!name) continue;
      const raw = values[name];
      if (raw == null) continue;
      const n = typeof raw === 'number' ? raw : raw.norm ?? raw.act;
      if (typeof n !== 'number' || !Number.isFinite(n)) continue;
      const delta = typeof raw.delta === 'number' && Number.isFinite(raw.delta)
        ? raw.delta
        : null;
      items.push({ mesh, n, delta });
      const layer = (ud.spec && ud.spec.layer) || ud.layer;
      if (layer != null) {
        let agg = layerAgg.get(layer);
        if (!agg) {
          agg = { n: [], d: [] };
          layerAgg.set(layer, agg);
        }
        agg.n.push(n);
        if (delta != null) agg.d.push(delta);
      }
    }
    if (!items.length) return;
    const nb = bounds(items.map((it) => it.n));
    const db = bounds(items.filter((it) => it.delta != null).map((it) => it.delta));
    for (const it of items) {
      const t = norm01(it.n, nb);
      const dt = it.delta == null ? 0 : norm01(it.delta, db);
      applyValueVisual(it.mesh, t, dt, true);
    }
    const aggLayers = [];
    for (const [layer, agg] of layerAgg) {
      const mn = agg.n.reduce((a, b) => a + b, 0) / agg.n.length;
      const md = agg.d.length ? agg.d.reduce((a, b) => a + b, 0) / agg.d.length : 0;
      aggLayers.push({ layer, mn, md });
    }
    const lnb = bounds(aggLayers.map((v) => v.mn));
    const ldb = bounds(aggLayers.map((v) => v.md));
    for (const v of aggLayers) {
      const tray = this.trays.get(v.layer);
      const slab = tray && tray.userData.slab;
      if (!slab) continue;
      applyValueVisual(slab, norm01(v.mn, lnb), norm01(v.md, ldb), false);
    }
  }
}
