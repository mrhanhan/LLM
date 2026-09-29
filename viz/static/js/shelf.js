import * as THREE from 'three';
import { buildPlane, buildCubes } from './matrix.js';

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

  pickables() {
    return this.meshes.filter((m) => m.userData && m.userData.role === 'slab');
  }
}
