import * as THREE from 'three';

const GAP = 1.25;
const SLAB_W = 4.2;
const SLAB_H = 0.35;
const SLAB_D = 1.2;
const BASE_OPACITY = 0.7;
const DIM_OPACITY = 0.25;
const FOCUS_OPACITY = 1.0;

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
  constructor(group, graph) {
    this.group = group;
    this.graph = graph;
    this.byLayer = new Map();
    this.trays = new Map();
    this.meshes = [];
    this.expanded = null;
    this._build();
  }

  _build() {
    const layers = this.graph.layers || [];
    const n = layers.length;
    layers.forEach((L, idx) => {
      const tray = new THREE.Group();
      tray.position.y = ((n - 1) / 2 - idx) * GAP;
      tray.userData.layerId = L.id;
      tray.userData.targetOpacity = BASE_OPACITY;

      const material = new THREE.MeshStandardMaterial({
        color: 0x2f63d8,
        transparent: true,
        opacity: 0.55,
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
  }

  expand(layerId) {
    this.expanded = this.expanded === layerId ? null : layerId;
    for (const [id, tray] of this.trays) {
      tray.userData.targetOpacity =
        id === this.expanded ? FOCUS_OPACITY : this.expanded ? DIM_OPACITY : BASE_OPACITY;
    }
  }

  pickables() {
    return this.meshes;
  }
}
