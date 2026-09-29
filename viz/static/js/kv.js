import * as THREE from 'three';
import { divergingRGB } from './colors.js';

const BAR_W = 0.14;
const BAR_MAX_H = 1.05;
const BAR_X = 2.75;
const BAR_Z = 0.95;

function meanAbs(grid) {
  if (!grid || !grid.values || !grid.values.length) return 0;
  let s = 0;
  let n = 0;
  for (const row of grid.values) {
    for (const v of row) {
      s += Math.abs(v);
      n += 1;
    }
  }
  return n ? s / n : 0;
}

export class KvBars {
  constructor(group) {
    this.group = group;
    this.bars = [];
  }

  _ensure(n) {
    while (this.bars.length < n) {
      const geo = new THREE.BoxGeometry(BAR_W, 1, BAR_W);
      geo.translate(0, 0.5, 0);
      const mat = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9 });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.visible = false;
      this.group.add(mesh);
      this.bars.push(mesh);
    }
  }

  update(kv, shelf, ctxLen = 64) {
    if (!kv || !shelf) return;
    const layers = kv.layers || [];
    this._ensure(layers.length);
    let maxLen = 1;
    let maxMean = 1e-6;
    const means = [];
    for (const L of layers) {
      maxLen = Math.max(maxLen, L.len || 0);
      const m = Math.max(meanAbs(L.k), meanAbs(L.v));
      means.push(m);
      maxMean = Math.max(maxMean, m);
    }
    const span = Math.min(maxLen, Math.max(ctxLen, 1));
    for (let i = 0; i < this.bars.length; i++) {
      const bar = this.bars[i];
      const L = layers[i];
      const tray = L ? shelf.trays.get(`L${L.layer}`) : null;
      if (!L || !tray) {
        bar.visible = false;
        continue;
      }
      bar.visible = true;
      bar.position.set(tray.position.x + BAR_X, tray.position.y, BAR_Z);
      const frac = Math.min(1, (L.len || 0) / span);
      bar.scale.y = Math.max(0.03, frac * BAR_MAX_H);
      const [r, g, b] = divergingRGB((means[i] / maxMean) * 2 - 1);
      bar.material.color.setRGB(r, g, b);
    }
  }

  reset() {
    for (const bar of this.bars) bar.visible = false;
  }

  dispose() {
    for (const bar of this.bars) {
      this.group.remove(bar);
      bar.geometry.dispose();
      bar.material.dispose();
    }
    this.bars.length = 0;
  }
}
