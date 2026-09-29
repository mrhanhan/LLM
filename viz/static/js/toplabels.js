import * as THREE from 'three';

const MAX_LABELS = 8;
const SCALE = 0.0058;
const LINE_H = 0.3;

export class TopLabels {
  constructor(group) {
    this.group = group;
    this.sprites = [];
  }

  _makeSprite(text, hot) {
    const fs = 40;
    const pad = 12;
    const cv = document.createElement('canvas');
    const measure = cv.getContext('2d');
    measure.font = `${fs}px sans-serif`;
    cv.width = Math.ceil(measure.measureText(text).width) + pad * 2;
    cv.height = fs + pad * 2;
    const ctx = cv.getContext('2d');
    ctx.font = `${fs}px sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = hot ? '#ffe08a' : '#9fd0ff';
    ctx.fillText(text, pad, cv.height / 2);
    const tex = new THREE.CanvasTexture(cv);
    tex.needsUpdate = true;
    const sprite = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false })
    );
    sprite.scale.set(cv.width * SCALE, cv.height * SCALE, 1);
    return sprite;
  }

  update(topn, anchor, chosenId = null) {
    this.dispose();
    if (!anchor || !topn || !topn.length) return;
    const items = topn.slice(0, MAX_LABELS);
    items.forEach((it, i) => {
      const label = it.token === '\n' ? '⏎' : (it.token || '·');
      const text = `${label}  ${(Number(it.prob) * 100).toFixed(1)}%`;
      const sprite = this._makeSprite(text, it.id === chosenId);
      sprite.position.set(anchor.x, anchor.y - i * LINE_H, anchor.z);
      this.group.add(sprite);
      this.sprites.push(sprite);
    });
  }

  dispose() {
    for (const s of this.sprites) {
      this.group.remove(s);
      if (s.material.map) s.material.map.dispose();
      s.material.dispose();
    }
    this.sprites.length = 0;
  }
}
