import { getGrid, getStats } from './api.js';
import { divergingRGB } from './colors.js';
import { showNeuronLinks, disposeCloud } from './cloud.js';

const BASE_BTN =
  'padding:5px 12px;border-radius:8px;cursor:pointer;font-size:12px;' +
  'border:1px solid #22345a;background:#0e1730;color:#9fb2d8;';
const ACTIVE_BTN =
  'padding:5px 12px;border-radius:8px;cursor:pointer;font-size:12px;' +
  'border:1px solid #2f63d8;background:#1c47a8;color:#fff;';

const rgb = ([r, g, b]) =>
  `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;

function fmt(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toFixed(4));
}

export class Panel {
  constructor(rootEl) {
    this.root = rootEl;
    this.body = rootEl.querySelector('#drawerBody') || rootEl;
    this.canvas =
      this.body.querySelector('#matrix2d') || document.createElement('canvas');
    this.canvas.id = 'matrix2d';
    this.spec = null;
    this.source = 'live';
    this.tiles = 192;
    this.seq = 0;
    this.built = false;
    this.cloudGroup = null;
    this.linksOn = false;
    this.linkSeq = 0;
  }

  _ensure() {
    if (this.built) return;
    const body = this.body;
    body.replaceChildren();

    const title = document.createElement('div');
    title.style.cssText =
      'display:flex;align-items:baseline;gap:8px;margin:0 0 4px;';
    this.nameEl = document.createElement('span');
    this.nameEl.style.cssText =
      'font-weight:700;font-size:15px;color:#dfe6f3;word-break:break-all;';
    this.labelEl = document.createElement('span');
    this.labelEl.style.cssText =
      'color:#9fb2d8;font-size:12px;white-space:nowrap;';
    title.append(this.nameEl, this.labelEl);

    this.metaEl = document.createElement('div');
    this.metaEl.style.cssText =
      'color:#6f86b0;font-size:12px;margin:0 0 10px;font-variant-numeric:tabular-nums;';

    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;justify-content:center;margin:0 0 10px;';
    this.canvas.style.cssText =
      'display:block;max-width:100%;background:#0a1020;border:1px solid #1d2a44;' +
      'border-radius:6px;image-rendering:pixelated;';
    wrap.append(this.canvas);

    const legend = document.createElement('div');
    legend.style.cssText = 'margin:0 0 12px;';
    this.legendCv = document.createElement('canvas');
    this.legendCv.width = 256;
    this.legendCv.height = 12;
    this.legendCv.style.cssText =
      'width:100%;height:12px;border-radius:4px;display:block;';
    const legendLabels = document.createElement('div');
    legendLabels.style.cssText =
      'display:flex;justify-content:space-between;color:#6f86b0;font-size:11px;' +
      'margin-top:2px;font-variant-numeric:tabular-nums;';
    this.vminEl = document.createElement('span');
    this.vmaxEl = document.createElement('span');
    legendLabels.append(this.vminEl, this.vmaxEl);
    legend.append(this.legendCv, legendLabels);

    const controls = document.createElement('div');
    controls.style.cssText = 'display:flex;gap:6px;align-items:center;';
    this.defaultBtn = document.createElement('button');
    this.defaultBtn.textContent = '适配';
    this.defaultBtn.onclick = () => this._select(192);
    this.blockBtn = document.createElement('button');
    this.blockBtn.textContent = '块';
    this.blockBtn.onclick = () => this._select(64);
    this.elementBtn = document.createElement('button');
    this.elementBtn.textContent = '元素';
    this.elementBtn.onclick = () => this._select(256);
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '关闭';
    closeBtn.style.cssText = BASE_BTN + 'margin-left:auto;';
    closeBtn.onclick = () => this.close();
    controls.append(this.defaultBtn, this.blockBtn, this.elementBtn, closeBtn);

    this.linkRow = document.createElement('div');
    this.linkRow.style.cssText = 'display:flex;margin-top:8px;';
    this.linkBtn = document.createElement('button');
    this.linkBtn.textContent = '神经元连线';
    this.linkBtn.style.cssText = BASE_BTN;
    this.linkBtn.onclick = () => this._toggleNeuronLinks();
    this.linkRow.append(this.linkBtn);

    body.append(title, this.metaEl, wrap, legend, controls, this.linkRow);
    this._setActive();
    this.built = true;
  }

  _refreshLinkBtn() {
    if (!this.linkBtn) return;
    this.linkBtn.style.cssText = this.linksOn ? ACTIVE_BTN : BASE_BTN;
    this.linkBtn.textContent = this.linksOn ? '关闭神经元连线' : '神经元连线';
  }

  linksOff() {
    if (!this.linksOn) return;
    this.linksOn = false;
    this.linkSeq += 1;
    if (this.cloudGroup) disposeCloud(this.cloudGroup);
    if (this.linkBtn) this.linkBtn.disabled = false;
    this._refreshLinkBtn();
  }

  async _toggleNeuronLinks() {
    if (!this.spec || this.spec.small !== true || !this.cloudGroup) return;
    const group = this.cloudGroup;
    if (this.linksOn) {
      this.linksOff();
      return;
    }
    this.linksOn = true;
    const mySeq = ++this.linkSeq;
    window.__cloudOff?.();
    const btn = this.linkBtn;
    if (btn) {
      btn.disabled = true;
      btn.textContent = '加载中…';
    }
    try {
      const rendered = await showNeuronLinks(
        group,
        this.spec.name,
        this.source,
        800,
        () => mySeq !== this.linkSeq
      );
      if (mySeq !== this.linkSeq) return;
      if (!rendered) this.linksOn = false;
    } catch (e) {
      if (mySeq !== this.linkSeq) return;
      this.linksOn = false;
      console.error('神经元连线加载失败', e);
    } finally {
      if (mySeq === this.linkSeq && btn) {
        btn.disabled = false;
        this._refreshLinkBtn();
      }
    }
  }

  _setActive() {
    this.defaultBtn.style.cssText = this.tiles === 192 ? ACTIVE_BTN : BASE_BTN;
    this.blockBtn.style.cssText = this.tiles === 64 ? ACTIVE_BTN : BASE_BTN;
    this.elementBtn.style.cssText = this.tiles === 256 ? ACTIVE_BTN : BASE_BTN;
  }

  _select(tiles) {
    if (this.tiles === tiles) return;
    this.tiles = tiles;
    this._setActive();
    this._draw(tiles);
  }

  async show(spec, source = 'live') {
    this.spec = spec;
    this.source = source || 'live';
    this.root.classList.remove('hidden');
    this._ensure();
    this.tiles = 192;
    this._setActive();
    this.linksOn = false;
    this.linkSeq += 1;
    if (this.linkBtn) this.linkBtn.disabled = false;
    this._refreshLinkBtn();
    if (this.linkRow) this.linkRow.style.display = spec.small === true ? 'flex' : 'none';
    this.nameEl.textContent = spec.name || '';
    this.labelEl.textContent = spec.label || '';
    const [o, i] = spec.shape || [];
    this.metaEl.textContent =
      `shape = ${o} × ${i} · ${spec.role || ''} · ${fmt(spec.elements)} 元素`;
    await this._draw(this.tiles);
  }

  async _draw(tiles) {
    if (!this.spec) return;
    const seq = ++this.seq;
    const { name } = this.spec;
    try {
      const [stats, grid] = await Promise.all([
        getStats(name, this.source),
        getGrid(name, this.source, tiles),
      ]);
      if (seq !== this.seq) return;
      this._renderHeat(grid.values, stats.absmax || 1);
      this._drawLegend(stats);
    } catch (e) {
      if (seq !== this.seq) return;
      console.error('加载矩阵失败', name, e);
      this._clear('加载失败');
    }
  }

  _clear(message) {
    const cv = this.canvas;
    if (cv.getContext) {
      cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
    }
    if (this.legendCv && this.legendCv.getContext) {
      this.legendCv.getContext('2d').clearRect(
        0, 0, this.legendCv.width, this.legendCv.height
      );
    }
    if (this.vminEl) this.vminEl.textContent = message || '';
    if (this.vmaxEl) this.vmaxEl.textContent = '';
  }

  _renderHeat(values, absmax) {
    const h = values.length;
    const w = h ? values[0].length : 0;
    if (!h || !w) return;
    const target = Math.max(
      120,
      Math.min(this.body.clientWidth || 360, 384)
    );
    const cell = Math.max(1, Math.floor(target / Math.max(h, w)));
    const cw = cell * w;
    const ch = cell * h;
    const cv = this.canvas;
    cv.width = cw;
    cv.height = ch;
    cv.style.width = `${cw}px`;
    cv.style.height = `${ch}px`;
    const ctx = cv.getContext('2d');
    for (let y = 0; y < h; y++) {
      const row = values[y];
      for (let x = 0; x < w; x++) {
        ctx.fillStyle = rgb(divergingRGB(row[x] / absmax));
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
  }

  _drawLegend(stats) {
    const ctx = this.legendCv.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, this.legendCv.width, 0);
    for (let i = 0; i <= 20; i++) {
      g.addColorStop(i / 20, rgb(divergingRGB((i / 20) * 2 - 1)));
    }
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, this.legendCv.width, this.legendCv.height);
    this.vminEl.textContent = `min ${fmt(stats.vmin)}`;
    this.vmaxEl.textContent = `max ${fmt(stats.vmax)}`;
  }

  close() {
    this.seq += 1;
    this.root.classList.add('hidden');
  }
}
