import { getGrid, getStats } from './api.js';
import { divergingRGB } from './colors.js';
import { showNeuronLinks, disposeCloud } from './cloud.js';

const BASE_BTN =
  'padding:5px 12px;border-radius:8px;cursor:pointer;font-size:12px;' +
  'border:1px solid #22345a;background:#0e1730;color:#9fb2d8;';
const ACTIVE_BTN =
  'padding:5px 12px;border-radius:8px;cursor:pointer;font-size:12px;' +
  'border:1px solid #2f63d8;background:#1c47a8;color:#fff;';
const TAB_BASE =
  'flex:1;padding:6px 2px;border-radius:8px;cursor:pointer;font-size:12px;' +
  'border:1px solid #22345a;background:#0e1730;color:#9fb2d8;';
const TAB_ACTIVE =
  'flex:1;padding:6px 2px;border-radius:8px;cursor:pointer;font-size:12px;' +
  'border:1px solid #2f63d8;background:#1c47a8;color:#fff;';

const TABS = [
  ['matrix', '矩阵'],
  ['result', '结果'],
  ['kv', 'KV Cache'],
  ['topn', 'TopN'],
];

const rgb = ([r, g, b]) =>
  `rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`;

function fmt(n) {
  const v = Number(n);
  if (!Number.isFinite(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toFixed(4));
}

function fmtBytes(n) {
  const v = Number(n);
  if (!Number.isFinite(v) || v <= 0) return '0 B';
  if (v >= 1 << 20) return `${(v / (1 << 20)).toFixed(2)} MB`;
  if (v >= 1 << 10) return `${(v / (1 << 10)).toFixed(1)} KB`;
  return `${v | 0} B`;
}

function el(tag, css, text) {
  const node = document.createElement(tag);
  if (css) node.style.cssText = css;
  if (text != null) node.textContent = text;
  return node;
}

function appendModule(root, mod, title) {
  if (!mod) return;
  if (title) root.appendChild(el('h4', 'margin:14px 0 6px;font-size:14px;color:#dfe6f3;', title));
  if (mod.desc) root.appendChild(el('div', 'color:#9fb2d8;font-size:12px;margin:0 0 8px;', mod.desc));
  for (const tex of String(mod.tex || '').split(/\n/).filter((x) => x.trim())) {
    const div = el('div', 'margin:8px 0;overflow-x:auto;');
    if (window.katex && window.katex.render) {
      try {
        window.katex.render(tex, div, { displayMode: true, throwOnError: false });
      } catch (e) {
        div.textContent = tex;
      }
    } else {
      div.textContent = tex;
    }
    root.appendChild(div);
  }
  if (mod.code) {
    const pre = el('pre',
      'margin:8px 0;padding:10px;border-radius:8px;overflow-x:auto;' +
      'background:#0a1020;border:1px solid #1d2a44;font-size:11px;line-height:1.5;');
    const code = document.createElement('code');
    code.textContent = String(mod.code).replace(/^```[a-z]*\n?/, '').replace(/```$/, '');
    pre.appendChild(code);
    root.appendChild(pre);
  }
}

export class Panel {
  constructor(rootEl) {
    this.root = rootEl;
    this.body = rootEl.querySelector('#drawerBody') || rootEl;
    this.source = 'live';
    this.seq = 0;
    this.tiles = 192;
    this.built = false;
    this.live = false;
    this.cloudGroup = null;
    this.linksOn = false;
    this.linkSeq = 0;
    this.tab = 'matrix';
    this.spec = null;
    this.infer = null;
    this.params = {
      temperature: 0.9,
      top_k: 20,
      top_p: 0.95,
      max_new_tokens: 40,
      top_n: 8,
      seed: '',
    };
  }

  _ensure() {
    if (this.built) return;
    this.body.replaceChildren();
    this.panes = {};

    this.tabBar = el('div', 'display:flex;gap:4px;margin:0 0 12px;');
    this._tabBtns = {};
    for (const [id, label] of TABS) {
      const btn = el('button', TAB_BASE, label);
      btn.onclick = () => this._selectTab(id);
      this.tabBar.appendChild(btn);
      this._tabBtns[id] = btn;
    }
    this.body.appendChild(this.tabBar);

    this.panes.matrix = el('div');
    this.panes.result = el('div');
    this.panes.kv = el('div');
    this.panes.topn = el('div');
    this.panes.arch = el('div');
    for (const pane of Object.values(this.panes)) this.body.appendChild(pane);

    this._buildMatrixPane();
    this._buildResultPane();
    this.built = true;
    this._selectTab('matrix');
  }

  _buildMatrixPane() {
    const body = this.panes.matrix;

    const title = el('div', 'display:flex;align-items:baseline;gap:8px;margin:0 0 4px;');
    this.nameEl = el('span', 'font-weight:700;font-size:15px;color:#dfe6f3;word-break:break-all;');
    this.labelEl = el('span', 'color:#9fb2d8;font-size:12px;white-space:nowrap;');
    title.append(this.nameEl, this.labelEl);

    this.metaEl = el('div',
      'color:#6f86b0;font-size:12px;margin:0 0 10px;font-variant-numeric:tabular-nums;');

    this.canvas = document.createElement('canvas');
    this.canvas.id = 'matrix2d';
    const wrap = el('div', 'display:flex;justify-content:center;margin:0 0 10px;');
    this.canvas.style.cssText =
      'display:block;max-width:100%;background:#0a1020;border:1px solid #1d2a44;' +
      'border-radius:6px;image-rendering:pixelated;';
    wrap.append(this.canvas);

    const legend = el('div', 'margin:0 0 12px;');
    this.legendCv = document.createElement('canvas');
    this.legendCv.width = 256;
    this.legendCv.height = 12;
    this.legendCv.style.cssText = 'width:100%;height:12px;border-radius:4px;display:block;';
    const labels = el('div',
      'display:flex;justify-content:space-between;color:#6f86b0;font-size:11px;' +
      'margin-top:2px;font-variant-numeric:tabular-nums;');
    this.vminEl = document.createElement('span');
    this.vmaxEl = document.createElement('span');
    labels.append(this.vminEl, this.vmaxEl);
    this.scaleEl = el('div', 'color:#6f86b0;font-size:11px;margin-top:2px;');
    legend.append(this.legendCv, labels, this.scaleEl);

    const controls = el('div', 'display:flex;gap:6px;align-items:center;');
    this.defaultBtn = el('button', null, '适配');
    this.defaultBtn.onclick = () => this._select(192);
    this.blockBtn = el('button', null, '块');
    this.blockBtn.onclick = () => this._select(64);
    this.elementBtn = el('button', null, '元素');
    this.elementBtn.onclick = () => this._select(256);
    const closeBtn = el('button', BASE_BTN + 'margin-left:auto;', '关闭');
    closeBtn.onclick = () => this.close();
    controls.append(this.defaultBtn, this.blockBtn, this.elementBtn, closeBtn);

    this.linkRow = el('div', 'display:flex;margin-top:8px;');
    this.linkBtn = el('button', BASE_BTN, '神经元连线');
    this.linkBtn.onclick = () => this._toggleNeuronLinks();
    this.linkRow.append(this.linkBtn);

    body.append(title, this.metaEl, wrap, legend, controls, this.linkRow);
    this._setActive();
  }

  _buildResultPane() {
    const body = this.panes.result;
    const grid = el('div',
      'display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0 0 12px;');
    const fields = [
      ['temperature', '温度', 0.05, ''],
      ['top_k', 'Top-K', 1, ''],
      ['top_p', 'Top-P', 0.05, ''],
      ['max_new_tokens', '最大 token', 1, ''],
      ['top_n', 'TopN', 1, ''],
      ['seed', '种子', 1, ''],
    ];
    this.paramInputs = {};
    for (const [key, label, step, placeholder] of fields) {
      const box = el('div');
      box.appendChild(el('div', 'color:#9fb2d8;font-size:11px;margin-bottom:3px;', label));
      const input = document.createElement('input');
      input.type = key === 'seed' ? 'text' : 'number';
      input.step = String(step);
      input.placeholder = placeholder;
      input.value = String(this.params[key] ?? '');
      input.style.cssText =
        'width:100%;padding:4px 6px;border-radius:7px;font-size:12px;' +
        'background:#0b1226;color:#dfe6f3;border:1px solid #243454;';
      input.addEventListener('change', () => {
        this.params[key] = input.value;
      });
      box.appendChild(input);
      this.paramInputs[key] = input;
      grid.appendChild(box);
    }

    this.resultStatus = el('div',
      'color:#6f86b0;font-size:11px;margin:0 0 6px;font-variant-numeric:tabular-nums;');
    this.resultText = el('pre',
      'margin:0 0 12px;padding:10px;border-radius:8px;white-space:pre-wrap;word-break:break-word;' +
      'background:#0a1020;border:1px solid #1d2a44;font-size:13px;line-height:1.7;' +
      'max-height:44vh;overflow-y:auto;font-family:inherit;');
    const hint = el('div', 'color:#6f86b0;font-size:11px;margin:0 0 8px;',
      '底栏 ▶ / ⏭ 触发推理；此处显示生成结果。');
    body.append(hint, grid, this.resultStatus, this.resultText);
  }

  _selectTab(name) {
    if (!this.panes) return;
    this.tab = name;
    for (const id of Object.keys(this.panes)) {
      this.panes[id].style.display = id === name ? 'block' : 'none';
    }
    for (const [id, btn] of Object.entries(this._tabBtns)) {
      btn.style.cssText = id === name ? TAB_ACTIVE : TAB_BASE;
    }
    if (name === 'result') this._renderResult();
    if (name === 'kv') this._renderKV();
    if (name === 'topn') this._renderTopN();
  }

  getInferParams() {
    const read = (key, isInt) => {
      const raw = this.paramInputs && this.paramInputs[key]
        ? this.paramInputs[key].value
        : this.params[key];
      if (raw === '' || raw == null) return null;
      const v = Number(raw);
      if (!Number.isFinite(v)) return null;
      return isInt ? Math.round(v) : v;
    };
    return {
      temperature: read('temperature') ?? 0.9,
      top_k: read('top_k', true) ?? 20,
      top_p: read('top_p') ?? 0.95,
      max_new_tokens: read('max_new_tokens', true) ?? 40,
      top_n: read('top_n', true) ?? 8,
      seed: read('seed'),
    };
  }

  openInference(prompt) {
    this.infer = { text: prompt, prompt, topn: [], kv: null, chosenId: null, n: 0 };
    this.root.classList.remove('hidden');
    this._ensure();
    this.tabBar.style.display = 'flex';
    this._selectTab('result');
  }

  setInference(data) {
    if (!this.infer) this.infer = { text: '', prompt: '', topn: [], kv: null, chosenId: null, n: 0 };
    Object.assign(this.infer, data);
    if (!this.built) return;
    if (this.tab === 'result') this._renderResult();
    else if (this.tab === 'kv') this._renderKV();
    else if (this.tab === 'topn') this._renderTopN();
  }

  _renderResult() {
    if (!this.built || !this.resultText) return;
    const inf = this.infer;
    if (!inf) {
      this.resultStatus.textContent = '';
      this.resultText.textContent = '';
      return;
    }
    const prompt = inf.prompt || '';
    const full = inf.text || '';
    const completion = full.startsWith(prompt) ? full.slice(prompt.length) : full;
    this.resultText.replaceChildren();
    this.resultText.append(el('span', 'color:#6f86b0;', prompt));
    this.resultText.append(el('span', null, completion));
    this.resultStatus.textContent = `已生成 ${inf.n || 0} 个 token`;
  }

  _renderTopN() {
    if (!this.built) return;
    const body = this.panes.topn;
    body.replaceChildren();
    const topn = (this.infer && this.infer.topn) || [];
    if (!topn.length) {
      body.appendChild(el('div', 'color:#6f86b0;font-size:12px;', '运行推理后显示输出层 topN 候选。'));
      return;
    }
    body.appendChild(el('div', 'color:#9fb2d8;font-size:12px;margin:0 0 8px;',
      `输出层 top-${topn.length} 候选（黄色为选中词）`));
    const maxp = Math.max(...topn.map((t) => Number(t.prob) || 0), 1e-6);
    for (const t of topn) {
      const hot = t.id === (this.infer && this.infer.chosenId);
      const row = el('div', 'display:flex;align-items:center;gap:8px;margin:0 0 5px;');
      row.append(el('span',
        'width:2.4em;text-align:center;color:' + (hot ? '#ffe08a' : '#dfe6f3') +
        ';font-size:13px;' + (t.token === '\n' ? 'opacity:.7;' : ''), t.token === '\n' ? '⏎' : (t.token || '·')));
      const track = el('div', 'flex:1;height:12px;border-radius:6px;background:#0a1020;overflow:hidden;');
      const fill = el('div', 'height:100%;border-radius:6px;background:' +
        (hot ? '#e0b23c' : '#2f63d8') + ';');
      fill.style.width = `${Math.max(2, (Number(t.prob) / maxp) * 100)}%`;
      track.appendChild(fill);
      row.append(track);
      row.append(el('span',
        'width:3.4em;text-align:right;color:#9fb2d8;font-size:11px;font-variant-numeric:tabular-nums;',
        `${(Number(t.prob) * 100).toFixed(1)}%`));
      body.appendChild(row);
    }
  }

  _drawGrid(cv, grid) {
    const rows = grid.rows || 0;
    const cols = grid.cols || 0;
    if (!rows || !cols) return;
    const cell = Math.max(2, Math.floor(220 / cols));
    cv.width = cell * cols;
    cv.height = cell * rows;
    cv.style.cssText =
      'display:block;background:#0a1020;border:1px solid #1d2a44;border-radius:4px;' +
      'image-rendering:pixelated;max-width:100%;';
    let absmax = 1e-6;
    for (const row of grid.values) for (const v of row) absmax = Math.max(absmax, Math.abs(v));
    const ctx = cv.getContext('2d');
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        ctx.fillStyle = rgb(divergingRGB((grid.values[y][x] || 0) / absmax));
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
  }

  _renderKV() {
    if (!this.built) return;
    const body = this.panes.kv;
    body.replaceChildren();
    const kv = this.infer && this.infer.kv;
    if (!kv || !kv.layers || !kv.layers.length) {
      body.appendChild(el('div', 'color:#6f86b0;font-size:12px;', '运行推理后显示逐层 KV cache。'));
      return;
    }
    body.appendChild(el('div',
      'color:#9fb2d8;font-size:12px;margin:0 0 10px;font-variant-numeric:tabular-nums;',
      `${kv.n_layer} 层 · 当前长度 ${kv.total_len} tok · 估算显存 ${fmtBytes(kv.total_bytes)}`));
    for (const L of kv.layers) {
      const box = el('div', 'margin:0 0 12px;padding:8px;border-radius:8px;background:#0a1020;border:1px solid #1d2a44;');
      box.appendChild(el('div',
        'color:#9fb2d8;font-size:12px;margin:0 0 6px;font-variant-numeric:tabular-nums;',
        `层 ${L.layer} · 长度 ${L.len} · ${fmtBytes(L.bytes)} · ${L.heads} 头 × ${L.head_dim}`));
      const row = el('div', 'display:flex;gap:10px;flex-wrap:wrap;');
      for (const [tag, grid] of [['K', L.k], ['V', L.v]]) {
        const col = el('div', 'flex:1;min-width:120px;');
        col.appendChild(el('div', 'color:#6f86b0;font-size:11px;margin:0 0 3px;', tag));
        const cv = document.createElement('canvas');
        this._drawGrid(cv, grid);
        col.appendChild(cv);
        row.appendChild(col);
      }
      box.appendChild(row);
      body.appendChild(box);
    }
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

  setLive(on) {
    this.live = !!on;
  }

  refreshLive() {
    if (!this.live || this.tab !== 'matrix' || !this.spec) return;
    this._draw(this.tiles);
  }

  async show(spec, source = 'live', globalAbsmax = null) {
    this.spec = spec;
    this.source = source || 'live';
    this.globalAbsmax = globalAbsmax || null;
    this.root.classList.remove('hidden');
    this._ensure();
    this.tabBar.style.display = 'flex';
    this.tiles = 192;
    this._setActive();
    this.linksOn = false;
    this.linkSeq += 1;
    if (this.linkBtn) this.linkBtn.disabled = false;
    this._refreshLinkBtn();
    this.linkRow.style.display = spec.small === true ? 'flex' : 'none';
    this.nameEl.textContent = spec.name || '';
    this.labelEl.textContent = spec.label || '';
    const [o, i] = spec.shape || [];
    this.metaEl.textContent =
      `shape = ${o} × ${i} · ${spec.role || ''} · ${fmt(spec.elements)} 元素`;
    this._selectTab('matrix');
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
      const absmax = this.globalAbsmax || stats.absmax || 1;
      this._renderHeat(grid.values, absmax);
      this._drawLegend(stats, absmax);
    } catch (e) {
      if (seq !== this.seq) return;
      console.error('加载矩阵失败', name, e);
      this._clear('加载失败');
    }
  }

  _clear(message) {
    const cv = this.canvas;
    if (cv && cv.getContext) cv.getContext('2d').clearRect(0, 0, cv.width, cv.height);
    if (this.legendCv && this.legendCv.getContext) {
      this.legendCv.getContext('2d').clearRect(0, 0, this.legendCv.width, this.legendCv.height);
    }
    if (this.vminEl) this.vminEl.textContent = message || '';
    if (this.vmaxEl) this.vmaxEl.textContent = '';
  }

  _renderHeat(values, absmax) {
    const h = values.length;
    const w = h ? values[0].length : 0;
    if (!h || !w) return;
    const target = Math.max(240, Math.min(this.body.clientWidth || 528, 1024));
    const cell = Math.max(1, Math.floor(target / Math.max(h, w)));
    const cw = cell * w;
    const ch = cell * h;
    const cv = this.canvas;
    cv.width = cw;
    cv.height = ch;
    const scale = Math.min(3, target / Math.max(cw, ch));
    cv.style.width = `${Math.round(cw * scale)}px`;
    cv.style.height = `${Math.round(ch * scale)}px`;
    const ctx = cv.getContext('2d');
    for (let y = 0; y < h; y++) {
      const row = values[y];
      for (let x = 0; x < w; x++) {
        ctx.fillStyle = rgb(divergingRGB(row[x] / absmax));
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }
  }

  _drawLegend(stats, absmax = null) {
    const ctx = this.legendCv.getContext('2d');
    const g = ctx.createLinearGradient(0, 0, this.legendCv.width, 0);
    for (let i = 0; i <= 20; i++) g.addColorStop(i / 20, rgb(divergingRGB((i / 20) * 2 - 1)));
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, this.legendCv.width, this.legendCv.height);
    const scale = absmax || stats.absmax || 1;
    this.vminEl.textContent = `-${fmt(scale)}`;
    this.vmaxEl.textContent = `+${fmt(scale)}`;
    if (this.scaleEl) {
      this.scaleEl.textContent = `全局归一化色标 ±${fmt(scale)}（与 3D 板材一致）`;
    }
  }

  showArchLayer(spec, info = {}) {
    this.root.classList.remove('hidden');
    this.built = false;
    this.seq += 1;
    this.infer = null;
    this.body.replaceChildren();
    this.panes = null;

    const body = this.body;
    const h = el('h2', 'margin:0 0 6px;font-size:16px;color:#dfe6f3;');
    body.appendChild(h);

    if (info.kind === 'module') {
      h.textContent = info.label || '模块';
      if (info.mod) appendModule(body, info.mod, '');
      else body.appendChild(el('div', 'color:#9fb2d8;font-size:12px;', '（仅结构占位，无公式/源码）'));
    } else {
      const rows = (spec && spec.rows) || [];
      const row = rows[info.idx] || {};
      const at = ((spec && spec.attn_specs) || {})[row.attn] || {};
      const ft = ((spec && spec.ffn_specs) || {})[row.ffn] || {};
      h.textContent = `层 ${info.idx}`;
      if (row.note) body.appendChild(el('div', 'color:#9fb2d8;font-size:12px;margin:0 0 6px;', `说明：${row.note}`));
      appendModule(body, at, `Attention · ${at.label || ''}`);
      appendModule(body, ft, `FFN/MoE · ${ft.label || ''}`);
    }
    for (const e of (spec && spec.extra_specs) || []) {
      if (info.kind === 'module' && e === info.mod) continue;
      appendModule(body, e, e.label || '');
    }
    const closeBtn = el('button', BASE_BTN + 'margin-top:14px;', '关闭');
    closeBtn.onclick = () => this.close();
    body.appendChild(closeBtn);
  }

  close() {
    this.seq += 1;
    this.root.classList.add('hidden');
  }
}
