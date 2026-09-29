import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { divergingRGB } from './colors.js';

const SPINE_COLOR = 0x3b78ff;
const SPINE_WIDTH = 4.5;
const INNER_COLOR = 0x2fbf6a;
const INNER_WIDTH = 1.6;

const _white = new THREE.Color(0xffffff);

// 矩阵名 -> 所属层 id（tok_emb -> emb；norm_f/lm_head -> final；blocks.{i}. -> L{i}）
export function layerOfMatrixName(name) {
  if (!name) return null;
  if (name === 'tok_emb.weight') return 'emb';
  if (name === 'norm_f.weight' || name === 'lm_head.weight') return 'final';
  const m = /^blocks\.(\d+)\./.exec(name);
  return m ? `L${m[1]}` : null;
}

// 层 id -> 代表矩阵名（用于取该层激活 act）
export function representativeMatrix(layerId) {
  if (!layerId) return null;
  if (layerId === 'emb') return 'tok_emb.weight';
  if (layerId === 'final') return 'lm_head.weight';
  const m = /^L(\d+)$/.exec(layerId);
  return m ? `blocks.${m[1]}.mlp.w3.weight` : null;
}

function actOf(values, name) {
  if (!values || !name) return null;
  const v = values[name];
  if (v == null) return null;
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  const a = v.act;
  return typeof a === 'number' && Number.isFinite(a) ? a : null;
}

export class Links {
  constructor(group, graph) {
    this.group = group;
    this.graph = graph || { connections: [] };
    this.lines = [];
    this.resolution = new THREE.Vector2(innerWidth || 1, innerHeight || 1);
  }

  _world(obj, out) {
    obj.getWorldPosition(out);
    this.group.worldToLocal(out);
    return out;
  }

  _addLine(a, b, conn, baseColor, baseWidth, kind) {
    const geom = new LineGeometry();
    geom.setPositions([a.x, a.y, a.z, b.x, b.y, b.z]);
    const mat = new LineMaterial({
      color: baseColor,
      linewidth: baseWidth,
      worldUnits: false,
      transparent: true,
      opacity: 0.8,
    });
    mat.resolution.copy(this.resolution);
    const line = new Line2(geom, mat);
    line.computeLineDistances();
    line.userData.connection = conn;
    line.userData.kind = kind;
    line.userData.baseColor = new THREE.Color(baseColor);
    line.userData.baseWidth = baseWidth;
    line.userData._norm = null;
    this.group.add(line);
    this.lines.push(line);
    return line;
  }

  _matrixMeshes(shelf, layerId) {
    const tray = shelf.trays.get(layerId);
    const map = new Map();
    for (const mesh of (tray && tray.userData.matrixMeshes) || []) {
      if (mesh.userData && mesh.userData.name) map.set(mesh.userData.name, mesh);
    }
    return map;
  }

  rebuild(shelf) {
    this.dispose();
    if (!shelf || !shelf.trays) return;
    this.group.updateMatrixWorld(true);
    const connections = this.graph.connections || [];
    const a = new THREE.Vector3();
    const b = new THREE.Vector3();
    for (const conn of connections) {
      if (conn.kind === 'spine') {
        const srcLayer = layerOfMatrixName(conn.src);
        const dstLayer = layerOfMatrixName(conn.dst);
        const srcTray = shelf.trays.get(srcLayer);
        const dstTray = shelf.trays.get(dstLayer);
        if (!srcTray || !dstTray) continue;
        this._world(srcTray, a);
        this._world(dstTray, b);
        this._addLine(a, b, conn, SPINE_COLOR, SPINE_WIDTH, 'spine');
      } else if (conn.kind === 'inner') {
        const layer = layerOfMatrixName(conn.src) || layerOfMatrixName(conn.dst);
        if (!layer || shelf.expanded !== layer) continue;
        const meshes = this._matrixMeshes(shelf, layer);
        const srcMesh = meshes.get(conn.src);
        const dstMesh = meshes.get(conn.dst);
        if (!srcMesh || !dstMesh) continue;
        this._world(srcMesh, a);
        this._world(dstMesh, b);
        this._addLine(a, b, conn, INNER_COLOR, INNER_WIDTH, 'inner');
      }
    }
  }

  update(values) {
    if (!values || !this.lines.length) return;
    const present = [];
    for (const line of this.lines) {
      const wf = line.userData.connection && line.userData.connection.weight_from;
      const raw = wf != null ? values[wf] : undefined;
      // tick 消息带 norm，token 消息带 act；优先 norm，回退 act
      const n = raw ? (raw.norm ?? raw.act) : undefined;
      line.userData._norm = typeof n === 'number' && Number.isFinite(n) ? n : null;
      if (line.userData._norm != null) present.push(line.userData._norm);
    }
    if (!present.length) return;
    let mn = present[0];
    let mx = present[0];
    for (const v of present) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const span = mx - mn;
    for (const line of this.lines) {
      const n = line.userData._norm;
      const t = n == null || span < 1e-9 ? 0.5 : (n - mn) / span;
      line.material.linewidth = line.userData.baseWidth * (0.5 + 1.8 * t);
      line.material.opacity = 0.3 + 0.65 * t;
      line.material.color.copy(line.userData.baseColor).lerp(_white, 0.35 * t);
    }
  }

  resizeLinks(w, h) {
    this.resolution.set(w, h);
    for (const line of this.lines) line.material.resolution.set(w, h);
  }

  dispose() {
    for (const line of this.lines) {
      this.group.remove(line);
      line.geometry.dispose?.();
      line.material.dispose?.();
    }
    this.lines.length = 0;
  }
}

const FLOW_GAP = 1.25;
const FLOW_WIDTH = 3.4;
const FLOW_BASE_OPACITY = 0.12;
const FLOW_BASE_COLOR = 0x3b78ff;
const FLOW_LERP = 0.15;

// 相邻托盘之间的半透明数据流带：颜色/透明度由各层代表矩阵 act 驱动
export class Flow {
  constructor(group, graph, shelf = null) {
    this.group = group;
    this.graph = graph || { layers: [] };
    this.shelf = shelf;
    this.ribbons = [];
    this._build();
  }

  _makeRibbon(x, z, y0, y1) {
    const w = FLOW_WIDTH / 2;
    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      'position',
      new THREE.BufferAttribute(
        new Float32Array([x - w, y0, z, x + w, y0, z, x + w, y1, z, x - w, y1, z]),
        3
      )
    );
    geom.setIndex([0, 1, 2, 0, 2, 3]);
    geom.computeVertexNormals();
    return geom;
  }

  _build() {
    this.dispose();
    const layers = this.graph.layers || [];
    if (layers.length < 2) return;
    const tmp = new THREE.Vector3();
    const n = layers.length;
    const posOf = (id, idx) => {
      const tray = this.shelf && this.shelf.trays ? this.shelf.trays.get(id) : null;
      if (tray) {
        tray.getWorldPosition(tmp);
        this.group.worldToLocal(tmp);
        return tmp.clone();
      }
      return new THREE.Vector3(0, ((n - 1) / 2 - idx) * FLOW_GAP, 0);
    };
    for (let i = 0; i < n - 1; i++) {
      const a = posOf(layers[i].id, i);
      const b = posOf(layers[i + 1].id, i + 1);
      const base = new THREE.Color(FLOW_BASE_COLOR);
      const mat = new THREE.MeshBasicMaterial({
        color: base.clone(),
        transparent: true,
        opacity: FLOW_BASE_OPACITY,
        side: THREE.DoubleSide,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(this._makeRibbon(a.x, a.z, a.y, b.y), mat);
      mesh.userData.rep = representativeMatrix(layers[i].id);
      mesh.userData.baseColor = base;
      mesh.userData.targetColor = base.clone();
      mesh.userData.targetOpacity = FLOW_BASE_OPACITY;
      mesh.userData._act = null;
      this.group.add(mesh);
      this.ribbons.push(mesh);
    }
  }

  setToken(values) {
    if (!this.ribbons.length) return;
    const present = [];
    for (const rb of this.ribbons) {
      const v = actOf(values, rb.userData.rep);
      rb.userData._act = v;
      if (v != null) present.push(v);
    }
    if (!present.length) return;
    let mn = present[0];
    let mx = present[0];
    for (const v of present) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const span = mx - mn;
    for (const rb of this.ribbons) {
      const v = rb.userData._act;
      if (v == null) continue;
      const t = span < 1e-9 ? 0.5 : (v - mn) / span;
      const [r, g, b] = divergingRGB(t * 2 - 1);
      rb.userData.targetColor.setRGB(r, g, b);
      rb.userData.targetOpacity = 0.18 + 0.62 * t;
    }
  }

  // 在主动画循环里调用，向目标颜色/透明度缓动
  tick() {
    for (const rb of this.ribbons) {
      const m = rb.material;
      const target = rb.userData.targetOpacity ?? FLOW_BASE_OPACITY;
      m.opacity += (target - m.opacity) * FLOW_LERP;
      m.color.lerp(rb.userData.targetColor || rb.userData.baseColor, FLOW_LERP);
    }
  }

  reset() {
    for (const rb of this.ribbons) {
      rb.userData._act = null;
      rb.userData.targetOpacity = FLOW_BASE_OPACITY;
      rb.userData.targetColor.copy(rb.userData.baseColor);
    }
  }

  // 流带用世界坐标（MeshBasicMaterial 无分辨率概念），resize 无需处理
  resize() {}

  dispose() {
    for (const rb of this.ribbons) {
      this.group.remove(rb);
      rb.geometry.dispose?.();
      rb.material.dispose?.();
    }
    this.ribbons.length = 0;
  }
}
