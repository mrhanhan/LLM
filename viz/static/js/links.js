import * as THREE from 'three';
import { Line2 } from 'three/addons/lines/Line2.js';
import { LineGeometry } from 'three/addons/lines/LineGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

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
      const n = raw ? raw.norm : undefined;
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
