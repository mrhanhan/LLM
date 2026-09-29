import * as THREE from 'three';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { getNeurons } from './api.js';

const BUCKET_WIDTHS = [1, 2, 4, 7];
const POS_COLOR = [1.0, 0.45, 0.2];
const NEG_COLOR = [0.25, 0.6, 1.0];
const LINE_OPACITY = 0.5;

const lineMats = [];

const modulePath = (name) => name.replace(/\.weight$/, '');

const _c = new THREE.Color();
const colorFor = (t) => _c.setHSL(0.66 * (1 - t), 0.85, 0.32 + 0.28 * t);

function merge(coords, norms) {
  return coords.map((p, i) => [p[0], p[1], p[2], norms[i] ?? 0]);
}

function nodePos(p, cx) {
  return [cx + p[0] * 0.55, p[1] * 2.6, p[2] * 2.6];
}

function cluster(points, cx) {
  const pos = new Float32Array(points.length * 3);
  const col = new Float32Array(points.length * 3);
  let mn = Infinity;
  let mx = -Infinity;
  for (const p of points) {
    const v = p[3];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  points.forEach((p, i) => {
    pos[3 * i] = cx + p[0] * 0.55;
    pos[3 * i + 1] = p[1] * 2.6;
    pos[3 * i + 2] = p[2] * 2.6;
    const t = mx - mn < 1e-9 ? 0.5 : (p[3] - mn) / (mx - mn);
    const c = colorFor(t);
    col[3 * i] = c.r;
    col[3 * i + 1] = c.g;
    col[3 * i + 2] = c.b;
  });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  return new THREE.Points(
    g,
    new THREE.PointsMaterial({
      size: 0.13,
      vertexColors: true,
      transparent: true,
      opacity: 0.95,
    })
  );
}

function buildEdges(group, edges, inPts, outPts) {
  if (!edges || !edges.length) return;
  let maxAbs = 0;
  for (const e of edges) {
    const a = Math.abs(e.w);
    if (a > maxAbs) maxAbs = a;
  }
  if (!(maxAbs > 1e-12)) maxAbs = 1;
  const buckets = BUCKET_WIDTHS.map(() => ({ pos: [], col: [] }));
  for (const e of edges) {
    const t = Math.abs(e.w) / maxAbs;
    const b = Math.min(BUCKET_WIDTHS.length - 1, Math.floor(t * BUCKET_WIDTHS.length));
    const ip = nodePos(inPts[e.i], -2.8);
    const op = nodePos(outPts[e.o], 2.8);
    const bucket = buckets[b];
    bucket.pos.push(ip[0], ip[1], ip[2], op[0], op[1], op[2]);
    const c = e.w >= 0 ? POS_COLOR : NEG_COLOR;
    bucket.col.push(c[0], c[1], c[2], c[0], c[1], c[2]);
  }
  buckets.forEach((bucket, i) => {
    if (!bucket.pos.length) return;
    const geo = new LineSegmentsGeometry();
    geo.setPositions(bucket.pos);
    geo.setColors(bucket.col);
    const mat = new LineMaterial({
      vertexColors: true,
      linewidth: BUCKET_WIDTHS[i],
      worldUnits: false,
      opacity: LINE_OPACITY,
      transparent: true,
    });
    mat.resolution.set(innerWidth, innerHeight);
    lineMats.push(mat);
    const seg = new LineSegments2(geo, mat);
    seg.computeLineDistances();
    group.add(seg);
  });
}

function clearCloud(group) {
  for (const m of lineMats) m.dispose?.();
  lineMats.length = 0;
  group.traverse((o) => {
    o.geometry?.dispose?.();
    o.material?.dispose?.();
  });
  group.clear();
}

export function resizeCloud(w, h) {
  for (const m of lineMats) m.resolution.set(w, h);
}

export async function showCloud(group, matrix, source = 'live', n = 160) {
  clearCloud(group);
  const data = await getNeurons(modulePath(matrix), source, n);
  if (data.error) throw new Error(data.error);
  const inPts = merge(data.in_coords || [], data.in_norm || []);
  const outPts = merge(data.out_coords || [], data.out_norm || []);
  group.add(cluster(inPts, -2.8), cluster(outPts, 2.8));
  buildEdges(group, data.edges, inPts, outPts);
}
