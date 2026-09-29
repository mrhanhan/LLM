import * as THREE from 'three';
import { getGrid, matrixPngUrl } from './api.js';
import { divergingRGB } from './colors.js';

const loader = new THREE.TextureLoader();

const HEIGHT = 0.9;

export function buildPlane(spec, source = 'live', tiles = 64, height = HEIGHT) {
  const geo = new THREE.PlaneGeometry(height, height);
  const tex = loader.load(matrixPngUrl(spec.name, source, tiles, 'global'));
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.LinearFilter;
  const mesh = new THREE.Mesh(
    geo,
    new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })
  );
  mesh.userData = { kind: 'matrix', name: spec.name, spec, shape: spec.shape };
  return mesh;
}

export async function buildCubes(spec, source = 'live', height = HEIGHT) {
  const { values } = await getGrid(spec.name, source, 256);
  const [o, i] = spec.shape;
  const rows = values.length;
  const cols = values[0] ? values[0].length : 0;
  let absmax = 1e-6;
  for (const row of values) {
    for (const v of row) absmax = Math.max(absmax, Math.abs(v));
  }
  const cell = height / Math.max(o, i, 1);
  const boxX = i === 1 ? Math.max(cell * 0.92, 0.05) : cell * 0.92;
  const boxY = o === 1 ? Math.max(cell * 0.92, 0.05) : cell * 0.92;
  const boxZ = cell * 0.92;
  const geo = new THREE.BoxGeometry(boxX, boxY, boxZ);
  const mat = new THREE.MeshBasicMaterial();
  const mesh = new THREE.InstancedMesh(geo, mat, Math.max(rows * cols, 1));
  const m = new THREE.Matrix4();
  const col = new THREE.Color();
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const k = r * cols + c;
      m.makeTranslation((c - (cols - 1) / 2) * cell, ((rows - 1) / 2 - r) * cell, 0);
      mesh.setMatrixAt(k, m);
      const [rr, gg, bb] = divergingRGB(values[r][c] / absmax);
      mesh.setColorAt(k, col.setRGB(rr, gg, bb));
    }
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.userData = { kind: 'matrix', name: spec.name, spec, shape: spec.shape };
  return mesh;
}
