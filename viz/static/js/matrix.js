import * as THREE from 'three';
import { getGrid, matrixPngUrl } from './api.js';
import { divergingRGB } from './colors.js';

const loader = new THREE.TextureLoader();

export function buildPlane(spec, source = 'live', tiles = 64, height = 0.9) {
  const aspect = spec.shape[0] / spec.shape[1];
  const geo = new THREE.PlaneGeometry(height / aspect, height);
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

export async function buildCubes(spec, source = 'live', height = 0.9) {
  const { values } = await getGrid(spec.name, source, 256);
  const rows = values.length;
  const cols = values[0] ? values[0].length : 0;
  let absmax = 1e-6;
  for (const row of values) {
    for (const v of row) absmax = Math.max(absmax, Math.abs(v));
  }
  const cell = height / Math.max(rows, cols, 1);
  const geo = new THREE.BoxGeometry(cell * 0.92, cell * 0.92, cell * 0.92);
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
