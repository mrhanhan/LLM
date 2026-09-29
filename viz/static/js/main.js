import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { getModel, getCell } from './api.js';
import { Shelf } from './shelf.js';
import { cellFromIntersection } from './matrix.js';

export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070a12);
const canvas = document.getElementById('scene');
export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
export const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 500);
camera.position.set(0, 0, 18);
export const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.AmbientLight(0xffffff, 0.8));
const key = new THREE.DirectionalLight(0xbcd2ff, 1.1); key.position.set(6, 10, 8); scene.add(key);

export const modelGroup = new THREE.Group();
scene.add(modelGroup);

let shelf = null;
const SOURCE = 'live';
try {
  const graph = await getModel(SOURCE);
  shelf = new Shelf(modelGroup, graph, SOURCE);
  window.__shelf = shelf;
} catch (e) {
  console.error('加载模型图失败', e);
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const hoverEl = document.getElementById('hover');
let pointerDown = null;
let hoverTimer = null;
let hoverSeq = 0;

function setPointer(e) {
  pointer.x = (e.clientX / innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / innerHeight) * 2 + 1;
}

function visibleMatrices() {
  return shelf.meshes.filter(
    (m) => m.userData && m.userData.kind === 'matrix' && m.visible
  );
}

renderer.domElement.addEventListener('pointermove', (e) => {
  if (!shelf || !hoverEl) return;
  setPointer(e);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(visibleMatrices(), false);
  if (!hits.length) {
    if (hoverTimer) clearTimeout(hoverTimer);
    hoverTimer = null;
    hoverSeq += 1;
    hoverEl.textContent = '';
    return;
  }
  const hit = hits[0];
  const { i, j } = cellFromIntersection(hit);
  const name = hit.object.userData.name;
  if (hoverTimer) clearTimeout(hoverTimer);
  const seq = ++hoverSeq;
  hoverTimer = setTimeout(async () => {
    hoverTimer = null;
    try {
      const cell = await getCell(name, i, j, SOURCE);
      if (seq !== hoverSeq) return;
      hoverEl.textContent = `i=${i} j=${j} value=${Number(cell.value).toFixed(4)}`;
    } catch (err) {
      if (seq === hoverSeq) hoverEl.textContent = `i=${i} j=${j} value=?`;
    }
  }, 120);
});

renderer.domElement.addEventListener('pointerdown', (e) => {
  pointerDown = { x: e.clientX, y: e.clientY };
});

renderer.domElement.addEventListener('pointerup', (e) => {
  if (!shelf || !pointerDown) return;
  const moved = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y);
  pointerDown = null;
  if (moved > 5) return;
  setPointer(e);
  raycaster.setFromCamera(pointer, camera);
  const candidates = shelf.meshes.filter((m) => m.visible);
  const hits = raycaster.intersectObjects(candidates, false);
  if (!hits.length) return;
  const ud = hits[0].object.userData || {};
  if (ud.kind === 'matrix') {
    window.__panel?.show?.(ud.spec, SOURCE);
  } else if (ud.role === 'slab') {
    shelf.expand(ud.layerId).catch(console.error);
  }
});

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

function lerpTrays() {
  if (!shelf) return;
  for (const tray of shelf.trays.values()) {
    const slab = tray.userData.slab;
    if (!slab) continue;
    const target = tray.userData.targetOpacity ?? 0.7;
    slab.material.opacity += (target - slab.material.opacity) * 0.15;
  }
}

function tick() {
  requestAnimationFrame(tick);
  controls.update();
  lerpTrays();
  renderer.render(scene, camera);
}
tick();
