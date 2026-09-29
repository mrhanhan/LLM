import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { getModel } from './api.js';
import { Shelf } from './shelf.js';

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
let pointerDown = null;
renderer.domElement.addEventListener('pointerdown', (e) => {
  pointerDown = { x: e.clientX, y: e.clientY };
});
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!shelf || !pointerDown) return;
  const moved = Math.hypot(e.clientX - pointerDown.x, e.clientY - pointerDown.y);
  pointerDown = null;
  if (moved > 5) return;
  pointer.x = (e.clientX / innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(shelf.pickables(), false);
  if (hits.length) shelf.expand(hits[0].object.userData.layerId).catch(console.error);
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
