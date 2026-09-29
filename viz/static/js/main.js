import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { getModel, getCell } from './api.js';
import { Shelf } from './shelf.js';
import { cellFromIntersection } from './matrix.js';
import { Panel } from './panel.js';
import { showCloud, resizeCloud, disposeCloud } from './cloud.js';

const panel = new Panel(document.getElementById('drawer'));
window.__panel = panel;

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

const cloudGroup = new THREE.Group();
scene.add(cloudGroup);

const CLOUD_BTN_STYLE =
  'margin-top:10px;padding:5px 12px;border-radius:8px;cursor:pointer;' +
  'font-size:12px;border:1px solid #22345a;background:#0e1730;color:#9fb2d8;';
let cloudOn = false;
let cloudMatrix = null;
let cloudSeq = 0;

function refreshCloudBtn() {
  const btn = document.getElementById('cloudBtn');
  if (!btn || btn.disabled) return;
  btn.textContent = cloudOn ? '关闭点云' : '点云';
}

async function onCloudClick() {
  const btn = document.getElementById('cloudBtn');
  if (!cloudMatrix) return;
  if (cloudOn) {
    cloudOn = false;
    cloudSeq += 1;
    disposeCloud(cloudGroup);
    refreshCloudBtn();
    return;
  }
  cloudOn = true;
  const mySeq = ++cloudSeq;
  if (btn) {
    btn.disabled = true;
    btn.textContent = '加载中…';
  }
  try {
    const rendered = await showCloud(
      cloudGroup,
      cloudMatrix,
      SOURCE,
      160,
      () => mySeq !== cloudSeq
    );
    if (mySeq !== cloudSeq) return;
    if (!rendered) cloudOn = false;
  } catch (e) {
    if (mySeq !== cloudSeq) return;
    cloudOn = false;
    console.error('点云加载失败', e);
  } finally {
    if (mySeq === cloudSeq && btn) {
      btn.disabled = false;
      refreshCloudBtn();
    }
  }
}

function mountCloudControl(spec) {
  cloudMatrix = spec.name;
  cloudOn = false;
  cloudSeq += 1;
  disposeCloud(cloudGroup);
  const body = panel.body;
  let btn = document.getElementById('cloudBtn');
  if (!btn) {
    btn = document.createElement('button');
    btn.id = 'cloudBtn';
    btn.textContent = '点云';
    btn.style.cssText = CLOUD_BTN_STYLE;
    btn.onclick = onCloudClick;
    body.append(btn);
  }
  btn.disabled = false;
  refreshCloudBtn();
}

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
    mountCloudControl(ud.spec);
  } else if (ud.role === 'slab') {
    shelf.expand(ud.layerId).catch(console.error);
  }
});

function resize() {
  renderer.setSize(innerWidth, innerHeight, false);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  resizeCloud(innerWidth, innerHeight);
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
