import { LiveRenderer } from './live-renderer.js';
import { buildRoom, START_CAMERA, WALK_BOUNDS } from './room.js';

const element = id => document.getElementById(id);
const canvas = element('room');
const radians = Math.PI / 180;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const freshCamera = () => ({ ...START_CAMERA, position: [...START_CAMERA.position] });
let camera = freshCamera();
let renderer, scene, queuedScene, frameFence;
let editStarted = 0;
let request = 0, builds = 0;
const keys = new Set();
let previousTime = 0;

function schedule() {
  if (renderer && !request) request = requestAnimationFrame(render);
}

function requestScene() {
  if (!renderer) return;
  const next = buildRoom({
    angle: Number(element('angle').value)*radians,
    secondBlind: element('second').checked,
    secondAngle: Number(element('angle2').value)*radians,
    layout: element('layout').value,
    floor: element('floor').value,
  });
  const last = queuedScene || scene;
  const sameObjects = (a,b) => a.length === b.length && a.every((object,i) => object === b[i]);
  if (last && sameObjects(next.rectangles,last.rectangles) && sameObjects(next.roundedBoxes,last.roundedBoxes) &&
      sameObjects(next.lightPosition,last.lightPosition) && sameObjects(next.lampRGB,last.lampRGB)) return;
  queuedScene = next;
  editStarted = performance.now();
  canvas.setAttribute('aria-busy','true');
  schedule();
}

function walk(dt) {
  const turn = Number(keys.has('ArrowRight')) - Number(keys.has('ArrowLeft'));
  camera.yaw += turn * dt * 1.6;
  const right = Number(keys.has('KeyD')) - Number(keys.has('KeyA'));
  const forward = Number(keys.has('KeyW') || keys.has('ArrowUp')) -
    Number(keys.has('KeyS') || keys.has('ArrowDown'));
  const step = dt * 2.3 / Math.max(1, Math.hypot(right, forward));
  const c = Math.cos(camera.yaw), s = Math.sin(camera.yaw);
  camera.position[0] = clamp(camera.position[0] + step * (c*right + s*forward), WALK_BOUNDS.minX, WALK_BOUNDS.maxX);
  camera.position[2] = clamp(camera.position[2] + step * (s*right - c*forward), WALK_BOUNDS.minZ, WALK_BOUNDS.maxZ);
}

function render(time) {
  request = 0;
  try {
    const gl = renderer.gl;
    // Keep at most one GPU frame outstanding. New input can replace an edit
    // that has yet to start, including while the last refinement is running.
    if (frameFence) {
      const completion = gl.clientWaitSync(frameFence, 0, 0);
      if (completion === gl.TIMEOUT_EXPIRED) { schedule(); return; }
      gl.deleteSync(frameFence);
      frameFence = null;
      if (completion === gl.WAIT_FAILED && !gl.isContextLost()) throw new Error('GPU frame fence failed.');
    }
    const started = performance.now();
    walk(Math.min((time - previousTime) / 1000, 0.05));
    previousTime = time;
    const changed = Boolean(queuedScene);
    let uploadMs = 0;
    if (changed) {
      scene = queuedScene;
      queuedScene = null;
      const beforeUpload = performance.now();
      renderer.setScene(scene);
      uploadMs = performance.now() - beforeUpload;
      builds++;
    }
    const resolution = element('resolution').value;
    const size = resolution === '2160' ? [3840, 2160] : resolution === '1080' ? [1920, 1080] : null;
    const mode = element('lighting').value;
    const intensity = Number(element('lamp').value) / 100;
    renderer.draw(camera, intensity, size, mode);
    if (element('measure').checked) {
      renderer.synchronize();
      const finished = performance.now();
      element('frame-time').textContent = (finished - started).toFixed(2) + ' ms';
      if (changed) element('edit-time').textContent = (finished - editStarted).toFixed(1) + ' ms';
    } else {
      element('frame-time').textContent = '—';
      element('edit-time').textContent = '—';
      frameFence = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
    }
    canvas.setAttribute('aria-busy','false');
    element('builds').textContent = builds;
    element('samples').textContent = renderer.samples;
    element('update-time').textContent = uploadMs.toFixed(2) + ' ms';
    element('dimensions').textContent = renderer.gl.drawingBufferWidth + ' × ' + renderer.gl.drawingBufferHeight;
    const refining = renderer.needsFrame && intensity > 0 && mode !== 'direct';
    if (keys.size || refining) schedule();
  } catch (error) {
    report(error);
  }
}

function report(error) {
  element('problem').hidden = false;
  element('problem').textContent = error.message;
  console.error(error);
}

for (const id of ['lamp','angle','angle2','second','layout','floor','lighting','resolution','measure']) {
  element(id).addEventListener('input', () => {
    if (['angle','angle2','second','layout','floor'].includes(id)) requestScene();
    element('lamp-value').textContent = element('lamp').value + '%';
    element('angle-value').textContent = element('angle').value + '°';
    element('angle2-value').textContent = element('angle2').value + '°';
    element('second-controls').hidden = !element('second').checked;
    schedule();
  });
}

const moveKeys = ['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];
window.addEventListener('keydown', event => {
  if (!moveKeys.includes(event.code) || event.ctrlKey || event.metaKey || event.altKey ||
      event.target.matches('input,select,button,textarea,[contenteditable=true]')) return;
  event.preventDefault();
  keys.add(event.code);
  schedule();
});
window.addEventListener('keyup', event => keys.delete(event.code));
window.addEventListener('blur', () => keys.clear());

let drag = null;
canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  canvas.focus();
  canvas.setPointerCapture(event.pointerId);
  drag = [event.clientX, event.clientY];
});
canvas.addEventListener('pointermove', event => {
  if (!drag || !canvas.hasPointerCapture(event.pointerId)) return;
  camera.yaw += (event.clientX - drag[0]) * 0.005;
  camera.pitch = clamp(camera.pitch - (event.clientY - drag[1]) * 0.005, -1.45, 1.45);
  drag = [event.clientX, event.clientY];
  schedule();
});
canvas.addEventListener('lostpointercapture', () => { drag = null; });
for (const button of document.querySelectorAll('[data-move]')) {
  button.addEventListener('pointerdown', event => {
    button.setPointerCapture(event.pointerId);
    keys.add(button.dataset.move);
    schedule();
  });
  button.addEventListener('lostpointercapture', () => keys.delete(button.dataset.move));
}
element('reset-camera').addEventListener('click', () => { camera = freshCamera(); schedule(); });
new ResizeObserver(schedule).observe(canvas);
canvas.addEventListener('webglcontextlost', event => {
  event.preventDefault();
  cancelAnimationFrame(request);
  request = 0;
  renderer = null;
  frameFence = null;
  scene = queuedScene = null;
  element('problem').textContent = 'Graphics context lost. Waiting for the browser to restore it.';
  element('problem').hidden = false;
});
function start() {
  try {
    renderer = new LiveRenderer(canvas);
    element('problem').hidden = true;
    scene = queuedScene = null;
    requestScene();
    schedule();
  } catch (error) { report(error); }
}
canvas.addEventListener('webglcontextrestored', start);
start();
