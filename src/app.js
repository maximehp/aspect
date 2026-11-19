import { LiveRenderer } from './room-display.js';
import { buildRoom, START_CAMERA, WALK_BOUNDS } from './room.js';

const element = id => document.getElementById(id);
const canvas = element('room');
const radians = Math.PI / 180;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const freshCamera = () => ({ ...START_CAMERA, position: [...START_CAMERA.position] });
let camera = freshCamera();
let renderer, scene, queuedScene;
const frameFences = [];
let queuedAt = 0, pending = null, ready = null;
let request = 0, builds = 0;
const keys = new Set();
let previousTime = 0;

function schedule() {
  if (renderer && !request) request = requestAnimationFrame(render);
}

function requestView() {
  schedule();
}

function requestScene(time = performance.now()) {
  if (!renderer) return;
  const requestedAt = performance.now();
  const t=time/1000, play=element('animate').checked;
  const next = buildRoom({
    slats:Number(element('slats').value),
    radius:Number(element('radius').value)/1000,
    secondRadius:Number(element('radius2').value)/1000,
    angle: (Number(element('angle').value)+(play?20*Math.sin(t*1.7):0))*radians,
    secondBlind: element('second').checked,
    secondAngle: (Number(element('angle2').value)+(play?24*Math.sin(t*1.1+.8):0))*radians,
    layout: element('layout').value,
    floor: element('floor').value,
  });
  const last = queuedScene || ready?.scene || pending?.scene || scene;
  const sameObjects = (a,b) => a.length === b.length && a.every((object,i) => object === b[i]);
  if (last && sameObjects(next.rectangles,last.rectangles) && sameObjects(next.roundedBoxes,last.roundedBoxes) &&
      sameObjects(next.lightPosition,last.lightPosition) && sameObjects(next.lampRGB,last.lampRGB)) return;
  queuedScene = next;
  queuedAt = requestedAt;
  canvas.setAttribute('aria-busy','true');
  beginScene();
}

function beginScene() {
  if (!queuedScene || pending || ready || !renderer) return;
  const target = renderer;
  const job = { scene: queuedScene, requested: queuedAt, started: performance.now() };
  pending = job;
  queuedScene = null;
  const resolution=element('resolution').value;
  const size=resolution==='2160'?[3840,2160]:resolution==='1080'?[1920,1080]:null;
  target.prepareScene(job.scene,camera,size).then(result => {
    if (target !== renderer) return;
    pending = null;
    ready = { ...job, result };
    cancelAnimationFrame(request);
    render(performance.now());
  }).catch(error => {
    if (target !== renderer) return;
    pending = null;
    canvas.setAttribute('aria-busy', String(Boolean(queuedScene)));
    report(error);
    beginScene();
  });
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
    while (frameFences.length) {
      const fence = frameFences[0];
      const completion = gl.clientWaitSync(fence, 0, 0);
      if (completion === gl.TIMEOUT_EXPIRED) break;
      gl.deleteSync(fence);
      frameFences.shift();
      if (completion === gl.WAIT_FAILED && !gl.isContextLost()) throw new Error('GPU frame fence failed.');
    }
    if (frameFences.length >= 2) { schedule(); return; }
    if (!ready && !scene) return;
    const completed = ready, started = completed?.started ?? performance.now();
    walk(Math.min((time - previousTime) / 1000, 0.05));
    previousTime = time;
    const changed = Boolean(completed);
    if (completed) {
      ready = null;
      scene = completed.scene;
      renderer.applyScene(scene, completed.result, started);
      builds++;
      element('problem').hidden = true;
    }
    const resolution = element('resolution').value;
    const size = resolution === '2160' ? [3840, 2160] : resolution === '1080' ? [1920, 1080] : null;
    const mode = element('lighting').value;
    const intensity = Number(element('lamp').value) / 100;
    renderer.draw(camera, intensity, size, mode);
    const preparationMs = changed ? performance.now() - started : 0;
    if (element('measure').checked) {
      renderer.synchronize();
      const finished = performance.now();
      element('frame-time').textContent = (finished - started).toFixed(2) + ' ms';
      if (changed) element('edit-time').textContent = (finished - completed.requested).toFixed(1) + ' ms';
    } else {
      element('frame-time').textContent = '—';
      element('edit-time').textContent = '—';
      frameFences.push(gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0));
      gl.flush();
    }
    canvas.setAttribute('aria-busy',String(Boolean(queuedScene || pending)));
    element('builds').textContent = builds;
    element('samples').textContent = renderer.iterations;
    element('update-time').textContent = preparationMs.toFixed(2) + ' ms';
    if (changed) {
      const t = renderer.timings;
      element('phases').textContent = [
        ['Surface basis', t.basis], ['Lamp visibility', t.visibility], ['Lamp input', t.lamp],
        ['Longest CPU sweep', t.workerMs], ['Operator assembly', t.assemblyMs],
        ['Equilibrium', t.solve], ['Appearance setup', t.appearance], ['Visible fields and draw list', renderer.viewMs || 0],
      ].map(([label, ms]) => label + ' ' + ms.toFixed(1) + ' ms').join(' · ');
    }
    element('dimensions').textContent = renderer.gl.drawingBufferWidth + ' × ' + renderer.gl.drawingBufferHeight;
    beginScene();
    if(element('animate').checked && !pending && !ready) requestScene(performance.now());
    if (keys.size) schedule();
  } catch (error) {
    report(error);
  }
}

function report(error) {
  element('problem').hidden = false;
  element('problem').textContent = error.message;
  console.error(error);
}

for (const id of ['lamp','angle','angle2','second','slats','radius','radius2','animate','layout','floor','lighting','resolution','measure']) {
  element(id).addEventListener('input', () => {
    if(id==='slats') for(const radius of ['radius','radius2']){
      element(radius).max=element('slats').value==='1'?'45':'12';
      element(radius).value=element('slats').value==='1'?'28':'8';
    }
    for(const radius of ['radius','radius2'])element(radius+'-value').textContent=element(radius).value+' mm';
    if (['angle','angle2','second','slats','radius','radius2','animate','layout','floor'].includes(id)) requestScene();
    else requestView();
    element('lamp-value').textContent = element('lamp').value + '%';
    element('angle-value').textContent = element('angle').value + '°';
    element('angle2-value').textContent = element('angle2').value + '°';
    element('second-controls').hidden = !element('second').checked;
  });
}

const moveKeys = ['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'];
window.addEventListener('keydown', event => {
  if (!moveKeys.includes(event.code) || event.ctrlKey || event.metaKey || event.altKey ||
      event.target.matches('input,select,button,textarea,[contenteditable=true]')) return;
  event.preventDefault();
  keys.add(event.code);
  requestView();
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
  requestView();
});
canvas.addEventListener('lostpointercapture', () => { drag = null; });
for (const button of document.querySelectorAll('[data-move]')) {
  button.addEventListener('pointerdown', event => {
    button.setPointerCapture(event.pointerId);
    keys.add(button.dataset.move);
    requestView();
  });
  button.addEventListener('lostpointercapture', () => keys.delete(button.dataset.move));
}
element('reset-camera').addEventListener('click', () => { camera = freshCamera(); requestView(); });
new ResizeObserver(requestView).observe(canvas);
canvas.addEventListener('webglcontextlost', event => {
  event.preventDefault();
  cancelAnimationFrame(request);
  request = 0;
  renderer?.dispose();
  renderer = null;
  pending = ready = null;
  frameFences.length = 0;
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
