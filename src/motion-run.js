import { LiveRenderer } from './room-display.js';
import { motionPose } from './motion-benchmark.js';

const run = document.querySelector('#run');
const progress = document.querySelector('#progress');
const results = document.querySelector('#results');
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));
const resolution = [3840, 2160],
  ticks = 120,
  deadline = 1000 / 60;
const ms = (n) => n.toFixed(1) + ' ms';

run.addEventListener('click', async () => {
  run.disabled = true;
  results.replaceChildren();
  const profile = document.querySelector('#profile'),
    stages = document.querySelector('#stages');
  profile.hidden = true;
  stages.replaceChildren();
  const integrator = document.querySelector('#integrator');
  const bodies = document.querySelector('#extra-bodies');
  const count = Number(bodies.value);
  bodies.disabled = true;
  integrator.disabled = true;
  let renderer, backend;
  const report = (label, value, target = results) => {
    const group = document.createElement('div'),
      name = document.createElement('dt'),
      output = document.createElement('dd');
    name.textContent = label;
    output.textContent = value;
    group.append(name, output);
    target.append(group);
  };
  try {
    progress.textContent = 'Preparing the first pose…';
    await nextFrame();
    const started = performance.now();
    renderer = new LiveRenderer(document.querySelector('#room'));
    if (integrator.value === 'gpu') {
      const { GpuContactIntegrals } = await import('./gpu-contact.js');
      backend = renderer.view.contactBackend = new GpuContactIntegrals(renderer);
    }
    const totals = new Map();
    function record(label, elapsed) {
      totals.set(label, (totals.get(label) || 0) + elapsed);
    }
    async function renderTick(tick, profile = false) {
      const start = performance.now();
      const { scene, camera } = motionPose(tick / 60, count);
      const authored = performance.now();
      const solution = await renderer.prepareScene(scene);
      const solved = performance.now();
      renderer.applyScene(scene, solution, start);
      const ready = performance.now();
      renderer.draw(camera, 1, resolution);
      const submitted = performance.now();
      renderer.synchronize();
      const finished = performance.now();
      if (profile) {
        const t = solution.timings,
          charts = renderer.view.chartMs;
        record('Scene, basis and view setup', authored - start + t.basis + ready - solved);
        record('Exchange and overlapping geometry', solved - authored - t.basis - t.solve);
        record('Equilibrium and directional light', t.solve);
        record('Visible light fields', charts);
        record('View geometry', renderer.viewMs - charts);
        record('Draw submission and upload', submitted - ready - renderer.viewMs);
        record('Wait for finished 4K pixels', finished - submitted);
        record('Within exchange · longest angular worker', t.workerMs);
        record('Within exchange · sparse operator assembly', t.assemblyMs);
        record('Within exchange · lamp visibility', t.visibility);
        record('Within exchange · lamp input', t.lamp);
        record('Within exchange · contact geometry', t.contactMs || 0);
      }
      if (renderer.gl.isContextLost()) throw Error('The graphics context was lost during the replay.');
      return finished - start;
    }
    await renderTick(0);
    const startWarm = performance.now(),
      cold = startWarm - started;
    const initialStats = backend && { ...backend.stats };
    const times = [];
    for (let tick = 1; tick <= ticks; tick++) {
      await nextFrame();
      times.push(await renderTick(tick, true));
      if (tick % 10 === 0) progress.textContent = tick + ' / ' + ticks + ' moving frames';
    }
    const elapsed = performance.now() - startWarm;
    times.sort((a, b) => a - b);
    report('Additional moving solids', String(count));
    report('Contact integrator', integrator.selectedOptions[0].textContent);
    report('Fully lit frames per second', ((ticks * 1000) / elapsed).toFixed(1));
    report('Moving poses finished', ticks + ' / ' + ticks);
    report('Total replay time', (elapsed / 1000).toFixed(2) + ' s');
    report('Median frame work', ms((times[ticks / 2 - 1] + times[ticks / 2]) / 2));
    report('95th percentile frame work', ms(times[Math.ceil(ticks * 0.95) - 1]));
    report('Worst frame work', ms(times.at(-1)));
    report('Frames over 16.7 ms of work', times.filter((t) => t > deadline).length + ' / ' + ticks);
    report('First pose', ms(cold));
    // Means add to the complete mean. The nested exchange timings overlap.
    profile.hidden = false;
    for (const [name, sum] of totals) report(name, ms(sum / ticks), stages);
    if (backend) {
      const s = backend.stats;
      report(
        'GPU integrals per moving frame',
        Math.round((16 * (s.polygons - initialStats.polygons)) / ticks).toLocaleString(),
      );
      report('Packing per frame', ms((s.packMs - initialStats.packMs) / ticks));
      report('GPU submission through readback', ms((s.dispatchReadMs - initialStats.dispatchReadMs) / ticks));
      report('Result reduction per frame', ms((s.reduceMs - initialStats.reduceMs) / ticks));
      report('Oversized polygons integrated on CPU', String(s.fallback - initialStats.fallback));
    }
    // A real-time game clock also advances while rendering. Draw the newest
    // due tick and count the intermediate poses the player never sees.
    progress.textContent = 'Measuring motion against the real-time game clock…';
    await renderTick(0);
    const clockStart = performance.now();
    let shown = 0,
      previous = 0;
    while (previous < ticks) {
      await nextFrame();
      const due = Math.min(ticks, Math.floor((performance.now() - clockStart) / deadline));
      if (due <= previous) continue;
      await renderTick(due);
      shown++;
      previous = due;
    }
    report('Real-time poses shown', shown + ' / ' + ticks);
    report('Real-time poses skipped', String(ticks - shown));
    report('Final pose lateness', ms(performance.now() - clockStart - ticks * deadline));
    progress.textContent = 'Replay complete';
  } catch (error) {
    progress.textContent = error.message;
  } finally {
    backend?.dispose();
    renderer?.dispose();
    bodies.disabled = false;
    integrator.disabled = false;
    run.disabled = false;
  }
});
