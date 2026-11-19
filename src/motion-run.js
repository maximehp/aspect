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
  const integrator = document.querySelector('#integrator');
  integrator.disabled = true;
  let renderer, backend;
  const report = (label, value) => {
    const group = document.createElement('div'),
      name = document.createElement('dt'),
      output = document.createElement('dd');
    name.textContent = label;
    output.textContent = value;
    group.append(name, output);
    results.append(group);
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
    async function renderTick(tick) {
      const start = performance.now();
      const { scene, camera } = motionPose(tick / 60);
      const solution = await renderer.prepareScene(scene);
      renderer.applyScene(scene, solution, start);
      renderer.draw(camera, 1, resolution);
      renderer.synchronize();
      if (renderer.gl.isContextLost()) throw Error('The graphics context was lost during the replay.');
      return performance.now() - start;
    }
    await renderTick(0);
    const startWarm = performance.now(),
      cold = startWarm - started;
    const initialStats = backend && { ...backend.stats };
    const times = [];
    for (let tick = 1; tick <= ticks; tick++) {
      await nextFrame();
      times.push(await renderTick(tick));
      if (tick % 10 === 0) progress.textContent = tick + ' / ' + ticks + ' moving frames';
    }
    const elapsed = performance.now() - startWarm;
    times.sort((a, b) => a - b);
    report('Contact integrator', integrator.selectedOptions[0].textContent);
    report('Fully lit frames per second', ((ticks * 1000) / elapsed).toFixed(1));
    report('Moving poses finished', ticks + ' / ' + ticks);
    report('Total replay time', (elapsed / 1000).toFixed(2) + ' s');
    report('Median frame work', ms((times[ticks / 2 - 1] + times[ticks / 2]) / 2));
    report('95th percentile frame work', ms(times[Math.ceil(ticks * 0.95) - 1]));
    report('Worst frame work', ms(times.at(-1)));
    report('Frames over 16.7 ms of work', times.filter((t) => t > deadline).length + ' / ' + ticks);
    report('First pose', ms(cold));
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
    integrator.disabled = false;
    run.disabled = false;
  }
});
