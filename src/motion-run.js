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
  const windowControl = document.querySelector('#window'),
    study = Number(windowControl.value);
  windowControl.disabled = true;
  const count = Number(bodies.value);
  let grouped = 0,
    covered = 0,
    runs = 0,
    lampTests = 0,
    arena = 0;
  bodies.disabled = true;
  integrator.disabled = true;
  let renderer, backend;
  const workCounts=new Float64Array(20);
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
    const production=['function','expanded','pointwise'].includes(integrator.value);
    const Engine=production?LiveRenderer:(await import('./comparison-display.js')).LiveRenderer;
    renderer = new Engine(document.querySelector('#room'), { contacts: integrator.value, regions: integrator.value!=='pointwise', expanded: ['expanded','pointwise'].includes(integrator.value) });
    backend = production?renderer.angular:renderer.view.contactBackend;
    const totals = new Map();
    function record(label, elapsed) {
      totals.set(label, (totals.get(label) || 0) + elapsed);
    }
    async function renderTick(tick, profile = false) {
      const start = performance.now();
      const { scene, camera } = motionPose(tick / 60, count, study);
      const authored = performance.now();
      const solution = await renderer.prepareScene(scene, camera, resolution);
      const solved = performance.now();
      renderer.applyScene(scene, solution, start);
      const ready = performance.now();
      renderer.draw(camera, 1, resolution);
      const submitted = performance.now();
      renderer.synchronize();
      const finished = performance.now();
      if (profile) {
        solution.counts?.forEach((v,i)=>workCounts[i]+=v);
        grouped += renderer.frameStats.grouped;
        covered += renderer.frameStats.covered;
        runs += renderer.frameStats.runs;
        lampTests += renderer.frameStats.lampTests;
        arena = Math.max(
          arena,
          renderer.countedBytes?.() ?? renderer.frameStats.bytes +
            (renderer.math.curved?.wasm.memory.buffer.byteLength||0) +
            renderer.stats.workerBytes +
            renderer.stats.bytes,
        );
        const t = solution.timings,
          charts = renderer.view.chartMs||0;
        record('Scene, basis and view setup', authored - start + t.basis + ready - solved);
        record('Exchange and overlapping geometry', solved - authored - t.basis - t.solve);
        record('Equilibrium and directional light', t.solve);
        record('Visible light fields', charts);
        record('View geometry', renderer.viewMs - charts);
        record('Draw submission and upload', submitted - ready - renderer.viewMs);
        record('Wait for finished 4K pixels', finished - submitted);
        record('Concurrent CPU scanline visibility', renderer.frameStats.scanMs || 0);
        record('Longest transport worker (overlapping)', t.workerMs);
        record('Within exchange · sparse operator assembly', t.assemblyMs);
        if(solution.modal){record('Surface coefficient construction', t.modalBuild);record('Surface coefficient solve and residual check', t.modalApply);}
        record('Within exchange · lamp visibility', t.visibility);
        record('Within exchange · lamp input', t.lamp);
        record('Within exchange · contact geometry and submission', t.contactMs || 0);
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
    if(production)for(const [i,label] of ['Generator slices','Lit endpoints','Analytic length integrals','Spherical evaluations','Rounded boundary feature evaluations','Adaptive splits','Unresolved error estimates','Clear angular apertures in updated interactions','Reused cylinder tangent equations','Regional moment evaluations','Regionally excluded interactions','Domains crossing a facing boundary','Domains outside the expansion tolerance','Domains with unresolved blockers','Point or curved interactions','Integrated source moment fields','Lamp moment slices','Exact conic refinements','Reused current-pose lamp slices','Evaluated current-pose lamp slices'].entries())
      report(label+' per moving frame',Math.round(workCounts[i]/ticks).toLocaleString());
    report('Moving window geometry', windowControl.selectedOptions[0].textContent);
    report('Covered pixels classified in runs', ((100 * grouped) / covered).toFixed(2) + '%');
    report('Exact lamp segment tests for frame visibility', Math.round(lampTests / ticks).toLocaleString());
    report('Peak counted transport, field and scan buffers', (arena / 1048576).toFixed(2) + ' MiB');
    report('Horizontal runs per frame', Math.round(runs / ticks).toLocaleString());
    report('Additional moving solids', String(count));
    report('Contact integrator', integrator.selectedOptions[0].textContent);
    report('Fully lit frames per second', ((ticks * 1000) / elapsed).toFixed(1));
    report('Moving poses finished', ticks + ' / ' + ticks);
    report('Total replay time', (elapsed / 1000).toFixed(2) + ' s');
    report('Median frame work', ms((times[ticks / 2 - 1] + times[ticks / 2]) / 2));
    report('95th percentile frame work', ms(times[Math.ceil(ticks * 0.95) - 1]));
    report('Worst frame work', ms(times.at(-1)));
    report('Frames within 1 ms of work', times.filter((t) => t <= 1).length + ' / ' + ticks);
    report('Frames over 16.7 ms of work', times.filter((t) => t > deadline).length + ' / ' + ticks);
    report('First pose', ms(cold));
    // Means add to the complete mean. The nested exchange timings overlap.
    profile.hidden = false;
    for (const [name, sum] of totals) report(name, ms(sum / ticks), stages);
    if (backend) {
      const s = backend.stats;
      report(
        production?'Submitted GPU contour integrals per moving frame':'Submitted GPU sector integrals per moving frame',
        Math.round(((production?1:16) * (s.polygons - initialStats.polygons)) / ticks).toLocaleString(),
      );
      if (backend.beginChart) {
        report(
          'Contact uploads per moving frame',
          ((s.uploadBytes - initialStats.uploadBytes) / ticks / 1024).toFixed(0) + ' KiB',
        );
        report('Peak contact texture and buffer payload', (s.peakBytes / 1048576).toFixed(2) + ' MiB');
        report(
          'Stationary polygons integrated on GPU, including first pose',
          s.staticPolygons.toLocaleString(),
        );
      } else if(!production) {
        report('Packing per frame', ms((s.packMs - initialStats.packMs) / ticks));
        report(
          'GPU submission through readback',
          ms((s.dispatchReadMs - initialStats.dispatchReadMs) / ticks),
        );
        report('Result reduction per frame', ms((s.reduceMs - initialStats.reduceMs) / ticks));
      }
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
    renderer?.dispose();
    windowControl.disabled = false;
    bodies.disabled = false;
    integrator.disabled = false;
    run.disabled = false;
  }
});
