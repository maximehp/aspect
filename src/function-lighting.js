import { selectPairs, markPairs, weakPairs, projectedRows, mergeParts } from './weak.js';
import { equilibriumSolve } from './krylov.js';
import { modalEquilibrium } from './modal-equilibrium.js';
import { ReceiverRegions } from './receiver-regions.js';
import { compileSurfaces } from './surface-functions.js';

const integrationTolerance = 2e-4;
export class FunctionLighting {
  constructor(angular, { regions = true, expanded = false } = {}) {
    this.angular = angular;
    this.expanded = expanded;
    this.pool = [];
    this.regions = regions ? new ReceiverRegions() : null;
  }
  dispose() {
    for (const slot of this.pool) {
      slot.reject?.(Error('Renderer disposed'));
      slot.worker.terminate();
    }
  }
  async update(scene) {
    const start = performance.now(),
      geometry = compileSurfaces(scene),
      { receivers, tiles, packet } = geometry;
    const n = receivers.length;
    const count = Math.min(n, 6, Math.max(1, (navigator.hardwareConcurrency || 2) - 1));
    while (this.pool.length < count)
      this.pool.push({
        worker: new Worker(new URL('./function-worker.js', import.meta.url), { type: 'module' }),
      });
    const flags = this.regions ? await this.regions.build(packet, integrationTolerance) : Array(20).fill(0);
    const selected = selectPairs(geometry);
    markPairs(geometry, selected);
    const prepared = performance.now();
    const pending = Promise.all(
      this.pool.slice(0, count).map(
        (slot, first) =>
          new Promise((resolve, reject) => {
            slot.reject = reject;
            slot.worker.onmessage = ({ data }) => {
              slot.reject = null;
              data.error ? reject(Error(data.error)) : resolve(data);
            };
            slot.worker.onerror = (e) => reject(Error(e.message));
            slot.worker.postMessage({ packet, first, stride: count, tolerance: integrationTolerance });
          }),
      ),
    );
    const pairs = weakPairs(geometry, selected);
    const extra = projectedRows(geometry, pairs, 'positive');
    let parts = await pending;
    const received = performance.now();
    const angular = this.angular;
    let links = 0,
      bytes =
        (this.regions?.wasm.memory.buffer.byteLength || 0) +
        (packet.regions?.byteLength || 0) +
        (packet.moments?.byteLength || 0) +
        (packet.lampIds?.byteLength || 0) +
        packet.domains.byteLength +
        packet.jacobians.byteLength +
        packet.shapes.byteLength +
        packet.tiles.byteLength +
        packet.nodes.byteLength +
        packet.points.byteLength,
      newJobs = 0;
    const jobCount = parts.reduce((sum, p) => sum + p.jobs.length / 3, 0);
    angular.reserve(jobCount);
    for (const part of parts) {
      part.counts.forEach((x, i) => (flags[i] += x));
      links += part.col.length;
      for (const array of [part.starts, part.col, part.weights, part.jobs, part.direct])
        bytes += array.byteLength;
      for (let j = 0; j < part.jobs.length; j += 3) {
        const i = part.first + part.stride * part.jobs[j],
          t = part.jobs[j + 1];
        angular.put(newJobs++, tiles[t], receivers[i]);
      }
    }
    angular.integrate(newJobs);
    let job = 0;
    for (const part of parts)
      for (let j = 0; j < part.jobs.length; j += 3) {
        const weight = angular.values[job++];
        part.weights[part.jobs[j + 2]] += weight;
      }
    mergeParts(parts, extra);
    bytes +=
      2 * (extra.col.byteLength + extra.weights.byteLength + extra.starts.byteLength) +
      pairs.reduce((n, p) => n + 3 * p.weights.byteLength + p.massA.byteLength + p.massB.byteLength, 0);
    const assembled = performance.now(),
      direct = new Float64Array(3 * n),
      color = new Float64Array(3 * n);
    let maxMass = 0;
    for (const p of parts)
      for (let row = 0; row + 1 < p.starts.length; row++) {
        const i = p.first + p.stride * row;
        direct.set(p.direct.subarray(3 * row, 3 * row + 3), 3 * i);
        color.set(geometry.shapes[receivers[i].object].color, 3 * i);
        let mass = 0;
        for (let j = p.starts[row]; j < p.starts[row + 1]; j++)
          mass += p.weights[j] = Math.max(0, p.weights[j]);
        maxMass = Math.max(maxMass, mass);
      }
    const solveStart = performance.now();
    const solved = this.expanded
      ? equilibrium(parts, direct, color)
      : await modalEquilibrium(geometry, parts, direct, color);
    const { incident: H, iterations, residual, bytes: solveBytes, modal } = solved;
    const contraction = maxMass * color.reduce((m, x) => Math.max(m, x), 0);
    const equilibriumError = contraction < 1 ? residual / (1 - contraction) : Infinity;
    const finished = performance.now();
    bytes += solveBytes + 2 * direct.byteLength;
    return {
      geometry,
      rows: parts,
      modal,
      incident: H,
      bytes,
      workerBytes: parts.reduce((a, p) => a + p.bytes, 0),
      counts: flags,
      maxMass,
      iterations,
      residual,
      links,
      newJobs,
      firstBounce: direct,
      equilibriumError,
      timings: {
        basis: prepared - start,
        workerMs: Math.max(...parts.map((p) => p.ms)),
        assemblyMs: assembled - received,
        solve: finished - assembled,
        operatorMs: solveStart - start + (modal?.buildMs || 0),
        modalBuild: modal?.buildMs || 0,
        modalApply: modal?.solveMs || 0,
        visibility: 0,
        lamp: 0,
        contactMs: 0,
      },
    };
  }
}

// Minimize the residual over a short Krylov basis. RGB are independent
// equations stored as one block-diagonal system. Check the actual equilibrium
// residual before publishing a pose, including a scene with zero input.
function equilibrium(parts, direct, color) {
  const x = direct.slice();
  const n = x.length,
    emitted = new Float64Array(n);
  function apply(v, result) {
    for (let i = 0; i < n; i++) emitted[i] = color[i] * v[i];
    for (const p of parts)
      for (let row = 0; row + 1 < p.starts.length; row++) {
        const i = 3 * (p.first + row * p.stride);
        let a = v[i],
          b = v[i + 1],
          c = v[i + 2];
        for (let k = p.starts[row]; k < p.starts[row + 1]; k++) {
          const j = 3 * p.col[k],
            w = p.weights[k];
          a -= w * emitted[j];
          b -= w * emitted[j + 1];
          c -= w * emitted[j + 2];
        }
        result[i] = a;
        result[i + 1] = b;
        result[i + 2] = c;
      }
  }
  return equilibriumSolve(direct, apply, 1e-9, x);
}
