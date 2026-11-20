import { compilePlaneShadows } from './plane-shadow.js';
import { ReceiverRegions } from './receiver-regions.js';

const integrationTolerance = 2e-4;
export class SourceIntegrals {
  constructor(angular) {
    this.angular = angular;
    this.pool = [];
    this.regions = new ReceiverRegions();
  }
  dispose() {
    for (const slot of this.pool) {
      slot.reject?.(Error('Renderer disposed'));
      slot.worker.terminate();
    }
  }
  async integrate(geometry) {
    const start = performance.now(),
      { receivers, packet } = geometry;
    const n = receivers.length;
    const count = Math.min(n, 6, Math.max(1, (navigator.hardwareConcurrency || 2) - 1));
    while (this.pool.length < count)
      this.pool.push({
        worker: new Worker(new URL('./function-worker.js', import.meta.url), { type: 'module' }),
      });
    const flags = await this.regions.build(packet, integrationTolerance);
    if (packet.lampIds) packet.planeField = compilePlaneShadows(geometry);
    const prepared = performance.now();
    const parts = await Promise.all(
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
    const received = performance.now();
    const angular = this.angular;
    let links = 0,
      bytes =
        (this.regions?.wasm.memory.buffer.byteLength || 0) +
        (packet.regions?.byteLength || 0) +
        (packet.moments?.byteLength || 0) +
        (packet.lampIds?.byteLength || 0) +
        (packet.planeField?.byteLength || 0) +
        (packet.lampEvents?.byteLength || 0) +
        packet.domains.byteLength +
        packet.jacobians.byteLength +
        packet.shapes.byteLength +
        packet.tiles.byteLength +
        packet.nodes.byteLength +
        packet.points.byteLength,
      newJobs = 0;
    const jobCount = parts.reduce((sum, p) => sum + p.jobs.length, 0);
    angular.reserve(jobCount);
    angular.setGeometry(geometry);
    for (const part of parts) {
      part.counts.forEach((x, i) => (flags[i] += x));
      links += part.col.length;
      for (const array of [
        part.starts,
        part.col,
        part.weights,
        part.jobs,
        part.jobStarts,
        part.direct,
        part.lamp,
      ])
        bytes += array.byteLength;
      for (let row = 0; row + 1 < part.jobStarts.length; row++) {
        const receiver = part.first + row * part.stride;
        for (let j = part.jobStarts[row]; j < part.jobStarts[row + 1]; j++) {
          angular.packed[2 * newJobs] = part.jobs[j];
          angular.packed[2 * newJobs + 1] = receiver;
          newJobs++;
        }
      }
    }
    angular.integrate(newJobs);
    let job = 0;
    const positions = new Uint32Array(geometry.vertices.length);
    bytes += positions.byteLength;
    for (const part of parts)
      for (let row = 0; row + 1 < part.jobStarts.length; row++) {
        for (let k = part.starts[row]; k < part.starts[row + 1]; k++) positions[part.col[k]] = k;
        for (let j = part.jobStarts[row]; j < part.jobStarts[row + 1]; j++, job++) {
          let tile = part.jobs[j],
            size = tile < 0 ? 1 : 4;
          if (tile < 0) tile = ~tile;
          else {
            tile >>= 3;
          }
          for (let k = 0; k < size; k++)
            part.weights[positions[packet.nodes[4 * tile + k]]] += angular.values[4 * job + k];
        }
      }
    const assembled = performance.now();
    let maxMass = 0;
    for (const p of parts)
      for (let row = 0; row + 1 < p.starts.length; row++) {
        let mass = 0;
        for (let j = p.starts[row]; j < p.starts[row + 1]; j++)
          mass += p.weights[j] = Math.max(0, p.weights[j]);
        maxMass = Math.max(maxMass, mass);
      }
    return {
      parts,
      maxMass,
      bytes,
      links,
      newJobs,
      counts: flags,
      workerBytes: parts.reduce((a, p) => a + p.bytes, 0),
      timings: {
        basis: prepared - start,
        workerMs: Math.max(...parts.map((p) => p.ms)),
        assemblyMs: assembled - received,
      },
    };
  }
}
