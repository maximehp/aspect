import { directionCount } from './beam-exchange.js';

// Angular contributions are independent. Reduce in direction order even when
// workers finish out of order; presentation always receives a complete solve.
export class BeamWorkers {
  constructor(count = Math.min(6, Math.max(1, (navigator.hardwareConcurrency || 2) - 1))) {
    this.pending = new Set();
    this.workers = [];
    try {
      for (let i = 0; i < count; i++)
        this.workers.push(new Worker(new URL('./beam-worker.js', import.meta.url), { type: 'module' }));
    } catch (error) {
      this.dispose();
      throw error;
    }
  }
  integrate(packet, fixed, changed) {
    return Promise.all(
      this.workers.map(
        (worker, i) =>
          new Promise((resolve, reject) => {
            this.pending.add(reject);
            worker.onmessage = ({ data }) => {
              this.pending.delete(reject);
              if (data.error) reject(Error(data.error));
              else resolve(data);
            };
            worker.onerror = (e) => {
              this.pending.delete(reject);
              this.failed = true;
              reject(Error(e.message));
            };
            worker.postMessage({
              packet,
              fixed,
              changed,
              first: Math.floor((i * directionCount) / this.workers.length),
              end: Math.floor(((i + 1) * directionCount) / this.workers.length),
            });
          }),
      ),
    );
  }
  dispose() {
    for (const reject of this.pending) reject(Error('Beam worker stopped.'));
    this.pending.clear();
    for (const w of this.workers) w.terminate();
  }
}
