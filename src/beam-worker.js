import { BeamWorkspace } from './beam-exchange.js';
const ready = WebAssembly.instantiateStreaming(fetch(new URL('./beam-sweep.wasm', import.meta.url))).then(
  ({ instance }) => new BeamWorkspace(instance.exports),
);
self.onmessage = async ({ data }) => {
  try {
    const result = (await ready).integrate(data.packet, data.fixed, data.first, data.end, data.changed);
    const transfer = [result.change.buffer];
    if (result.stationary) transfer.push(result.stationary.buffer);
    self.postMessage(result, transfer);
  } catch (e) {
    self.postMessage({ error: e.message });
  }
};
