import { FilletTransport } from './fillet-transport.js';
import { BeamWorkspace } from './beam-exchange.js';
const ready = WebAssembly.instantiateStreaming(fetch(new URL('./beam-sweep.wasm', import.meta.url))).then(
  ({ instance }) => new BeamWorkspace(instance.exports),
);
const curvedReady = WebAssembly.compileStreaming(
  fetch(new URL('./fillet-kernel.wasm', import.meta.url)),
).then((module) => new FilletTransport(module));
self.onmessage = async ({ data }) => {
  try {
    const start = performance.now();
    const core = await ready;
    const result =
      data.surface && !data.changed
        ? { stationary: null, workingBytes: core.wasm.memory.buffer.byteLength }
        : core.integrate(
            data.packet,
            data.fixed,
            data.first,
            data.end,
            data.changed,
            data.stride,
            !!data.surface,
          );
    if (data.surface) {
      const curved = await curvedReady;
      result.change = curved.integrate(data.surface, data.first, data.stride);
      result.workingBytes += curved.wasm.memory.buffer.byteLength;
      result.kernelMs = performance.now() - start;
    }
    const transfer = [result.change.buffer];
    if (result.stationary) transfer.push(result.stationary.buffer);
    self.postMessage(result, transfer);
  } catch (e) {
    self.postMessage({ error: e.message });
  }
};
