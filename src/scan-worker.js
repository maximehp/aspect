import { Scanlines } from './fillet-transport.js';
const ready = WebAssembly.compileStreaming(fetch(new URL('./fillet-kernel.wasm', import.meta.url))).then(
  (module) => new Scanlines(module),
);
self.onmessage = async ({ data }) => {
  try {
    const scan = await ready,
      start = performance.now();
    scan.prepare(data.scene);
    const result = scan.render(data.camera, ...data.size);
    result.runs = result.runs.slice();
    result.bytes = scan.wasm.memory.buffer.byteLength;
    result.ms = performance.now() - start;
    self.postMessage(result, [result.runs.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
