import { conicImports } from './silhouette.js';
const compiled = WebAssembly.compileStreaming(fetch(new URL('./function-transport.wasm', import.meta.url)));
let engine, current;
self.onmessage = async ({ data: { packet, first, stride, tolerance } }) => {
  try {
    current = packet;
    engine ??= new WebAssembly.Instance(await compiled, {
      math: Math,
      geometry: conicImports(() => ({ packet: current, memory: engine.memory })),
    }).exports;
    const { shapes, tiles, points, lamp } = packet,
      n = points.byteLength / 56,
      k = tiles.byteLength / 168;
    const rows = Math.ceil((n - first) / stride),
      count = k * rows;
    let cursor = Number(engine.__heap_base);
    function space(len) {
      const p = cursor;
      cursor += Math.ceil(len / 8) * 8;
      return p;
    }
    const src = space(shapes.byteLength),
      chart = space(tiles.byteLength),
      dest = space(points.byteLength),
      light = space(24),
      out = space(count * 20),
      jobs = space(count),
      counts = space(160),
      sliceCache = space(64 * 16384),
      regions = space(packet.regions?.byteLength || 0),
      moments = space(packet.moments?.byteLength || 0),
      catalog = space(packet.lampIds?.byteLength || 0);
    if (cursor > engine.memory.buffer.byteLength)
      engine.memory.grow(Math.ceil((cursor - engine.memory.buffer.byteLength) / 65536));
    const mem = engine.memory.buffer;
    new Float64Array(mem, src, shapes.length).set(shapes);
    new Uint8Array(mem, chart, tiles.byteLength).set(new Uint8Array(tiles));
    new Uint8Array(mem, dest, points.byteLength).set(new Uint8Array(points));
    new Float64Array(mem, light, 3).set(lamp);
    new Uint8Array(mem, counts, 160).fill(0);
    if (packet.regions) {
      new Uint8Array(mem, regions, packet.regions.length).set(packet.regions);
      new Float64Array(mem, moments, packet.moments.length).set(packet.moments);
      new Int32Array(mem, catalog, packet.lampIds.length).set(packet.lampIds);
    }

    const start = performance.now();
    engine.transport(
      src,
      shapes.length / 19,
      chart,
      k,
      dest,
      n,
      light,
      first,
      stride,
      tolerance,
      out,
      jobs,
      counts,
      packet.regions ? regions : 0,
      moments,
      packet.regions ? catalog : 0,
      sliceCache,
    );
    const kernelMs = performance.now() - start;
    const values = new Float32Array(mem, out, 5 * count),
      gpu = new Uint8Array(mem, jobs, count);
    const { nodes, colors, lampRGB } = packet,
      chartIds = new Int32Array(tiles),
      direct = new Float64Array(rows * 3);
    const starts = new Uint32Array(rows + 1),
      indices = [],
      weights = [],
      angular = [],
      sourceCount = packet.sourceCount ?? n,
      local = new Float64Array(sourceCount),
      positions = new Int32Array(sourceCount);
    const needed = new Uint8Array(sourceCount);
    for (let row = 0; row < rows; row++) {
      local.fill(0);
      needed.fill(0);
      let pending = angular.length;
      for (let t = 0; t < k; t++) {
        const ix = row * k + t,
          b = 5 * ix,
          object = chartIds[42 * t + 37];
        if (!values[b] && !values[b + 1] && !values[b + 2] && !values[b + 3] && !values[b + 4] && !gpu[ix])
          continue;
        const jac = packet.jacobians[t];
        for (let j = 0; j < 4; j++) local[nodes[4 * t + j]] += jac * values[b + j];
        for (let c = 0; c < 3; c++)
          direct[3 * row + c] += jac * values[b + 4] * colors[3 * object + c] * lampRGB[c];
        if (gpu[ix]) {
          angular.push(row, t, nodes[4 * t]);
          needed[nodes[4 * t]] = 1;
        }
      }
      for (let j = 0; j < sourceCount; j++)
        if (local[j] || needed[j]) {
          positions[j] = indices.length;
          indices.push(j);
          weights.push(local[j]);
        }
      for (let z = pending + 2; z < angular.length; z += 3) angular[z] = positions[angular[z]];
      starts[row + 1] = indices.length;
    }
    const result = {
      first,
      stride,
      starts,
      direct,
      col: Uint32Array.from(indices),
      weights: Float64Array.from(weights),
      jobs: Uint32Array.from(angular),
      counts: Array.from(new BigInt64Array(mem, counts, 20), Number),
      bytes: mem.byteLength,
      ms: kernelMs,
    };
    current = null;
    self.postMessage(
      result,
      Object.values(result)
        .filter(ArrayBuffer.isView)
        .map((a) => a.buffer),
    );
  } catch (e) {
    current = null;
    self.postMessage({ error: e.stack || e.message });
  }
};
