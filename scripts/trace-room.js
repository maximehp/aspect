// Observe the same lamp visibility and reflected field used by the playable room.
import { fieldValue, leaves } from '../src/field-domain.js';
import { chartPoint, writeChart } from '../src/surface-functions.js';
import { conicImports } from '../src/silhouette.js';
import { dot, sub } from '../src/cpu-geometry.js';

// The mean of a bilinear function on an axis-aligned subrectangle equals its
// value at the center. Integrate only the intersection of each leaf and region.
function reflectedMean(chart, values, bounds) {
  const [a, b, l, h] = bounds,
    rgb = [0, 0, 0];
  for (const leaf of leaves(chart.domain)) {
    const q = leaf.bounds,
      u = Math.max(a, q[0]),
      v = Math.max(l, q[2]);
    const w = Math.min(b, q[1]) - u,
      t = Math.min(h, q[3]) - v;
    if (w <= 0 || t <= 0) continue;
    const at = fieldValue(chart, u + w / 2, v + t / 2, values);
    for (let c = 0; c < 3; c++) rgb[c] += (at[c] * chart.shape.color[c] * w * t) / ((b - a) * (h - l));
  }
  return rgb;
}

async function shadowSections(field, chart, cuts) {
  const packet = field.lamp,
    n = packet.shapes.length / 19;
  let wasm;
  const imports = { math: Math, geometry: conicImports(() => ({ packet, memory: wasm.memory })) };
  wasm = (
    await WebAssembly.instantiateStreaming(
      fetch(new URL('../src/function-transport.wasm', import.meta.url)),
      imports,
    )
  ).instance.exports;
  let end = Number(wasm.__heap_base);
  const alloc = (size) => {
    const p = end;
    end += 8 * Math.ceil(size / 8);
    return p;
  };
  const ws = alloc(wasm.transportBytes(n)),
    shapes = alloc(packet.shapes.byteLength),
    light = alloc(24),
    surface = alloc(200),
    ids = alloc(4 * n),
    interval = alloc(16 * (n + 1)),
    stats = alloc(160);
  if (end > wasm.memory.buffer.byteLength)
    wasm.memory.grow(Math.ceil((end - wasm.memory.buffer.byteLength) / 65536));
  const memory = wasm.memory.buffer;
  new Float64Array(memory, shapes, packet.shapes.length).set(packet.shapes);
  new Float64Array(memory, light, 3).set(packet.lamp);
  writeChart(memory, surface, { ...chart, bounds: [0, 1, 0, 1] });
  wasm.prepareTransport(ws, shapes, n, light);
  const count = wasm.lampBlockers(ws, shapes, n, surface, ids);
  return cuts.map((v) => {
    const [p, normal] = chartPoint(chart, 0.5, v);
    const k =
      dot(normal, sub(packet.lamp, p)) > 0
        ? wasm.lampIntervals(
            ws,
            shapes,
            surface,
            light,
            ids,
            count,
            (2 * v - 1) * chart.height,
            interval,
            stats,
          )
        : 0;
    const edges = new Float64Array(memory, interval, 2 * k);
    return {
      v,
      litU: Array.from({ length: k }, (_, i) => [
        0.5 + (0.5 * edges[2 * i]) / chart.width,
        0.5 + (0.5 * edges[2 * i + 1]) / chart.width,
      ]),
    };
  });
}

export async function traceRoom(renderer, scene) {
  const solution = await renderer.prepareScene(scene),
    { direct, illumination, operator } = solution;
  const g = illumination.geometry;
  const id = (name) => g.shapes.findIndex((s) => s.id === name);
  const wall = id('north-wall'),
    floor = id('floor'),
    ceiling = id('ceiling');
  const region = [0.5, 0.75, 0.4, 0.6];
  const descriptor = (field) => ({ lamp: null, geometry: g, indirect: field });
  const slatLight = new Float64Array(illumination.indirect.length);
  for (let i = 0; i < g.shapes.length; i++)
    if (g.shapes[i].axes) {
      const from = operator.reflect(direct, i);
      from.indirect.forEach((v, j) => (slatLight[j] += v));
    }
  const viaSlat = descriptor(slatLight);
  const chart = g.maps[wall][0];
  function handoff(field) {
    const toFloor = operator.reflect(field, wall),
      toCeiling = operator.reflect(toFloor, floor);
    return {
      floor: reflectedMean(g.maps[floor][1], toFloor.indirect, region),
      ceiling: reflectedMean(g.maps[ceiling][0], toCeiling.indirect, region),
    };
  }
  const contact = (name, side, area) => reflectedMean(g.maps[id(name)][side], illumination.indirect, area);
  return {
    region,
    wallSections: await shadowSections(illumination, chart, [0.2, 0.35, 0.45, 0.5, 0.55, 0.65, 0.8]),
    wallFromSlats: reflectedMean(chart, slatLight, region),
    fromLampThroughWall: handoff(direct),
    fromSlatsThroughWall: handoff(viaSlat),
    bench: contact('bench-seat-bottom', 0, [0, 0.5, 0, 1]),
    behindArtwork: contact('north-wall', 0, [0.5, (1 + 1.11 / 3.4) / 2, 0.5, (1 + 1.07 / 1.85) / 2]),
    settledResidual: solution.residual,
    closure: operator.closure,
    maxMass: solution.maxMass,
    refinement: solution.history.at(-1),
  };
}
