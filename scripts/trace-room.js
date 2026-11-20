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
    stats = alloc(160),
    planeField = alloc(packet.planeField.byteLength);
  if (end > wasm.memory.buffer.byteLength)
    wasm.memory.grow(Math.ceil((end - wasm.memory.buffer.byteLength) / 65536));
  const memory = wasm.memory.buffer;
  new Float64Array(memory, shapes, packet.shapes.length).set(packet.shapes);
  new Float64Array(memory, light, 3).set(packet.lamp);
  const tile = field.geometry.tiles.findIndex((t) => t.offset === chart.offset);
  const program = new Float64Array(memory, planeField, packet.planeField.length);
  program.set(packet.planeField);
  // Restrict the same plane-wide program to the whole chart for this query.
  program[4 * tile + 1] = program[4 * tile + 2] = 0;
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
            planeField,
            tile,
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

// Restrict the playable biaffine visibility function to one point on the wall.
// Its four edge functions are now lines in normalized floor coordinates.
export function sheetShadowSection(geometry, source, uv, receiver, blocker) {
  const { sheetForms: f } = geometry.packet;
  function position(g, u, v) {
    const k = f[g.offset] + 9 * blocker,
      s = (2 * u - 1) * g.width,
      t = (2 * v - 1) * g.height;
    return [0, 3, 6].map((i) => f[k + i] + s * f[k + i + 1] + t * f[k + i + 2]);
  }
  const x = position(source, ...uv),
    r = position(receiver, 0.5, 0.5)[0],
    sign = Math.sign(r);
  if (!sign || [0, 1].some((u) => [0, 1].some((v) => sign * position(receiver, u, v)[0] <= 0))) return null;
  if (sign * x[0] >= 0) return { crossesSheetPlane: false };
  function edges(u, v) {
    const y = position(receiver, u, v);
    return [1, 2].flatMap((k) =>
      [-1, 1].map((s) => sign * (y[0] - x[0] + s * (y[0] * x[k] - x[0] * y[k]))),
    );
  }
  const O = edges(0, 0),
    U = edges(1, 0),
    V = edges(0, 1);
  return {
    wallUV: uv,
    blocker: geometry.shapes[blocker].id,
    halfPlanes: O.map((c, i) => [U[i] - c, V[i] - c, c]),
  };
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
  const domainPairs = {
    wholeDomainMoments: 0,
    excluded: 0,
    crossesHorizon: 0,
    nearField: 0,
    visibilityChangesOrUnproved: 0,
  };
  for (const cell of g.cells) {
    if (cell.g !== g.maps[floor][1].offset) continue;
    const group = g.receivers[cell.first].group - 1;
    g.tiles.forEach((tile, k) => {
      if (tile.offset !== chart.offset) return;
      const mode = g.packet.regions[group * g.tiles.length + k];
      const reason = [
        '',
        'wholeDomainMoments',
        'excluded',
        'crossesHorizon',
        'nearField',
        'visibilityChangesOrUnproved',
      ][mode];
      if (reason) domainPairs[reason]++;
    });
  }
  return {
    region,
    wallFloorDomainPairs: domainPairs,
    wallToFloorIntegratedLines: solution.axisCounts.get(floor * g.shapes.length + wall) || 0,
    wallSections: await shadowSections(illumination, chart, [0.2, 0.35, 0.45, 0.5, 0.55, 0.65, 0.8]),
    wallToFloorVisibility: sheetShadowSection(
      g,
      chart,
      [0.12, 0.35],
      g.maps[floor][1],
      id('bench-seat-top'),
    ),
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

// Serialize the incident functions for an independent last-leg measurement.
export function referenceSnapshot(scene, { illumination, pose }) {
  return {
    scene,
    ...(pose?.authoredToSurface && pose.shapes.length !== pose.authoredToSurface.length
      ? {
          authoredToSurface: Array.from(pose.authoredToSurface),
          compiledShapes: Array.from(pose.records),
        }
      : {}),
    charts: illumination.geometry.maps.map((m) => m.map(({ domain, empty }) => ({ domain, empty }))),
    indirect: Array.from(illumination.indirect),
  };
}

// For an isolated emitting surface, integral(received over every other surface)
// cannot exceed integral(emitted). This checks the reverse direction from the
// unit-hemisphere bound at each receiver. Use the actual playable operator.
export function auditEnergy(solution) {
  const { illumination, operator, firstBounce } = solution,
    g = illumination.geometry;
  const areas = new Float64Array(g.vertices.length);
  for (const q of g.receivers) for (let k = 0; k < 4; k++) areas[q.ids[k]] += q.area * q.hat[k];
  function integral(values, material = false) {
    const total = [0, 0, 0];
    g.vertices.forEach((v, i) => {
      for (let c = 0; c < 3; c++)
        total[c] += areas[i] * values[3 * i + c] * (material ? g.shapes[v.object].color[c] : 1);
    });
    return total;
  }
  const groups = new Map();
  g.vertices.forEach((v, i) => {
    if (!groups.has(v.component)) groups.set(v.component, []);
    groups.get(v.component).push(i);
  });
  const field = { geometry: g, lamp: null, indirect: new Float64Array(illumination.indirect.length) };
  const transfers = [];
  for (const [component, ids] of groups) {
    let area = 0;
    const object = g.vertices[ids[0]].object,
      rho = g.shapes[object].color;
    for (const i of ids) {
      area += areas[i];
      for (let c = 0; c < 3; c++) field.indirect[3 * i + c] = rho[c] > 0 ? 1 / rho[c] : 0;
    }
    const received = integral(operator.reflect(field, object).indirect);
    transfers.push({
      object: g.shapes[object].id,
      side: g.shapes[object].axes ? 'solid' : component % 2,
      area,
      receivedOverEmitted: received.map((v, c) => (rho[c] > 0 ? v / area : null)),
    });
    for (const i of ids) field.indirect.fill(0, 3 * i, 3 * i + 3);
  }
  const fromIndirect = operator.reflect({ ...illumination, lamp: null });
  return {
    surfaces: transfers,
    indirectEmitted: integral(illumination.indirect, true),
    receivedFromIndirect: integral(fromIndirect.indirect),
    receivedFromLampLitSurfaces: integral(firstBounce.indirect),
    totalIndirectReceived: integral(illumination.indirect),
  };
}
