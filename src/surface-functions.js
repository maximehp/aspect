// The general light field is L(surface position, outgoing direction, channel).
// Here a Lambertian material compiles to rho/pi times incident irradiance.
// The directional transport kernel accounts for visibility and projected area.
// Surface charts are integration domains on the analytic solid, never draw primitives.
import { dot, cross, unit } from './cpu-geometry.js';
import { continuousBasis, leaves, hats } from './field-domain.js';
const scale = (a, k) => a.map((v) => v * k);
const add = (a, ...bs) => a.map((v, i) => bs.reduce((s, b) => s + b[i], v));
const norm = (a) => Math.hypot(...a);
const halfPi = Math.PI / 2;

// Natural charts are dictated solely by the authored equation. Integration
// points, approximation functions and their coefficients have distinct roles.
export function chartPoint(g, a, b) {
  const { c, ax, u, v, n, width, height, r, type } = g;
  if (!type) return [add(c, scale(ax, (2 * a - 1) * width), scale(v, (2 * b - 1) * height)), n];
  const arc = add(
    scale(u, b === 1 ? 0 : Math.cos(b * halfPi)),
    scale(v, b === 0 ? 0 : Math.sin(b * halfPi)),
  );
  if (type === 1) return [add(c, scale(ax, (2 * a - 1) * width), scale(arc, r)), arc];
  const normal = add(
    scale(arc, a === 1 ? 0 : Math.cos(a * halfPi)),
    scale(ax, a === 0 ? 0 : Math.sin(a * halfPi)),
  );
  return [add(c, scale(normal, r)), normal];
}
export function authoredCharts(scene) {
  const { shapes } = scene,
    charts = [],
    maps = [];
  for (const [object, s] of shapes.entries()) {
    const map = [];
    maps.push(map);
    function chart(slot, type, c, ax, u, v, n, width, height, r) {
      const g = { slot, type, c, ax, u, v, n, width, height, r, object, offset: charts.length, shape: s };
      g.empty =
        (!type && (!width || !height)) ||
        (type === 1 && !width) ||
        (!s.axes &&
          shapes.every((o) => {
            if (o === s) return true;
            const span = o.axes
              ? o.radius +
                o.axes.reduce((sum, a, i) => sum + Math.abs(dot(n, a)) * (o.halfSize[i] - o.radius), 0)
              : Math.abs(dot(n, o.axisU)) + Math.abs(dot(n, o.axisV));
            return (
              dot(
                n,
                o.center.map((x, k) => x - c[k]),
              ) +
                span <=
              0
            );
          }));
      charts.push(g);
      map[slot] = g;
    }
    if (!s.axes) {
      const U = norm(s.axisU),
        V = norm(s.axisV),
        n = unit(cross(s.axisU, s.axisV));
      // Each affine source uses a unit generator. Its transverse derivative
      // contributes the area Jacobian separately for skew authored rectangles.
      for (let side = 0; side < 2; side++)
        chart(side, 0, s.center, unit(s.axisU), [0, 0, 0], unit(s.axisV), scale(n, side ? -1 : 1), U, V, 0);
      continue;
    }
    const { axes, h } = { axes: s.axes, h: s.halfSize.map((x) => x - s.radius) },
      r = s.radius;
    for (let d = 0; d < 3; d++) {
      const a = (d + 1) % 3,
        b = (d + 2) % 3;
      for (let side = 0; side < 2; side++) {
        const n = scale(axes[d], side ? -1 : 1);
        chart(
          2 * d + side,
          0,
          add(s.center, scale(n, s.halfSize[d])),
          axes[a],
          [0, 0, 0],
          axes[b],
          n,
          h[a],
          h[b],
          0,
        );
      }
      for (let k = 0; k < 4; k++) {
        const u = scale(axes[a], k & 2 ? -1 : 1),
          v = scale(axes[b], k & 1 ? -1 : 1);
        chart(
          6 + 4 * d + k,
          1,
          add(s.center, scale(u, h[a]), scale(v, h[b])),
          axes[d],
          u,
          v,
          [0, 0, 0],
          h[d],
          0,
          r,
        );
      }
    }
    for (let k = 0; k < 8; k++) {
      const [u, v, z] = axes.map((a, j) => scale(a, k & (1 << j) ? -1 : 1));
      chart(
        18 + k,
        2,
        add(s.center, scale(u, h[0]), scale(v, h[1]), scale(z, h[2])),
        z,
        u,
        v,
        [0, 0, 0],
        0,
        0,
        r,
      );
    }
  }

  return { shapes, charts, maps };
}

// A shared edge has one trace of the finite-element function. Carry a new
// knot across all incident faces, including the seam of a round fillet.
function positionKey(g, u, v) {
  const { shape, slot, object } = g;
  if (!shape.axes) return [object, slot, u, v].join('/');
  // Symbolic seam identity, including the collapsed pole of an octant.
  // Incident charts agree exactly even if their trigonometric evaluations
  // of the same point round to neighbouring IEEE values.
  const core = [0, 0, 0];
  let normal;
  function arc(d, i, j, si, sj, t) {
    if (t === 0) return 'face/' + i + '/' + si;
    if (t === 1) return 'face/' + j + '/' + sj;
    return 'arc/' + d + '/' + si + '/' + sj + '/' + t;
  }
  if (slot < 6) {
    const d = slot >> 1,
      i = (d + 1) % 3,
      j = (d + 2) % 3,
      sign = slot & 1 ? -1 : 1;
    core[d] = sign;
    core[i] = 2 * u - 1;
    core[j] = 2 * v - 1;
    normal = 'face/' + d + '/' + sign;
  } else if (slot < 18) {
    const d = (slot - 6) >> 2,
      i = (d + 1) % 3,
      j = (d + 2) % 3,
      k = (slot - 6) & 3;
    core[d] = 2 * u - 1;
    core[i] = k & 2 ? -1 : 1;
    core[j] = k & 1 ? -1 : 1;
    normal = arc(d, i, j, core[i], core[j], v);
  } else {
    for (let i = 0; i < 3; i++) core[i] = (slot - 18) & (1 << i) ? -1 : 1;
    if (u === 1) normal = 'face/2/' + core[2];
    else if (u === 0) normal = arc(2, 0, 1, core[0], core[1], v);
    else if (v === 0) normal = arc(1, 2, 0, core[2], core[0], 1 - u);
    else if (v === 1) normal = arc(0, 1, 2, core[1], core[2], u);
    else normal = ['octant', slot, u, v].join('/');
  }
  for (let i = 0; i < 3; i++) if (shape.halfSize[i] === shape.radius) core[i] = 0;
  return [object, ...core, normal].join('/');
}
export function fieldGeometry(scene, domains, roots, order = 3) {
  const { shapes, charts, maps } = domains,
    tiles = [],
    receivers = [],
    cells = [],
    groups = [];
  charts.forEach((g, i) => (g.domain = roots[i]));
  const { vertices, supports } = continuousBasis(charts, positionKey);
  const X = order === 3 ? [-Math.sqrt(0.6), 0, Math.sqrt(0.6)] : [-Math.sqrt(1 / 3), Math.sqrt(1 / 3)];
  const W = order === 3 ? [5 / 9, 8 / 9, 5 / 9] : [1, 1];
  for (const g of charts) {
    if (g.empty) continue;
    for (const leaf of leaves(g.domain)) {
      const [a, b, l, h] = leaf.bounds,
        m = (a + b) / 2,
        t = (l + h) / 2,
        ids = leaf.ids;
      let c = g.c;
      if (!g.type) c = chartPoint(g, m, t)[0];
      else if (g.type === 1) c = add(g.c, scale(g.ax, (2 * m - 1) * g.width));
      const tile = {
        ...g,
        c,
        width: (b - a) * g.width,
        height: (h - l) * g.height,
        bounds: [a, b, l, h],
        nodes: ids,
      };
      tiles.push(tile);
      let group = 0;
      if (!g.type) {
        groups.push({
          c,
          a: scale(g.ax, tile.width),
          b: scale(g.v, tile.height),
          n: g.n,
          object: g.object,
        });
        group = groups.length;
      }
      const first = receivers.length;
      for (let y = 0; y < X.length; y++)
        for (let x = 0; x < X.length; x++) {
          const A = (1 + X[x]) / 2,
            B = (1 + X[y]) / 2,
            u = a + (b - a) * A,
            v = l + (h - l) * B,
            [p, n] = chartPoint(g, u, v);
          const J =
            g.type === 0
              ? 4 * g.width * g.height * norm(cross(g.ax, g.v))
              : g.type === 1
                ? 2 * g.width * g.r * halfPi
                : g.r * g.r * halfPi * halfPi * Math.cos(u * halfPi);
          const hat = hats(A, B),
            test = new Map();
          for (let k = 0; k < 4; k++)
            for (const [id, w] of vertices[ids[k]].basis) test.set(id, (test.get(id) || 0) + hat[k] * w);
          receivers.push({
            p,
            n,
            group,
            object: g.object,
            ids,
            hat,
            test: [...test],
            area: (b - a) * (h - l) * 0.25 * W[x] * W[y] * J,
          });
        }
      cells.push({ g: g.offset, leaf, first, end: receivers.length });
    }
  }
  const domainsBuffer = new ArrayBuffer(104 * groups.length);
  groups.forEach((g, i) => {
    new Float64Array(domainsBuffer, i * 104, 12).set([...g.c, ...g.a, ...g.b, ...g.n]);
    new Int32Array(domainsBuffer, i * 104 + 96, 2)[0] = g.object;
  });
  const tileBuffer = new ArrayBuffer(200 * tiles.length);
  tiles.forEach((c, i) => writeChart(tileBuffer, 200 * i, c));
  const points = new ArrayBuffer(56 * receivers.length);
  receivers.forEach((r, i) => writeReceiver(points, 56 * i, r));
  return {
    shapes,
    charts,
    maps,
    tiles,
    receivers,
    groups,
    vertices,
    supports,
    cells,
    packet: {
      domains: domainsBuffer,
      sourceCount: vertices.length,
      jacobians: Float64Array.from(tiles, (t) => (t.type ? 1 : norm(cross(t.ax, t.v)))),
      definitions: shapes.map((s) => s.definition ?? null),
      shapes: scene.records,
      tiles: tileBuffer,
      points,
      lamp: new Float64Array(scene.lightPosition),
      nodes: Uint32Array.from(tiles.flatMap((t) => t.nodes)),
      colors: Float64Array.from(shapes.flatMap((s) => s.color)),
      lampRGB: scene.lampRGB,
    },
  };
}
export function writeChart(buffer, offset, c) {
  new Float64Array(buffer, offset, 18).set([
    ...c.c,
    ...c.ax,
    ...c.u,
    ...c.v,
    ...c.n,
    c.width,
    c.height,
    c.r,
  ]);
  new Int32Array(buffer, offset + 144, 6).set([c.type, c.object, c.slot, 0, 0, 0]);
  new Float64Array(buffer, offset + 168, 4).set(c.bounds.map((x) => x * halfPi));
}
export function writeReceiver(buffer, offset, { p, n, object, group = 0 }) {
  new Float64Array(buffer, offset, 6).set([...p, ...n]);
  new Int32Array(buffer, offset + 48, 2).set([object, group]);
}
