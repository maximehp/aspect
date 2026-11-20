// The general light field is L(surface position, outgoing direction, channel).
// Here a Lambertian material compiles to rho/pi times incident irradiance.
// The directional transport kernel accounts for visibility and projected area.
// Surface charts are integration domains on the analytic solid, never draw primitives.
import { dot, cross, unit } from './cpu-geometry.js';
import { shapeRecord } from './fillet-transport.js';
const scale = (a, k) => a.map((v) => v * k);
const add = (a, ...bs) => a.map((v, i) => bs.reduce((s, b) => s + b[i], v));
const norm = (a) => Math.hypot(...a);
const halfPi = Math.PI / 2;

export function compileSurfaces(scene) {
  const shapes = [...scene.rectangles, ...scene.roundedBoxes],
    receivers = [],
    tiles = [],
    charts = [],
    maps = [];
  for (const [object, s] of shapes.entries()) {
    const shared = new Map(),
      map = [];
    maps.push(map);
    function chart(slot, type, c, ax, u, v, n, width, height, r, cols, rows) {
      const ids = [],
        offset = charts.length;
      const empty =
        !s.axes &&
        shapes.every((other) => {
          if (other === s) return true;
          const extent = other.axes
            ? other.radius +
              other.axes.reduce(
                (sum, a, i) => sum + Math.abs(dot(n, a)) * (other.halfSize[i] - other.radius),
                0,
              )
            : Math.abs(dot(n, other.axisU)) + Math.abs(dot(n, other.axisV));
          return (
            dot(
              n,
              other.center.map((v, i) => v - c[i]),
            ) +
              extent <=
            1e-10
          );
        });
      if (empty) {
        charts.push(
          (map[slot] = {
            object,
            slot,
            cols: 1,
            rows: 1,
            ids: [-1, -1, -1, -1],
            knots: [
              [0, 1],
              [0, 1],
            ],
          }),
        );
        return;
      }
      const knots = [cols, rows].map((count) => Array.from({ length: count + 1 }, (_, i) => i / count));
      if (!s.axes) {
        const ranges = [s.axisU, s.axisV],
          dual = shapeRecord(s);
        const features = [];
        for (const r of scene.rectangles) {
          const gap = dot(
            n,
            r.center.map((x, j) => x - s.center[j]),
          );
          if (
            r === s ||
            gap <= 1e-6 ||
            gap > Math.min(0.35 * Math.min(width, height), 2 * Math.min(norm(r.axisU), norm(r.axisV))) ||
            Math.abs(dot(unit(cross(r.axisU, r.axisV)), n)) < 0.99999
          )
            continue;
          const bounds = ranges.map((_, d) => {
            const axis = dual.slice(3 + d * 3, 6 + d * 3),
              q =
                0.5 +
                0.5 *
                  dot(
                    r.center.map((x, i) => x - s.center[i]),
                    axis,
                  );
            const h = 0.5 * (Math.abs(dot(r.axisU, axis)) + Math.abs(dot(r.axisV, axis)));
            return [Math.max(0, q - h), Math.min(1, q + h)];
          });
          if (bounds.every(([a, b]) => b > a + 1e-7)) features.push({ gap, bounds });
        }
        if (features.length)
          for (let d = 0; d < 2; d++) {
            const count = Math.min(12, Math.ceil(norm(ranges[d]) / 0.42));
            knots[d] = Array.from({ length: count + 1 }, (_, i) => i / count);
          }
        const covered = [];
        for (const { gap, bounds } of features.sort((a, b) => a.gap - b.gap)) {
          if (covered.some((q) => bounds.every(([l, h], i) => l >= q[i][0] - 1e-7 && h <= q[i][1] + 1e-7)))
            continue;
          covered.push(bounds);
          for (let d = 0; d < 2; d++) {
            const [l, h] = bounds[d],
              w = gap / (2 * norm(ranges[d]));
            const cuts = [l - w, l + w, l + 4 * w, (l + h) / 2, h - 4 * w, h - w, h + w];
            if (w < (h - l) / 12) cuts.push(l, l + 2 * w, l + 8 * w, h - 8 * w, h - 2 * w, h);
            knots[d].push(...cuts.filter((v) => v > 1e-5 && v < 1 - 1e-5));
          }
        }
        for (let d = 0; d < 2; d++)
          knots[d] = knots[d].sort((a, b) => a - b).filter((x, i, a) => !i || x - a[i - 1] > 1e-5);
      }
      cols = knots[0].length - 1;
      rows = knots[1].length - 1;
      const desc = { object, slot, type, c, ax, u, v, n, width, height, r, cols, rows, ids, offset, knots };
      charts.push(desc);
      map[slot] = desc;
      function at(a, b) {
        if (type === 0) return [add(c, scale(ax, (2 * a - 1) * width), scale(v, (2 * b - 1) * height)), n];
        const arc = add(scale(u, Math.cos(b * halfPi)), scale(v, Math.sin(b * halfPi)));
        if (type === 1) return [add(c, scale(ax, (2 * a - 1) * width), scale(arc, r)), arc];
        const normal = add(scale(arc, Math.cos(a * halfPi)), scale(ax, Math.sin(a * halfPi)));
        return [add(c, scale(normal, r)), normal];
      }
      for (let j = 0; j <= rows; j++)
        for (let i = 0; i <= cols; i++) {
          const [p, normal] = at(knots[0][i], knots[1][j]);
          const key = [...p, ...normal].map((x) => Math.round(x * 1e10)).join(',');
          if (!shared.has(key)) {
            // Use the interior limit of a sheet at an intersection of authored walls.
            const pos = s.axes
              ? p
              : add(
                  s.center,
                  scale(
                    p.map((x, k) => x - s.center[k]),
                    0.999995,
                  ),
                );
            shared.set(key, receivers.length);
            receivers.push({ p: pos, n: normal, object });
          }
          ids.push(shared.get(key));
        }
      for (let j = 0; j < rows; j++)
        for (let i = 0; i < cols; i++) {
          if ((!type && (!width || !height)) || (type === 1 && !width)) continue;
          let origin = c;
          if (type === 0)
            origin = add(
              c,
              scale(ax, (knots[0][i] + knots[0][i + 1] - 1) * width),
              scale(v, (knots[1][j] + knots[1][j + 1] - 1) * height),
            );
          if (type === 1) origin = add(c, scale(ax, ((2 * (i + 0.5)) / cols - 1) * width));
          const ix = j * (cols + 1) + i;
          tiles.push({
            ...desc,
            c: origin,
            width: width * (knots[0][i + 1] - knots[0][i]),
            height: height * (knots[1][j + 1] - knots[1][j]),
            nodes: [ids[ix], ids[ix + 1], ids[ix + cols + 1], ids[ix + cols + 2]],
          });
        }
    }
    if (!s.axes) {
      const U = norm(s.axisU),
        V = norm(s.axisV),
        n = unit(cross(s.axisU, s.axisV));
      // Each affine source uses a unit generator. Its transverse derivative
      // contributes the area Jacobian separately for skew authored rectangles.
      for (let side = 0; side < 2; side++)
        chart(
          side,
          0,
          s.center,
          unit(s.axisU),
          [0, 0, 0],
          unit(s.axisV),
          scale(n, side ? -1 : 1),
          U,
          V,
          0,
          Math.min(12, Math.ceil(U / 0.42)),
          Math.min(12, Math.ceil(V / 0.42)),
        );
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
          1,
          1,
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
          1,
          1,
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
        1,
        1,
      );
    }
  }
  // Each group is a continuous domain on a receiver plane. Adjacent domains
  // share boundary nodes. Curved receivers continue to use their actual normals.
  const groups = [];
  for (const g of charts) {
    if (g.type !== 0) continue;
    for (let j = 0; j < g.rows; j += 4)
      for (let i = 0; i < g.cols; i += 4) {
        const lastI = Math.min(i + 4, g.cols),
          lastJ = Math.min(j + 4, g.rows),
          [u, v] = g.knots;
        const center = add(
          g.c,
          scale(g.ax, g.width * (u[i] + u[lastI] - 1)),
          scale(g.v, g.height * (v[j] + v[lastJ] - 1)),
        );
        const shape = shapes[g.object],
          factor = shape.axes ? 1 : 0.999995;
        const c = add(
          shape.center,
          scale(
            center.map((x, d) => x - shape.center[d]),
            factor,
          ),
        );
        const a = scale(g.ax, g.width * (u[lastI] - u[i]) * factor),
          b = scale(g.v, g.height * (v[lastJ] - v[j]) * factor);
        const id = groups.length + 1;
        groups.push({ c, a, b, n: g.n, object: g.object });
        for (let y = j; y <= lastJ; y++)
          for (let x = i; x <= lastI; x++) {
            const r = receivers[g.ids[y * (g.cols + 1) + x]];
            if (!r.group) r.group = id;
          }
      }
  }
  const domains = new ArrayBuffer(104 * groups.length);
  groups.forEach((g, i) => {
    new Float64Array(domains, i * 104, 12).set([...g.c, ...g.a, ...g.b, ...g.n]);
    new Int32Array(domains, i * 104 + 96, 2)[0] = g.object;
  });
  const tileBuffer = new ArrayBuffer(168 * tiles.length);
  tiles.forEach((chart, k) => writeChart(tileBuffer, k * 168, chart));
  const points = new ArrayBuffer(56 * receivers.length);
  receivers.forEach((receiver, k) => writeReceiver(points, k * 56, receiver));
  return {
    shapes,
    receivers,
    tiles,
    charts,
    maps,
    groups,
    packet: {
      domains,
      jacobians: Float64Array.from(tiles, (t) => (t.type === 0 ? norm(cross(t.ax, t.v)) : 1)),
      definitions: shapes.map((s) => s.definition ?? null),
      shapes: new Float64Array(shapes.flatMap(shapeRecord)),
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
  new Float64Array(buffer, offset, 18).set([...c.c, ...c.ax, ...c.u, ...c.v, ...c.n, c.width, c.height, c.r]);
  new Int32Array(buffer, offset + 144, 6).set([c.type, c.object, c.slot, 0, 0, 0]);
}
export function writeReceiver(buffer, offset, { p, n, object, group = 0 }) {
  new Float64Array(buffer, offset, 6).set([...p, ...n]);
  new Int32Array(buffer, offset + 48, 2).set([object, group]);
}
