// A sheet entering the emitting hemisphere creates a finite contact line.
// Measure each side separately wherever that line bounds the local radiance.
import { exact, addQ, subQ, mulQ, approximate } from './authored-scalar.js';
import { dot, sub } from './cpu-geometry.js';
import { leaves, split } from './field-domain.js';
export function contactLines(g, shapes) {
  if (g.empty || g.type) return [];
  const skew = dot(g.ax, g.v),
    det = 1 - skew * skew;
  if (det < 1e-8) return [];
  const contacts = [];
  for (const s of shapes) {
    if (s === g.shape || s.axes) continue;
    const center = sub(s.center, g.c);
    const z = dot(g.n, center),
      a = dot(g.n, s.axisU),
      b = dot(g.n, s.axisV);
    const eps =
      32 *
      Number.EPSILON *
      (1 + Math.hypot(...g.c) + Math.hypot(...s.center) + Math.hypot(...s.axisU) + Math.hypot(...s.axisV));
    if (z + Math.abs(a) + Math.abs(b) < -eps || z - Math.abs(a) - Math.abs(b) > eps) continue;
    const vertices = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ],
      points = [];
    const heights = vertices.map(([u, v]) => {
      const h = z + u * a + v * b;
      if (Math.abs(h) > eps) return h;
      // Preserve the sign of a thin gap when forming the four corner sums.
      return approximate(
        g.n.reduce((sum, n, i) => {
          let x = subQ(exact(s.center[i]), exact(g.c[i]));
          x = addQ(x, exact(u * s.axisU[i]));
          x = addQ(x, exact(v * s.axisV[i]));
          return addQ(sum, mulQ(exact(n), x));
        }, exact(0)),
      );
    });
    if (Math.max(...heights) <= 0 || Math.min(...heights) > 0) continue;
    for (let i = 0; i < 4; i++) {
      const p = vertices[i],
        q = vertices[(i + 1) % 4];
      const l = heights[i],
        h = heights[(i + 1) % 4];
      let t;
      if (l === 0) t = 0;
      else if (l * h < 0) t = l / (l - h);
      else continue;
      const u = p[0] + (q[0] - p[0]) * t,
        v = p[1] + (q[1] - p[1]) * t;
      const d = center.map((x, k) => x + u * s.axisU[k] + v * s.axisV[k]);
      const x = dot(d, g.ax),
        y = dot(d, g.v);
      points.push([
        0.5 + (x - skew * y) / (2 * det * g.width),
        0.5 + (y - skew * x) / (2 * det * g.height),
      ]);
    }
    if (points.length < 2) continue;
    for (let axis = 0; axis < 2; axis++) {
      const xs = points.map((p) => p[axis]),
        ys = points.map((p) => p[1 - axis]);
      const cut = (Math.min(...xs) + Math.max(...xs)) / 2;
      if (Math.max(...xs) - Math.min(...xs) > 2e-13 || cut <= 1e-12 || cut >= 1 - 1e-12) continue;
      const lo = Math.max(0, Math.min(...ys)),
        hi = Math.min(1, Math.max(...ys));
      if (hi - lo > 1e-12) contacts.push({ axis, cut, lo, hi });
    }
  }
  // Collinear overlapping sheets define one physical cut.
  const knots = [[], []];
  const snap = (d, x) => {
    const y = knots[d].find((t) => Math.abs(t - x) < 2e-13);
    if (y !== undefined) return y;
    knots[d].push(x);
    return x;
  };
  for (const c of contacts) {
    c.cut = snap(c.axis, c.cut);
    c.lo = snap(1 - c.axis, c.lo);
    c.hi = snap(1 - c.axis, c.hi);
  }
  contacts.sort((a, b) => a.axis - b.axis || a.cut - b.cut || a.lo - b.lo);
  const result = [];
  for (const c of contacts) {
    const last = result.at(-1);
    if (last && c.cut === last.cut && c.axis === last.axis && c.lo <= last.hi)
      last.hi = Math.max(last.hi, c.hi);
    else result.push(c);
  }
  return result;
}
export function splitContacts(root, contacts) {
  for (const { axis, cut, lo, hi } of contacts) {
    const d = 2 * axis,
      t = 2 - d;
    for (const x of [lo, hi])
      for (const leaf of leaves(root)) {
        const b = leaf.bounds;
        if (cut >= b[d] && cut <= b[d + 1]) split(leaf, 1 - axis, x);
      }
    for (const leaf of leaves(root)) {
      const b = leaf.bounds;
      if (b[t] >= lo && b[t + 1] <= hi) split(leaf, axis, cut);
    }
  }
}
// Directions in cyclic order east, north, west, south. At a corner of an
// enclosure, the three exterior quadrants are connected and share one value.
export function contactSector(g, u, v, leaf) {
  if (!g.contacts?.length) return 0;
  let rays = 0;
  for (const { axis, cut, lo, hi } of g.contacts) {
    const x = axis ? v : u,
      t = axis ? u : v;
    if (x !== cut || t < lo || t > hi) continue;
    if (t < hi) rays |= axis ? 1 : 2;
    if (t > lo) rays |= axis ? 4 : 8;
  }
  if (!(rays & (rays - 1))) return 0;
  const [a, b, l, h] = leaf.bounds,
    east = a + b > 2 * u,
    north = l + h > 2 * v;
  let sector = north ? (east ? 0 : 1) : east ? 3 : 2;
  while (!(rays & (1 << sector))) sector = (sector + 3) % 4;
  return sector + 1;
}
