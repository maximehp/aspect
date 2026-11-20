// Perspective preserves convex hulls inside a positive-depth slab. A filleted
// box is the convex hull of its eight corner balls. Its outline on a plane
// consists of arcs of their ellipses and finite tangent generators joining them.
import { dot, sub } from './cpu-geometry.js';

export function filletFootprint(chart, box, eye) {
  const n = chart.n,
    e = sub(chart.c, eye),
    p = sub(box.center, eye);
  const h = dot(e, n),
    z = dot(p, n),
    r = box.radius,
    axes = box.axes;
  const core = box.halfSize.map((x) => x - r);
  const depthRadius = r + axes.reduce((s, a, i) => s + core[i] * Math.abs(dot(a, n)), 0);
  const margin = 1e-9 * (1 + Math.abs(h) + Math.abs(z) + depthRadius);
  if (Math.sign(h) * z <= depthRadius + margin || Math.sign(h) * (h - z) <= depthRadius + margin)
    return null;
  const skew = dot(chart.ax, chart.v),
    det = 1 - skew * skew;
  if (det < 1e-8) return null;
  const U = chart.ax.map((x, i) => (x - skew * chart.v[i]) / det);
  const V = chart.v.map((x, i) => (x - skew * chart.ax[i]) / det);
  const du = dot(e, U),
    dv = dot(e, V);
  const curves = [];
  const shift = (p, axis, d) => p.map((x, i) => x + axis[i] * d);
  const project = (v) => [(h * dot(v, U)) / dot(v, n) - du, (h * dot(v, V)) / dot(v, n) - dv];
  let low = Infinity,
    high = -Infinity;
  for (let mask = 0; mask < 8; mask++) {
    if (core.some((x, i) => !x && mask & (1 << i))) continue;
    const q = axes.reduce((v, axis, i) => shift(v, axis, (mask & (1 << i) ? 1 : -1) * core[i]), p);
    const Z = dot(n, q),
      u = dot(U, q),
      v = dot(V, q),
      den = Z * Z - r * r;
    const scale = (h * h * r * r) / den;
    const xx = scale * (dot(U, U) + (u * u) / den),
      xy = scale * (dot(U, V) + (u * v) / den),
      yy = scale * (dot(V, V) + (v * v) / den);
    const x = (h * Z * u) / den - du,
      y = (h * Z * v) / den - dv,
      dx2 = xx - xy * (xy / yy);
    if (!(dx2 > 1e-12 * xx && yy > 0)) return null;
    const dy = Math.sqrt(yy);
    for (const sign of [-1, 1])
      curves.push({ lo: y - dy, hi: y + dy, fn: [x, y, xy / yy, sign * Math.sqrt(dx2), dy] });
    low = Math.min(low, y - dy);
    high = Math.max(high, y + dy);
  }
  const cuts = [low, high];
  for (let d = 0; d < 3; d++) {
    if (!core[d]) continue;
    const i = (d + 1) % 3,
      j = (d + 2) % 3;
    for (const si of [-1, 1])
      for (const sj of [-1, 1]) {
        const a = -dot(p, axes[i]) - si * core[i],
          b = -dot(p, axes[j]) - sj * core[j];
        const D = a * a + b * b,
          disc = D - r * r;
        if (Math.abs(disc) <= 1e-13 * (D + r * r)) return null;
        if (disc < 0) continue;
        for (const sign of [-1, 1]) {
          const nx = (r * a - sign * Math.sqrt(disc) * b) / D,
            ny = (r * b + sign * Math.sqrt(disc) * a) / D;
          if (nx * si < -1e-13 || ny * sj < -1e-13) continue;
          const mid = shift(shift(p, axes[i], si * core[i] + r * nx), axes[j], sj * core[j] + r * ny);
          const A = project(shift(mid, axes[d], -core[d])),
            B = project(shift(mid, axes[d], core[d]));
          cuts.push(A[1], B[1]);
          if (B[1] !== A[1])
            curves.push({
              lo: Math.min(A[1], B[1]),
              hi: Math.max(A[1], B[1]),
              fn: [A[0], A[1], (B[0] - A[0]) / (B[1] - A[1]), 0, 1],
            });
        }
      }
  }
  // A support line can leave a spherical arc only at a tangent generator
  // endpoint. Between these events the same two features bound the section.
  const stops = [...new Set(cuts)].sort((a, b) => a - b),
    bands = [];
  for (let i = 1; i < stops.length; i++) {
    const lo = stops[i - 1],
      hi = stops[i],
      t = (lo + hi) / 2;
    let L,
      R,
      xl = Infinity,
      xr = -Infinity;
    for (const c of curves) {
      if (t < c.lo || t > c.hi) continue;
      const [x, y, s, dx, dy] = c.fn,
        d = t - y;
      const value = x + s * d + dx * Math.sqrt(Math.max(0, (1 - d / dy) * (1 + d / dy)));
      if (value < xl) {
        xl = value;
        L = c;
      }
      if (value > xr) {
        xr = value;
        R = c;
      }
    }
    if (!L || !R) return null;
    const prior = bands.at(-1);
    if (prior && prior.L === L && prior.R === R) prior.hi = hi;
    else bands.push({ lo, hi, L, R });
  }
  if (!bands.length) return null;
  const active = [...new Set(bands.flatMap((b) => [b.L, b.R]))];
  return [
    bands.length,
    ...bands.flatMap((b) => [b.lo, b.hi, active.indexOf(b.L), active.indexOf(b.R)]),
    ...active.flatMap((c) => c.fn),
  ];
}
