// Restrict the shutter cast by a finite receiver region onto an intermediate
// plane. Its corners suffice to prove full obstruction by one planar sheet.
// With ordered depths, their extremal projections also bound partial shadows.
import { exact, addQ, subQ, mulQ } from './authored-scalar.js';
import { dot, cross, sub, unit } from './cpu-geometry.js';
import { square, split, shadowVolume, projectedFootprint } from './visibility.js';
import { sheetVertices, boxVertices } from './lamp-visibility.js';

export function extent(shape, axis) {
  if (!shape.axes) return Math.abs(dot(shape.axisU, axis)) + Math.abs(dot(shape.axisV, axis));
  return (
    shape.radius * Math.hypot(...axis) +
    shape.axes.reduce((s, a, i) => s + Math.abs(dot(a, axis)) * (shape.halfSize[i] - shape.radius), 0)
  );
}

// Return the closed half-spaces containing an entire shape. Endpoints on
// an oblique plane use the same scalar triple product as axis-aligned ones.
function sidesOf(plane) {
  const n = cross(plane.axisU, plane.axisV),
    scale = Math.hypot(...plane.axisU) * Math.hypot(...plane.axisV);
  let exactNormal;
  return (s) => {
    const mid = dot(n, sub(s.center, plane.center)),
      r = extent(s, n);
    const size = s.axes ? Math.hypot(...s.halfSize) : Math.hypot(...s.axisU) + Math.hypot(...s.axisV);
    const guard = 1e-12 * scale * (1 + size + Math.hypot(...plane.center) + Math.hypot(...s.center));
    if (mid - r > guard) return 1;
    if (mid + r < -guard) return 2;
    if ((mid - r < -guard && mid + r > guard) || s.axes) return 0;
    if (!exactNormal) {
      const u = plane.axisU.map(exact),
        v = plane.axisV.map(exact);
      exactNormal = n.map((_, i) =>
        subQ(mulQ(u[(i + 1) % 3], v[(i + 2) % 3]), mulQ(u[(i + 2) % 3], v[(i + 1) % 3])),
      );
    }
    const project = (v) => v.map((x, i) => mulQ(x, exactNormal[i])).reduce(addQ);
    const center = project(s.center.map((x, i) => subQ(exact(x), exact(plane.center[i]))));
    const abs = (q) => (q.n < 0n ? { n: -q.n, d: q.d } : q);
    const radius = addQ(abs(project(s.axisU.map(exact))), abs(project(s.axisV.map(exact))));
    return Number(subQ(center, radius).n >= 0n) | (Number(addQ(center, radius).n <= 0n) << 1);
  };
}
// A blocked segment goes from a receiver q through some b to a source w.
// If min_B n.b > max_Q n.q, convexity implies n.w >= min_B n.b. Clip the
// complete source chart by this necessary condition, even when the caster
// touches (or crosses) one of the two endpoint planes.
function supportFootprint(w, q, b) {
  const delta = sub(b.center, q.center),
    base = sub(w.center, b.center);
  let p = square;
  const normals = [w.axisU, w.axisV, q.axisU, q.axisV, delta, ...(b.axes || [b.axisU, b.axisV])];
  for (const n of normals) {
    const mid = dot(n, delta),
      rb = extent(b, n),
      rq = extent(q, n);
    const u = dot(n, w.axisU),
      v = dot(n, w.axisV),
      c = dot(n, base);
    const pad = 1e-11 * (1 + Math.abs(mid) + rq + rb + Math.abs(u) + Math.abs(v) + Math.abs(c));
    if (Math.abs(mid) <= rb + rq + pad) continue;
    const sign = Math.sign(mid);
    [p] = split(p, sign * u, sign * v, sign * c + rb + pad, 0, false);
    if (p.length < 3) break;
  }
  return p;
}
function hull(points) {
  const sorted = points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  function side(list) {
    const result = [];
    for (const p of list) {
      while (result.length >= 2) {
        const a = result.at(-2),
          b = result.at(-1);
        if ((b[0] - a[0]) * (p[1] - a[1]) > (p[0] - a[0]) * (b[1] - a[1])) break;
        result.pop();
      }
      result.push(p);
    }
    result.pop();
    return result;
  }
  return side(sorted).concat(side(sorted.slice().reverse()));
}
export function intersect(p, q) {
  for (let k = 0; k < q.length && p.length; k++) {
    const a = q[k],
      b = q[(k + 1) % q.length],
      u = a[1] - b[1],
      v = b[0] - a[0];
    [p] = split(p, u, v, -u * a[0] - v * a[1], 0, false);
  }
  return p;
}
function span(p, t) {
  let lo = Infinity,
    hi = -Infinity;
  for (let i = 0; i < p.length; i++) {
    const a = p[i],
      b = p[(i + 1) % p.length];
    if (a[1] > t === b[1] > t) continue;
    const x = a[0] + ((b[0] - a[0]) * (t - a[1])) / (b[1] - a[1]);
    lo = Math.min(lo, x);
    hi = Math.max(hi, x);
  }
  return [lo, hi];
}

export class ReceiverAperture {
  constructor(chart, region, blockers) {
    this.width = chart.width;
    this.records = [];
    this.cuts = [];
    const W = {
      center: chart.c,
      axisU: chart.ax.map((x) => x * chart.width),
      axisV: chart.v.map((x) => x * chart.height),
    };
    const receiving = sheetVertices(region),
      normal = unit(cross(W.axisU, W.axisV));
    const z = (x) => dot(sub(x, chart.c), normal);
    const sign = Math.sign(receiving.reduce((sum, p) => sum + z(p), 0)) || 1;
    const R = receiving.map((x) => sign * z(x));
    const d = dot(chart.ax, chart.v),
      det = 1 - d * d;
    const local = (p) => {
      const v = sub(p, chart.c),
        x = dot(v, chart.ax),
        t = dot(v, chart.v);
      return [(x - d * t) / (chart.width * det), (t - d * x) / (chart.height * det)];
    };
    const wallSide = sidesOf(W),
      receiverSide = sidesOf(region);
    const opposite = (mask) => ((mask & 1) << 1) | ((mask & 2) >> 1);
    const behindWall = opposite(wallSide(region)),
      behindReceiver = opposite(receiverSide(W));
    for (const s of blockers) {
      if (wallSide(s) & behindWall || receiverSide(s) & behindReceiver) continue;
      const vertices = s.axes ? boxVertices(s) : sheetVertices(s),
        B = vertices.map((x) => sign * z(x));
      if (!s.axes) {
        const side = sidesOf(s);
        if (side(W) & side(region)) continue;
      }
      let possible = square,
        covered = [];
      if (det > 1e-8 && Math.min(...B) > 1e-9 && Math.max(...B) < Math.min(...R) - 1e-9) {
        const points = [];
        for (let i = 0; i < receiving.length; i++)
          for (let j = 0; j < vertices.length; j++) {
            const b = vertices[j],
              r = receiving[i],
              t = B[j] / (R[i] - B[j]);
            points.push(local(b.map((x, k) => x + t * (x - r[k]))));
          }
        possible = intersect(hull(points), square);
        if (possible.length < 3) continue;
      }
      if (possible === square) {
        possible = supportFootprint(W, region, s);
        if (possible.length < 3) continue;
      }
      if (!s.axes) {
        covered = square;
        for (const eye of receiving) {
          const poly = projectedFootprint(W, shadowVolume(s, eye));
          if (!poly) {
            covered = [];
            break;
          }
          covered = intersect(covered, poly);
          if (covered.length < 3) break;
        }
      }
      const toMeters = (p) => p.map(([x, t]) => [x * chart.width, t * chart.height]);
      const outer = toMeters(possible),
        inner = toMeters(covered);
      this.records.push({ s, outer, inner });
      for (const p of [outer, inner]) for (const [x, t] of p) this.cuts.push(t);
    }
  }
  row(t) {
    const spans = this.records.map(({ s, outer, inner }) => ({
      s,
      outer: span(outer, t),
      inner: span(inner, t),
    }));
    const cuts = [-this.width, this.width];
    for (const r of spans) for (const [lo, hi] of [r.outer, r.inner]) if (hi > lo) cuts.push(lo, hi);
    return {
      cuts,
      at(x) {
        if (spans.some((v) => x > v.inner[0] && x < v.inner[1])) return null;
        return spans.filter((v) => x >= v.outer[0] && x <= v.outer[1]).map((v) => v.s);
      },
    };
  }
}
