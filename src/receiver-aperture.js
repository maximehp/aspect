// Restrict the shutter cast by a finite receiver region onto an intermediate
// plane. Its corners suffice to prove full obstruction by one planar sheet.
// With ordered depths, their extremal projections also bound partial shadows.
import { dot, cross, sub, unit } from './cpu-geometry.js';
import { square, split, shadowVolume, projectedFootprint } from './visibility.js';
import { sheetVertices, boxVertices } from './lamp-visibility.js';

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
function intersect(p, q) {
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
    const wallCorners = sheetVertices(W),
      receiving = sheetVertices(region),
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
    for (const s of blockers) {
      const vertices = s.axes ? boxVertices(s) : sheetVertices(s),
        B = vertices.map((x) => sign * z(x));
      if (!s.axes) {
        const n = cross(s.axisU, s.axisV),
          hs = wallCorners.concat(receiving).map((x) => dot(sub(x, s.center), n));
        // Touching only the boundary of the bundle has zero area.
        if (hs.every((h) => h >= 0) || hs.every((h) => h <= 0)) continue;
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
