// These functions belong to the authored plane and the current pose. Refining
// its indirect brightness changes neither the lamp nor the boundary equations.
import { projectedFootprint, shadowVolume } from './visibility.js';
import { filletFootprint } from './fillet-footprint.js';
import { dot, sub } from './cpu-geometry.js';

// Intersect the plane with the union of the sheet shadows. Sweep the original
// edges. Keep a break only when an exposed endpoint changes its defining line.
function bands(chart, planes) {
  const surface = {
    center: chart.c,
    axisU: chart.ax.map((x) => x * chart.width),
    axisV: chart.v.map((x) => x * chart.height),
  };
  const stops = new Set([-1, 1]),
    edges = [],
    polygons = [];
  for (let j = 0; j < planes.length; j++) {
    if (j === chart.object) continue;
    const uv = projectedFootprint(surface, planes[j]);
    if (!uv) continue;
    const boundary = [];
    for (let i = 0; i < uv.length; i++) {
      const p = uv[i],
        q = uv[(i + 1) % uv.length];
      if (p[1] === q[1]) continue;
      const slope = (q[0] - p[0]) / (q[1] - p[1]);
      const edge = {
        low: Math.min(p[1], q[1]),
        high: Math.max(p[1], q[1]),
        slope,
        at: (t) => p[0] + (t - p[1]) * slope,
      };
      stops.add(edge.low);
      stops.add(edge.high);
      for (const other of edges) {
        const a = Math.max(edge.low, other.low),
          b = Math.min(edge.high, other.high);
        if (!(b > a)) continue;
        const left = edge.at(a) - other.at(a),
          right = edge.at(b) - other.at(b);
        if ((left < 0 && right > 0) || (left > 0 && right < 0))
          stops.add(a + ((b - a) * left) / (left - right));
      }
      boundary.push(edge);
    }
    edges.push(...boundary);
    polygons.push(boundary);
  }
  const cuts = [...stops].sort((a, b) => a - b),
    spans = [];
  for (let j = 1; j < cuts.length; j++) {
    const a = cuts[j - 1],
      b = cuts[j],
      mid = (a + b) / 2;
    if (!(mid > a && mid < b)) return null;
    const intervals = [];
    for (const boundary of polygons) {
      const active = boundary
        .filter((e) => e.low < mid && mid < e.high)
        .sort((l, r) => l.at(mid) - r.at(mid));
      if (active.length >= 2) intervals.push([active[0], active.at(-1)]);
    }
    intervals.sort((p, q) => p[0].at(mid) - q[0].at(mid));
    const merged = [];
    for (const [l, r] of intervals) {
      const prev = merged.at(-1);
      if (prev && l.at(mid) <= prev[1].at(mid)) {
        if (r.at(mid) > prev[1].at(mid)) prev[1] = r;
      } else merged.push([l, r]);
    }
    const last = spans.at(-1);
    if (
      last &&
      merged.length === last.edges.length &&
      merged.every(([l, r], i) => l === last.edges[i][0] && r === last.edges[i][1])
    )
      last.b = b;
    else spans.push({ a, b, edges: merged });
  }
  const records = [],
    pieces = [];
  for (const { a, b, edges } of spans) {
    records.push(a * chart.height, b * chart.height, pieces.length, edges.length);
    for (const [l, r] of edges)
      pieces.push(...[l.at(a), r.at(a), l.at(b), r.at(b)].map((x) => x * chart.width));
  }
  return [spans.length, ...records, ...pieces];
}

export function poseShadows(surface, lamp) {
  const planes = surface.shapes.map((s) => (s.axes ? null : shadowVolume(s, lamp))),
    cache = new Map();
  return (chart) => {
    if (!cache.has(chart)) {
      const outlines = surface.shapes.map((s, i) =>
        s.axes && i !== chart.object ? filletFootprint(chart, s, lamp) : null,
      );
      cache.set(chart, { sheets: bands(chart, planes), outlines });
    }
    return cache.get(chart);
  };
}
export function compilePlaneShadows(geometry, shadowOn) {
  const { charts, tiles, packet } = geometry,
    offsets = new Map(),
    data = Array(4 * tiles.length).fill(0);
  for (let k = 0; k < tiles.length; k++) {
    const tile = tiles[k];
    if (tile.type) continue;
    const chart = charts[tile.offset];
    if (dot(tile.n, sub(packet.lamp, tile.c)) > 0 && !offsets.has(chart)) {
      const { sheets, outlines } = shadowOn(chart);
      const start = sheets ? data.length : 0;
      if (sheets) for (const value of sheets) data.push(value);
      const table = data.length;
      data.push(...Array(outlines.length).fill(0));
      outlines.forEach((f, i) => {
        if (!f) return;
        data[table + i] = data.length;
        data.push(...f);
      });
      offsets.set(chart, [start, table]);
    }
    const [start, table] = offsets.get(chart) || [0, 0],
      [a, b, l, h] = tile.bounds;
    data[4 * k] = start;
    data[4 * k + 1] = (a + b - 1) * chart.width;
    data[4 * k + 2] = (l + h - 1) * chart.height;

    data[4 * k + 3] = table;
  }
  return new Float64Array(data);
}
