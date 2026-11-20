// On a plane, the boundary of a finite sheet's point-light shadow is a line.
// Compile the union once for the authored chart. Between consecutive vertices
// each blocked generator interval has two affine endpoints.
import { projectedFootprint, partitionReceiver, shadowVolume } from './visibility.js';
import { dot, sub } from './cpu-geometry.js';

function bands(chart, planes) {
  const surface = {
    center: chart.c,
    axisU: chart.ax.map((x) => x * chart.width),
    axisV: chart.v.map((x) => x * chart.height),
  };
  const clips = [];
  for (let j = 0; j < planes.length; j++)
    if (j !== chart.object) {
      const uv = projectedFootprint(surface, planes[j]);
      if (uv) clips.push(uv);
    }
  const dark = partitionReceiver(clips).filter((c) => !c.lit);
  const stops = [...new Set([-1, 1, ...dark.flatMap((c) => c.uv.map((p) => p[1]))])].sort((a, b) => a - b);
  const records = [],
    pieces = [];
  for (let j = 1; j < stops.length; j++) {
    const a = stops[j - 1],
      b = stops[j],
      mid = (a + b) / 2;
    if (!(mid > a && mid < b)) return null;
    const intervals = [];
    for (const cell of dark) {
      let left, right;
      for (let i = 0; i < cell.uv.length; i++) {
        const p = cell.uv[i],
          q = cell.uv[(i + 1) % cell.uv.length];
        if (p[1] < mid === q[1] < mid) continue;
        const slope = (q[0] - p[0]) / (q[1] - p[1]);
        const edge = {
          x: p[0] + (mid - p[1]) * slope,
          lo: p[0] + (a - p[1]) * slope,
          hi: p[0] + (b - p[1]) * slope,
        };
        if (!left || edge.x < left.x) left = edge;
        if (!right || edge.x > right.x) right = edge;
      }
      if (left && right && right.x > left.x) intervals.push([left, right]);
    }
    intervals.sort((p, q) => p[0].x - q[0].x);
    const merged = [];
    for (const [l, r] of intervals) {
      const last = merged.at(-1);
      if (last && l.x <= last[1].x) {
        if (r.x > last[1].x) last[1] = r;
      } else merged.push([l, r]);
    }
    records.push(a * chart.height, b * chart.height, pieces.length, merged.length);
    for (const [l, r] of merged)
      pieces.push(l.lo * chart.width, r.lo * chart.width, l.hi * chart.width, r.hi * chart.width);
  }
  return [records.length / 4, ...records, ...pieces];
}

export function compilePlaneShadows(geometry) {
  const { shapes, charts, tiles, packet } = geometry,
    ns = shapes.length,
    nt = tiles.length;
  const planes = shapes.map((s) => (s.axes ? null : shadowVolume(s, packet.lamp))),
    programs = new Map(),
    data = Array(4 * nt).fill(0);
  for (let k = 0; k < nt; k++) {
    const tile = tiles[k],
      ids = k * (ns + 1),
      count = packet.lampIds[ids];
    if (tile.type || count <= 0) continue;
    let sheets = 0;
    while (sheets < count && !shapes[packet.lampIds[ids + 1 + sheets] >> 3].axes) sheets++;
    if (!sheets) continue;
    const chart = charts[tile.offset];
    if (!programs.has(chart)) {
      const program = bands(chart, planes),
        offset = program ? data.length : 0;
      programs.set(chart, offset);
      if (program) for (const value of program) data.push(value);
    }
    const start = programs.get(chart);
    if (!start) continue;
    const d = sub(tile.c, chart.c),
      uv = dot(chart.ax, chart.v),
      det = 1 - uv * uv;
    if (det < 1e-8) continue;
    data[4 * k] = start;
    data[4 * k + 3] = sheets;
    data[4 * k + 1] = (dot(d, chart.ax) - uv * dot(d, chart.v)) / det;
    data[4 * k + 2] = (dot(d, chart.v) - uv * dot(d, chart.ax)) / det;
  }
  return new Float64Array(data);
}
