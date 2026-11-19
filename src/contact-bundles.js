// A convex hull encloses every segment connecting these receiver sites to
// their visible source pieces. A separating plane can reject a moving solid
// for the whole group, before constructing any eye-dependent silhouettes.
import { cross, dot, point, shadowAzimuthCount, sub } from './cpu-geometry.js';

const groupsBySites = new WeakMap();
const leafSize = 12;

function hullPlanes(points) {
  const planes = [];
  // Work in local coordinates so large translations don't affect the hull.
  const origin = points[0];
  const vertices = points.map((p) => sub(p, origin));
  for (let a = 0; a < vertices.length; a++)
    for (let b = a + 1; b < vertices.length; b++) {
      const edge = sub(vertices[b], vertices[a]);
      for (let c = b + 1; c < vertices.length; c++) {
        const n = cross(edge, sub(vertices[c], vertices[a]));
        const length = Math.sqrt(dot(n, n));
        if (length < 1e-12) continue;
        let x = n[0] / length,
          y = n[1] / length,
          z = n[2] / length;
        const at = dot(vertices[a], [x, y, z]);
        let lo = Infinity,
          hi = -Infinity;
        for (const p of vertices) {
          const t = p[0] * x + p[1] * y + p[2] * z;
          lo = Math.min(lo, t);
          hi = Math.max(hi, t);
        }
        if (at > lo + 1e-9 && at < hi - 1e-9) continue;
        if (at < hi - 1e-9) {
          x = -x;
          y = -y;
          z = -z;
          hi = -lo;
        }
        if (!planes.some((p) => p[0] * x + p[1] * y + p[2] * z > 1 - 1e-10)) planes.push([x, y, z, hi]);
      }
    }
  return { origin, planes: Float64Array.from(planes.flat()) };
}

function boundsOfSites(records) {
  const lo = [Infinity, Infinity, Infinity],
    hi = [-Infinity, -Infinity, -Infinity];
  for (const { p } of records)
    for (let k = 0; k < 3; k++) {
      lo[k] = Math.min(lo[k], p[k]);
      hi[k] = Math.max(hi[k], p[k]);
    }
  return { lo, hi };
}

function divide(records, groups) {
  const { lo, hi } = boundsOfSites(records);
  if (records.length > leafSize) {
    let axis = 0;
    for (let k = 1; k < 3; k++) if (hi[k] - lo[k] > hi[axis] - lo[axis]) axis = k;
    records.sort((a, b) => a.p[axis] - b.p[axis]);
    const mid = records.length >> 1;
    divide(records.slice(0, mid), groups);
    divide(records.slice(mid), groups);
    return;
  }
  const points = [];
  for (const x of new Set([lo[0], hi[0]]))
    for (const y of new Set([lo[1], hi[1]])) for (const z of new Set([lo[2], hi[2]])) points.push([x, y, z]);
  let left = 1,
    right = -1,
    bottom = 1,
    top = -1;
  for (const { sample } of records)
    for (const { uv } of sample.lit)
      for (const [u, v] of uv) {
        left = Math.min(left, u);
        right = Math.max(right, u);
        bottom = Math.min(bottom, v);
        top = Math.max(top, v);
      }
  const mode = records[0].sample.m;
  for (const u of [left, right]) for (const v of [bottom, top]) points.push(point(mode, u, v));
  const group = hullPlanes(points);
  groups.push(group);
  group.candidates = [];
  for (const { sample } of records) sample.bundle = group;
}

// The first plane of a rounded shadow is at the nearest core depth minus
// the fillet radius. Every ray accepted by the renderer must cross it.
// Project the physical body back onto that plane: transverse coordinates
// contract toward the eye axis, which passes through the core. Both ends
// lie within R of the transversely projected solid center. The 24 sampled
// tangents inflate that disk by at most sec(pi/24). The axial offset from
// the center is at most R too. This ball therefore encloses the entire cap,
// including points admitted by the renderer's circumscribed silhouette.
const capFactor = Math.hypot(1, 1 / Math.cos(Math.PI / shadowAzimuthCount));
function capRadius(shape) {
  const r = shape.radius,
    h = shape.halfSize;
  return capFactor * (r + Math.hypot(h[0] - r, h[1] - r, h[2] - r));
}
function mayCross(group, center, radius) {
  const planes = group.planes,
    d = sub(center, group.origin);
  const margin =
    1e-8 *
    (1 +
      Math.abs(center[0]) +
      Math.abs(center[1]) +
      Math.abs(center[2]) +
      radius +
      Math.abs(d[0]) +
      Math.abs(d[1]) +
      Math.abs(d[2]));
  for (let i = 0; i < planes.length; i += 4) {
    if (planes[i] * d[0] + planes[i + 1] * d[1] + planes[i + 2] * d[2] > planes[i + 3] + radius + margin)
      return false;
  }
  return true;
}
export function prepareBundles(sites, moving) {
  if (!sites.length) return;
  let groups = groupsBySites.get(sites);
  if (!groups) {
    const bySource = new Map();
    for (const site of sites)
      for (const sample of site.samples) {
        if (!bySource.has(sample.m)) bySource.set(sample.m, []);
        bySource.get(sample.m).push({ p: site.p, sample });
      }
    groups = [];
    for (const records of bySource.values()) divide(records, groups);
    groupsBySites.set(sites, groups);
  }
  const radii = moving.map(capRadius);
  for (const group of groups) {
    group.candidates.length = 0;
    for (let i = 0; i < moving.length; i++)
      if (mayCross(group, moving[i].center, radii[i])) group.candidates.push(i);
  }
}
