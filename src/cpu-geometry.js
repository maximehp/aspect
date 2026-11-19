import { shadowVolume, projectedFootprint, partitionReceiver } from './visibility.js';
export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
export const unit = (a) => {
  const l = Math.sqrt(dot(a, a));
  return a.map((x) => x / l);
};
export const point = (r, u, v) => r.center.map((x, k) => x + r.axisU[k] * u + r.axisV[k] * v);
const scale = (v, a) => v.map((x) => x * a);
// Tangent coordinates for six signed-normal radiance charts. The physical
// emission sites, normals, areas and visibility come from the filleted surface.
export function boxCharts(b) {
  const result = [];
  for (let i = 0; i < 3; i++)
    for (const sign of [1, -1]) {
      const a = (i + 1) % 3,
        c = (i + 2) % 3;
      result.push({
        id: b.id + '/' + i + '/' + sign,
        center: b.center.map((x, k) => x + sign * b.axes[i][k] * b.halfSize[i]),
        axisU: scale(b.axes[a], sign * b.halfSize[a]),
        axisV: scale(b.axes[c], b.halfSize[c]),
        color: b.color,
        owner: b,
      });
    }
  return result;
}
// The support of a rounded box is the maximum over the eight spherical
// corners. Support planes bound the perspective silhouette in angular space.
function roundedProjection(shape, eye) {
  const rel = sub(shape.center, eye),
    core = shape.halfSize.map((h) => h - shape.radius);
  const closest = rel.slice();
  for (let i = 0; i < 3; i++) {
    const u = Math.max(-core[i], Math.min(core[i], -dot(rel, shape.axes[i])));
    for (let k = 0; k < 3; k++) closest[k] += u * shape.axes[i][k];
  }
  const distance = Math.sqrt(dot(closest, closest));
  if (distance <= shape.radius + 1e-8) return null;
  const z = unit(closest),
    x = unit(cross(z, Math.abs(z[0]) < 0.8 ? [1, 0, 0] : [0, 1, 0])),
    y = cross(z, x);
  return { rel, core, distance, z, x, y };
}

// Four of the full silhouette's uniformly spaced tangent directions are ±x
// and ±y in this frame. Bounds on transverse position and positive depth
// enclose all four slopes. Rejecting a receiver outside this wider cone
// avoids constructing the remaining support planes at that integration site.
export function shadowBounds(shape, eye) {
  const basis = roundedProjection(shape, eye);
  if (!basis) return null;
  const { rel, core, distance, x, y, z } = basis;
  const extent = (axis) =>
    shape.radius + core.reduce((a, h, i) => a + h * Math.abs(dot(axis, shape.axes[i])), 0);
  const near = distance - shape.radius,
    far = dot(z, rel) + extent(z);
  const planes = [z];
  for (const axis of [x, y]) {
    const c = dot(rel, axis),
      r = extent(axis),
      lo = c - r,
      hi = c + r;
    const left = lo / (lo < 0 ? near : far),
      right = hi / (hi > 0 ? near : far);
    planes.push(
      axis.map((v, k) => v - left * z[k]),
      axis.map((v, k) => right * z[k] - v),
    );
  }
  const tolerance = planes.map((n) => 1e-9 * (1 + Math.hypot(...n)));
  return { basis, planes, near, eye, tolerance };
}
export function canShadow(rect, bounds) {
  if (!bounds) return false;
  const delta = sub(rect.center, bounds.eye);
  for (let i = 0; i < bounds.planes.length; i++) {
    const n = bounds.planes[i],
      a = dot(n, delta),
      b = Math.abs(dot(n, rect.axisU)) + Math.abs(dot(n, rect.axisV)),
      h = i ? 0 : bounds.near;
    if (a + b < h - (bounds.tolerance[i] + 1e-9 * (Math.abs(a) + b + h))) return false;
  }
  return true;
}
// The angular stencil and a solid's corner offsets are independent of the
// eye. Only their perspective tangents change when another receiver uses it.
export const shadowAzimuthCount = 24;
export const shadowAzimuths = Array.from({ length: shadowAzimuthCount }, (_, i) => {
  const a = (2 * Math.PI * i) / shadowAzimuthCount;
  return [Math.cos(a), Math.sin(a)];
});
const cornerOffsets = new WeakMap();
function coreOffsets(shape, core) {
  let offsets = cornerOffsets.get(shape);
  if (!offsets) {
    offsets = Array.from({ length: 8 }, (_, i) =>
      core.map((_, k) => shape.axes.reduce((s, ax, j) => s + ((i >> j) & 1 ? 1 : -1) * ax[k] * core[j], 0)),
    );
    cornerOffsets.set(shape, offsets);
  }
  return offsets;
}
export function volume(shape, eye, projection) {
  if (!shape.axes) return shadowVolume(shape, eye);
  const basis = projection || roundedProjection(shape, eye);
  if (!basis) return null;
  const { rel, core, distance, z, x, y } = basis;
  const planes = [{ normal: z, origin: eye.map((v, k) => v + (distance - shape.radius) * z[k]) }];
  const corners = coreOffsets(shape, core).map((o) => [rel[0] + o[0], rel[1] + o[1], rel[2] + o[2]]);
  const depth = corners.map((p) => dot(z, p));
  const radius = shape.radius,
    r2 = radius * radius;
  const denominator = depth.map((d) => d * d - r2);
  const dz = shape.axes.map((axis) => dot(axis, z));
  function addPlane(ca, sa) {
    const v = [x[0] * ca + y[0] * sa, x[1] * ca + y[1] * sa, x[2] * ca + y[2] * sa];
    const vx = dot(shape.axes[0], v),
      vy = dot(shape.axes[1], v),
      vz = dot(shape.axes[2], v);
    const cornerAt = (h) => +(vx > h * dz[0]) | (+(vy > h * dz[1]) << 1) | (+(vz > h * dz[2]) << 2);
    const tangent = (i) => {
      const X = dot(v, corners[i]),
        Z = depth[i];
      return (X * Z + radius * Math.sqrt(Math.max(0, X * X + Z * Z - r2))) / denominator[i];
    };
    // At slope h all eight spherical corners have the same support radius.
    // Maximizing X − hZ on the box selects the corner by three signs. Its
    // tangent increases h until that same corner supports the whole solid.
    let i = cornerAt(0),
      h = tangent(i);
    for (let step = 0; ; step++) {
      const next = cornerAt(h);
      if (next === i) break;
      const candidate = tangent(next);
      if (candidate <= h || step === 7) {
        // At floating-point ties use the full definition.
        for (let k = 0; k < 8; k++) h = Math.max(h, tangent(k));
        break;
      }
      i = next;
      h = candidate;
    }
    planes.push({ normal: z.map((q, k) => q * h - v[k]), origin: eye });
  }
  for (const [c, s] of shadowAzimuths) addPlane(c, s);
  for (let i = 0; i < 8; i++)
    for (let j = 0; j < 3; j++)
      if (!(i & (1 << j))) {
        const n = cross(corners[i], corners[i | (1 << j)]);
        const nx = dot(n, x),
          ny = dot(n, y);
        const length = Math.sqrt(nx * nx + ny * ny);
        const c = length ? nx / length : 1,
          s = length ? ny / length : 0;
        addPlane(c, s);
        addPlane(-c, -s);
      }
  return planes;
}
export function regions(rect, blockers, initial, previous) {
  if (initial && !initial.some((c) => c.lit)) return initial;
  const footprints = [];
  const pairs = previous?.pairs;
  for (const [shape, planes] of blockers) {
    if (!planes || shape === rect || shape === rect.owner) continue;
    let poly = pairs?.get(planes);
    if (poly === undefined) {
      poly = projectedFootprint(rect, planes);
      pairs?.set(planes, poly);
    }
    if (poly) footprints.push(poly);
  }
  if (initial && !footprints.length) return initial;
  // The original operand order matters at coincident boundaries. A receiver
  // reuses its partition only when the ordered nonempty footprints agree.
  if (
    previous?.cells &&
    previous.initial === initial &&
    previous.footprints.length === footprints.length &&
    footprints.every((poly, i) => poly === previous.footprints[i])
  )
    return previous.cells;
  const cells = partitionReceiver(footprints, initial);
  if (previous) Object.assign(previous, { initial, footprints, cells });
  return cells;
}
