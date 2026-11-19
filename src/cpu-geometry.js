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
function roundedPoint(b, signs, dirs) {
  const n = unit(dirs),
    p = b.center.slice(),
    normal = [0, 0, 0];
  for (let i = 0; i < 3; i++)
    for (let k = 0; k < 3; k++) {
      p[k] += b.axes[i][k] * (signs[i] * (b.halfSize[i] - b.radius) + b.radius * n[i]);
      normal[k] += b.axes[i][k] * n[i];
    }
  return { p, normal };
}
// Flat faces, circular edge strips and spherical corner octants. All vertices
// lie on the analytic fillet. Chord tolerance controls only its drawing outline.
export function shell(b) {
  const faces = [],
    steps = 2;
  for (let d = 0; d < 3; d++)
    for (const side of [-1, 1]) {
      const a = (d + 1) % 3,
        c = (d + 2) % 3;
      const samples = (axis) => {
        const values = [];
        for (let j = steps; j >= 0; j--) values.push([-1, -j / steps]);
        const spans = Math.ceil((2 * (b.halfSize[axis] - b.radius)) / 0.7);
        for (let j = 1; j < spans; j++) values.push([(2 * j) / spans - 1, 0]);
        for (let j = 0; j <= steps; j++) values.push([1, j / steps]);
        return values;
      };
      const us = samples(a),
        vs = samples(c);
      const vertices = us.map(([sa, u]) =>
        vs.map(([sc, v]) => {
          const signs = [],
            dirs = [];
          signs[d] = side;
          signs[a] = sa;
          signs[c] = sc;
          dirs[d] = side;
          dirs[a] = u;
          dirs[c] = v;
          return roundedPoint(b, signs, dirs);
        }),
      );
      for (let i = 0; i + 1 < us.length; i++)
        for (let j = 0; j + 1 < vs.length; j++) {
          const verts = [vertices[i][j], vertices[i + 1][j], vertices[i + 1][j + 1], vertices[i][j + 1]];
          if (side < 0) verts.reverse();
          faces.push(verts);
        }
    }
  return faces;
}
export function broadFaces(b) {
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
export function volume(shape, eye, projection) {
  if (!shape.axes) return shadowVolume(shape, eye);
  const basis = projection || roundedProjection(shape, eye);
  if (!basis) return null;
  const { rel, core, distance, z, x, y } = basis;
  const planes = [{ normal: z, origin: eye.map((v, k) => v + (distance - shape.radius) * z[k]) }];
  const corners = Array.from({ length: 8 }, (_, i) =>
    rel.map((v, k) => v + shape.axes.reduce((s, ax, j) => s + ((i >> j) & 1 ? 1 : -1) * ax[k] * core[j], 0)),
  );
  // Include the corner-to-corner tangent directions of the core outline.
  const angles = Array.from({ length: 24 }, (_, i) => (2 * Math.PI * i) / 24);
  for (let i = 0; i < 8; i++)
    for (let j = 0; j < 3; j++)
      if (!(i & (1 << j))) {
        const n = cross(corners[i], corners[i | (1 << j)]);
        const a = Math.atan2(dot(n, y), dot(n, x));
        angles.push(a, a + Math.PI);
      }
  for (const a of angles) {
    const v = x.map((q, k) => q * Math.cos(a) + y[k] * Math.sin(a));
    let h = -Infinity;
    for (const p of corners) {
      const X = dot(v, p),
        Z = dot(z, p),
        r = shape.radius;
      h = Math.max(h, (X * Z + r * Math.sqrt(Math.max(0, X * X + Z * Z - r * r))) / (Z * Z - r * r));
    }
    planes.push({ normal: z.map((q, k) => q * h - v[k]), origin: eye });
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
