import { joinSheets } from './join-sheets.js';
import { roundedBoxExtent } from './rounded-box.js';
import { shadowVolume } from './visibility.js';
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
const scale = (v, a) => v.map((x) => x * a);
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

// The angular stencil and a solid's corner offsets are independent of the
// eye. Only their perspective tangents change when another receiver uses it.
export const shadowAzimuthCount = 24;
export const shadowAzimuths = Array.from({ length: shadowAzimuthCount }, (_, i) => {
  const a = (2 * Math.PI * i) / shadowAzimuthCount;
  return [Math.cos(a), Math.sin(a)];
});
function coreOffsets(shape,core){
  return Array.from({length:8},(_,i)=>core.map((_,k)=>
    shape.axes.reduce((s,ax,j)=>s+((i>>j)&1?1:-1)*ax[k]*core[j],0)));
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

function shapeRecord(s) {
  if (s.axes)
    return [
      ...s.center,
      ...s.axes.flat(),
      ...s.halfSize,
      s.radius,
      ...roundedBoxExtent(s),
    ];
  const normal = unit(cross(s.axisU, s.axisV));
  const u = cross(s.axisV, normal),
    v = cross(normal, s.axisU);
  return [
    ...s.center,
    ...scale(u, 1 / dot(u, s.axisU)),
    ...scale(v, 1 / dot(v, s.axisV)),
    ...normal,
    0,
    0,
    0,
    0,
    ...s.axisU.map((v, i) => Math.abs(v) + Math.abs(s.axisV[i])),
  ];
}

// Capture one pose at submission. Visibility, source integration and display
// all read the same ordered shape equations, including mutable author edits.
export function sceneGeometry(scene) {
  const { rectangles, roundedBoxes, lightPosition, lampRGB } = structuredClone({
    rectangles: scene.rectangles,
    roundedBoxes: scene.roundedBoxes,
    lightPosition: scene.lightPosition,
    lampRGB: scene.lampRGB,
  });
  const joined = joinSheets(rectangles),
    shapes = [...joined.map((x) => x.shape), ...roundedBoxes];
  const authoredToSurface = new Uint32Array(rectangles.length + roundedBoxes.length);
  joined.forEach((e, i) => {
    for (const id of e.ids) authoredToSurface[id] = i;
  });
  roundedBoxes.forEach((s, i) => (authoredToSurface[i + rectangles.length] = i + joined.length));
  return {
    shapes,
    lightPosition,
    lampRGB,
    authoredToSurface,
    records: Float64Array.from(shapes.flatMap(shapeRecord)),
  };
}
