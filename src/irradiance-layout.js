// Surface locations at which the continuous incident diffuse field is estimated.
// A common planar parameterization gives adjoining pieces identical values at
// their join. Reflectance belongs to the visible material and is applied later.
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const mul = (a, k) => a.map((x) => x * k);
const add = (a, b, k = 1) => a.map((x, i) => x + k * b[i]);
const unit = (a) => mul(a, 1 / Math.hypot(...a));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

function sheetFrame(rect) {
  const area = cross(rect.axisU, rect.axisV),
    jac = Math.hypot(...area),
    n = mul(area, 1 / jac);
  const corners = [
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1],
  ].map(([u, v]) => add(add(rect.center, rect.axisU, u), rect.axisV, v));
  return { rect, n, corners, du: mul(cross(rect.axisV, n), 1 / jac), dv: mul(cross(n, rect.axisU), 1 / jac) };
}

function commonEdge(a, b, epsilon) {
  for (let i = 0; i < 4; i++)
    for (let j = 0; j < 4; j++) {
      const p = a.corners[i],
        q = b.corners[j],
        e = add(a.corners[(i + 1) % 4], p, -1),
        f = add(b.corners[(j + 1) % 4], q, -1);
      const length = Math.hypot(...e),
        direction = mul(e, 1 / length),
        offset = add(q, p, -1);
      if (
        Math.hypot(...cross(direction, unit(f))) > 1e-10 ||
        Math.hypot(...cross(offset, direction)) > epsilon
      )
        continue;
      const inward = cross(a.n, direction);
      if (dot(add(a.rect.center, p, -1), inward) * dot(add(b.rect.center, p, -1), inward) >= 0) continue;
      const t = dot(offset, direction),
        end = t + dot(f, direction);
      const lo = Math.max(0, Math.min(t, end)),
        hi = Math.min(length, Math.max(t, end));
      if (hi - lo > epsilon) return [add(p, direction, lo), add(p, direction, hi)];
    }
  return null;
}
function barrierAcross(edge, face, normal, epsilon) {
  const a = add(edge[0], face.rect.center, -1),
    b = add(edge[1], face.rect.center, -1);
  if (Math.abs(dot(a, face.n)) > epsilon || Math.abs(dot(b, face.n)) > epsilon) return false;
  const height = -dot(a, normal),
    extent = Math.abs(dot(face.rect.axisU, normal)) + Math.abs(dot(face.rect.axisV, normal));
  if (height + extent <= epsilon) return false;
  let lo = 0,
    hi = 1;
  for (const d of [face.du, face.dv]) {
    const x = dot(a, d),
      delta = dot(b, d) - x;
    if (Math.abs(delta) < 1e-12) {
      if (Math.abs(x) > 1 + 1e-12) return false;
      continue;
    }
    const u = (-1 - x) / delta,
      v = (1 - x) / delta;
    lo = Math.max(lo, Math.min(u, v));
    hi = Math.min(hi, Math.max(u, v));
  }
  return hi - lo > 1e-10;
}
function components(members, normal, all) {
  const parent = members.map((_, i) => i);
  function root(i) {
    while (parent[i] !== i) i = parent[i];
    return i;
  }
  for (let i = 0; i < members.length; i++)
    for (let j = i + 1; j < members.length; j++) {
      const a = members[i].frame,
        b = members[j].frame;
      const epsilon =
        128 *
        Number.EPSILON *
        Math.max(
          1,
          ...a.rect.center.map(Math.abs),
          ...b.rect.center.map(Math.abs),
          Math.hypot(...a.rect.axisU),
          Math.hypot(...a.rect.axisV),
          Math.hypot(...b.rect.axisU),
          Math.hypot(...b.rect.axisV),
        );
      const edge = commonEdge(a, b, epsilon);
      if (!edge || all.some((f) => f !== a && f !== b && barrierAcross(edge, f, normal, epsilon))) continue;
      parent[root(i)] = root(j);
    }
  const groups = new Map();
  for (const [i, member] of members.entries()) {
    const k = root(i);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(member);
  }
  return groups.values();
}

// Near a parallel plate, the incident field changes over approximately the
// distance between the two planes. Sample on both sides of its silhouette.
// Ordinary broad surfaces keep their regular, hardware-interpolated lookup.
function placeContacts(g, sheets, fullContacts) {
  const ranges = [g.du, g.dv],
    scale = [Math.hypot(...g.u), Math.hypot(...g.v)];
  const candidates = [];
  const contacts = [[], []];
  for (const { rect, n, corners } of sheets) {
    if (Math.abs(dot(n, g.n)) < 1e-8 && (fullContacts || Math.min(g.w, g.h) === 2)) {
      const heights = corners.map((p) => dot(add(p, g.origin, -1), g.n));
      if (
        Math.min(...heights) < 1e-6 &&
        Math.max(...heights) > 1e-5 &&
        heights.some((h) => Math.abs(h) < 1e-6)
      ) {
        const projected = ranges.map((axis) => corners.map((p) => dot(axis, p) + axis[3]));
        for (let d = 0; d < 2; d++) {
          const a = projected[d],
            b = projected[1 - d],
            c = a[0];
          if (
            c <= 0.001 ||
            c >= 0.999 ||
            Math.max(...a) - Math.min(...a) > 1e-7 ||
            Math.min(...b) < -0.00001 ||
            Math.max(...b) > 1.00001
          )
            continue;
          const delta = (0.004 * Math.min(...scale)) / scale[d];
          contacts[d].push(c - delta, c + delta);
        }
      }
    }
    if (Math.abs(dot(n, g.n)) < 1 - 1e-8) continue;
    const gap = dot(add(rect.center, g.origin, -1), g.n);
    if (gap < 1e-5 || gap > 0.15 * Math.min(Math.hypot(...rect.axisU), Math.hypot(...rect.axisV))) continue;
    // A receiver-aligned L-shaped join can include space outside its geometry.
    // Bounds merely select sample coordinates; the path queries use actual geometry.
    const bounds = ranges.map((axis) => {
      const values = corners.map((p) => dot(axis, p) + axis[3]);
      return [Math.min(...values), Math.max(...values)];
    });
    if (
      bounds.some(([lo, hi]) => lo < -0.00001 || hi > 1.00001) ||
      bounds.every(([lo, hi]) => hi - lo > 0.99)
    )
      continue;
    candidates.push({ bounds, gap });
  }
  if (!candidates.length && !contacts.some((axis) => axis.length)) return;
  let coords = [g.w, g.h].map((count) => Array.from({ length: count }, (_, i) => i / (count - 1)));
  function mergeKnots(values) {
    return values.sort((a, b) => a - b).filter((v, i, list) => !i || v - list[i - 1] > 0.0001);
  }
  let hasContacts = false;
  for (const [d, edges] of contacts.entries())
    if (edges.length) {
      const knots = mergeKnots([...coords[d], ...edges]);
      const axis = mergeKnots([
        ...knots,
        ...knots.slice(1).flatMap((v, i) => (v - knots[i] > 0.22 ? [(v + knots[i]) * 0.5] : [])),
      ]);
      if (axis.length <= (fullContacts ? 23 : 15)) {
        coords[d] = axis;
        hasContacts = true;
      }
    }
  const nearer = [];
  for (const { bounds, gap } of candidates.sort((a, b) => a.gap - b.gap)) {
    if (nearer.some((prev) => bounds.every(([a, b], i) => a >= prev[i][0] && b <= prev[i][1]))) continue;
    const next = coords.map((axis, d) => {
      const [lo, hi] = bounds[d],
        delta = (1.8 * gap) / scale[d];
      const cuts = [
        lo - delta,
        lo + delta,
        lo + 3.5 * delta,
        0.5 * (lo + hi),
        hi - 3.5 * delta,
        hi - delta,
        hi + delta,
      ].filter((v) => v > 0 && v < 1);
      return [
        ...axis.filter((v) => v === 0 || v === 1 || cuts.every((w) => Math.abs(v - w) > delta * 0.5)),
        ...cuts,
      ]
        .sort((a, b) => a - b)
        .filter((v, i, list) => !i || v - list[i - 1] > 0.0001);
    });
    // 15 breakpoints per axis leave one component for the count.
    if (next.some((axis) => axis.length > 15)) continue;
    nearer.push(bounds);
    coords = next;
  }
  if (!nearer.length && !hasContacts) return;
  g.coords = coords;
  g.w = coords[0].length;
  g.h = coords[1].length;
}

function grid(origin, u, v, n) {
  const a = Math.hypot(...u),
    b = Math.hypot(...v);
  const du = mul(u, 1 / (a * a)),
    dv = mul(v, 1 / (b * b));
  const w = clamp(1 + Math.ceil(a / 1.4), 2, 10),
    h = clamp(1 + Math.ceil(b / 1.4), 2, 10);
  const g = { origin, u, v, n, w, h, du: [...du, -dot(origin, du)], dv: [...dv, -dot(origin, dv)] };
  return g;
}
// Rectangles can be edited in place by the calling game. Comparing their
// coordinates keeps the blind-only path cheap without relying on immutability.
export function planarCharts(rectangles, cache, contacts = []) {
  const geometry = rectangles.flatMap((r) => [...r.center, ...r.axisU, ...r.axisV]);
  if (cache.geometry?.length === geometry.length && geometry.every((v, i) => v === cache.geometry[i]))
    return cache.charts;
  const slots = [],
    groups = new Map(),
    grids = [];
  const all = rectangles.map(sheetFrame);
  for (const [id, frame] of all.entries()) {
    const { rect: r, n } = frame;
    let canonical = n;
    const major = n.findIndex((x) => Math.abs(x) > 1e-8);
    if (n[major] < 0) canonical = mul(n, -1);
    const plane = dot(canonical, r.center);
    const key = [...canonical, plane].map((x) => Math.round(x * 1e8)).join('/');
    let g = groups.get(key);
    if (!g) {
      const u = unit(r.axisU),
        v = unit(cross(canonical, u));
      g = { n: canonical, u, v, plane, members: [] };
      groups.set(key, g);
    }
    g.members.push({ id, r, n, frame });
    slots[id] = [];
  }
  for (const g of groups.values()) {
    for (const sign of [1, -1]) {
      const n = mul(g.n, sign);
      for (const group of components(g.members, n, all)) {
        const lo = [Infinity, Infinity],
          hi = [-Infinity, -Infinity];
        for (const { r } of group)
          for (const [k, ax] of [g.u, g.v].entries()) {
            const extent = Math.abs(dot(ax, r.axisU)) + Math.abs(dot(ax, r.axisV)),
              c = dot(ax, r.center);
            lo[k] = Math.min(lo[k], c - extent);
            hi[k] = Math.max(hi[k], c + extent);
          }
        const o = add(add(mul(g.n, g.plane), g.u, lo[0]), g.v, lo[1]);
        const field = grid(o, mul(g.u, hi[0] - lo[0]), mul(g.v, hi[1] - lo[1]), n);
        placeContacts(
          field,
          all,
          contacts.some(
            (f) => group.length === 1 && group[0].r.id === f.receiver.id && dot(n, f.n) > 0.99999,
          ),
        );
        grids.push(field);
        for (const m of group) slots[m.id][dot(m.n, n) > 0 ? 0 : 1] = field;
      }
    }
  }
  cache.geometry = geometry;
  return (cache.charts = { slots, grids });
}

export function fieldLayout(scene, cache = {}) {
  const planar = planarCharts(scene.rectangles, cache),
    slots = [...planar.slots],
    grids = [...planar.grids];
  for (const [i, box] of (scene.roundedBoxes || []).entries()) {
    const faces = [];
    for (let d = 0; d < 3; d++) {
      const a = (d + 1) % 3,
        b = (d + 2) % 3;
      // Face samples live on the flat core section, just inside its fillet.
      const u = mul(box.axes[a], 2 * (box.halfSize[a] - box.radius * 0.999));
      const v = mul(box.axes[b], 2 * (box.halfSize[b] - box.radius * 0.999));
      for (const sign of [1, -1]) {
        const n = mul(box.axes[d], sign),
          c = add(box.center, n, box.halfSize[d]);
        faces.push(grid(add(add(c, u, -0.5), v, -0.5), u, v, n));
      }
    }
    grids.push(...faces);
    slots[scene.rectangles.length + i] = faces;
  }
  let x = 0,
    y = 0,
    rowHeight = 0,
    count = 0;
  const width = Math.max(16, Math.ceil(Math.sqrt(grids.reduce((s, g) => s + g.w * g.h, 0))));
  for (const g of grids) {
    if (x + g.w > width) {
      x = 0;
      y += rowHeight;
      rowHeight = 0;
    }
    g.x = x;
    g.y = y;
    x += g.w;
    rowHeight = Math.max(rowHeight, g.h);
    count += g.w * g.h;
  }
  const height = Math.max(1, y + rowHeight);
  const points = new Float32Array(width * height * 4),
    normals = new Float32Array(points.length);
  let nextRow = Math.max(1, slots.length) * 6;
  for (const g of grids)
    if (g.coords) {
      g.coordinateRow = nextRow;
      nextRow += 2;
    }
  const descriptors = new Float32Array(nextRow * 16);
  for (const g of grids)
    if (g.coords)
      for (const [d, axis] of g.coords.entries()) {
        const k = (g.coordinateRow + d) * 16;
        descriptors.set(axis, k);
        descriptors[k + 15] = axis.length;
      }
  for (const [index, g] of grids.entries()) {
    for (let j = 0; j < g.h; j++)
      for (let i = 0; i < g.w; i++) {
        // Take the interior limit at a wall boundary. A point exactly in an
        // intersecting plane can otherwise launch paths straight through it.
        const su = g.coords ? g.coords[0][i] : i / (g.w - 1);
        const sv = g.coords ? g.coords[1][j] : j / (g.h - 1);
        const p = add(add(g.origin, g.u, clamp(su, 0.00015, 0.99985)), g.v, clamp(sv, 0.00015, 0.99985));
        const idx = ((g.y + j) * width + g.x + i) * 4;
        // Seed the sample sequence in surface coordinates, independent of atlas packing.
        points.set([...p, 1 + i + j * 16 + index * 256], idx);
        normals.set([...g.n, index + 1], idx);
      }
  }
  for (const [id, faces] of slots.entries())
    for (const [i, g] of faces.entries()) {
      const k = (6 * id + i) * 16;
      descriptors.set([g.x, g.y, g.coordinateRow ? -g.coordinateRow : g.w, g.h], k);
      descriptors.set(g.du, k + 4);
      descriptors.set(g.dv, k + 8);
      descriptors.set([...g.n, faces.length === 6 ? 1 : 0], k + 12);
    }
  return {
    width,
    height,
    points,
    normals,
    descriptors,
    count,
    objectCount: slots.length,
    planarFaces: planar.slots,
  };
}

export const chartMappingGLSL = `
float coordinateAt(int row,float t) {
  vec4 values[4];
  for(int i=0;i<4;i++) values[i]=texelFetch(irradianceMap,ivec2(i,row),0);
  int lo=0, hi=int(values[3].w)-1;
  for(int k=0;k<4;k++) {
    int mid=(lo+hi)/2;
    if(t>=values[mid>>2][mid&3])lo=mid;else hi=mid;
  }
  float a=values[lo>>2][lo&3], b=values[hi>>2][hi&3];
  return float(lo)+clamp((t-a)/max(b-a,1e-8),0.,1.);
}
vec2 chartAddress(int row, vec3 p){
  vec4 grid=texelFetch(irradianceMap,ivec2(0,row),0);
  vec4 u=texelFetch(irradianceMap,ivec2(1,row),0),v=texelFetch(irradianceMap,ivec2(2,row),0);
  vec2 uv=clamp(vec2(dot(u.xyz,p)+u.w,dot(v.xyz,p)+v.w),0.,1.);
  vec2 offset;
  if(grid.z<0.) {
    int mapRow=int(-grid.z);
    offset=vec2(coordinateAt(mapRow,uv.x),coordinateAt(mapRow+1,uv.y));
  } else offset=uv*(grid.zw-1.);
  return (grid.xy+vec2(.5)+offset)/vec2(textureSize(irradianceField,0));
}`;
export const indirectGLSL = `
uniform highp sampler2D irradianceField, irradianceMap;
${chartMappingGLSL}
vec3 fieldFace(int row,vec3 p){
  return texture(irradianceField,chartAddress(row,p)).rgb;
}
vec3 indirectAt(int id,vec3 p,vec3 n){
 int row=6*id;
 vec4 f=texelFetch(irradianceMap,ivec2(3,row),0);
 if(f.w==0.)return fieldFace(row+int(dot(f.xyz,n)<0.),p);
 vec3 v=vec3(dot(n,f.xyz),dot(n,texelFetch(irradianceMap,ivec2(3,row+2),0).xyz),
  dot(n,texelFetch(irradianceMap,ivec2(3,row+4),0).xyz));
 return v.x*v.x*fieldFace(row+int(v.x<0.),p)+v.y*v.y*fieldFace(row+2+int(v.y<0.),p)
   +v.z*v.z*fieldFace(row+4+int(v.z<0.),p);
}
`;

// Bidirectional map between physical sheet coordinates and a texture lattice.
// Both the sample generator and the presentation vertices use the same knots.
export function chartCoordinate(g, d, t) {
  const knots = g.coords?.[d];
  if (!knots) return t;
  let i = 0;
  while (i + 2 < knots.length && t > knots[i + 1]) i++;
  return (i + (t - knots[i]) / (knots[i + 1] - knots[i])) / (knots.length - 1);
}
export function chartPoint(g, u, v) {
  const local = [u, v];
  if (g.coords)
    for (let d = 0; d < 2; d++) {
      const knots = g.coords[d],
        s = local[d] * (knots.length - 1),
        i = Math.min(knots.length - 2, Math.floor(s));
      local[d] = knots[i] + (s - i) * (knots[i + 1] - knots[i]);
    }
  return g.origin.map((a, i) => a + g.u[i] * local[0] + g.v[i] * local[1]);
}
// Inside each chart interval the world-to-texture coordinate map is affine.
// Split only the presentation polygons where a change of slope requires it.
// These triangles neither define scene geometry nor discretize the light.
export function chartPolygons(polygon, faces) {
  let cells = [polygon];
  for (const face of faces) {
    if (!face.coords) continue;
    for (const [d, knots] of face.coords.entries()) {
      const axis = d === 0 ? face.du : face.dv;
      const next = [];
      for (let cell of cells) {
        // Knots increase along this axis. Once the low half is emitted, no
        // later knot can cut it. A convex cell has only one remaining high half.
        let distances = cell.map((p) => dot(axis, p) + axis[3]);
        let lo = Math.min(...distances),
          hi = Math.max(...distances);
        for (let i = 1; i + 1 < knots.length; i++) {
          const knot = knots[i];
          if (hi - knot < 1e-10) break;
          if (lo - knot > -1e-10 || Math.abs(2 * knot - knots[i - 1] - knots[i + 1]) < 1e-10) continue;
          const low = [],
            high = [],
            highDistances = [];
          for (let k = 0; k < cell.length; k++) {
            const j = (k + 1) % cell.length,
              a = cell[k],
              b = cell[j];
            const x = distances[k] - knot,
              y = distances[j] - knot;
            if (x <= 0) low.push(a);
            if (x >= 0) {
              high.push(a);
              highDistances.push(distances[k]);
            }
            if ((x < 0 && y > 0) || (x > 0 && y < 0)) {
              const t = x / (x - y),
                point = a.map((v, m) => v + t * (b[m] - v));
              low.push(point);
              high.push(point);
              highDistances.push(dot(axis, point) + axis[3]);
            }
          }
          if (low.length >= 3) next.push(low);
          cell = high;
          distances = highDistances;
          lo = knot;
        }
        if (cell.length >= 3) next.push(cell);
      }
      cells = next;
    }
  }
  return cells;
}
