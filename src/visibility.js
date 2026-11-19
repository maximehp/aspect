// Direct shadows from one point light on finite two-sided planar sheets.
// UV coordinates span [-1, 1] along each half-axis. The light and every
// sheet use the same local world coordinate system. Inputs must be finite,
// with unique sheet ids and two linearly independent half-axes.
const square = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
const roundoff = 32 * Number.EPSILON;

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const sameVec = (a, b) => a && b && a.every((v, i) => v === b[i]);

function sameGeometry(a, b) {
  return a && sameVec(a.center, b.center) &&
    sameVec(a.axisU, b.axisU) && sameVec(a.axisV, b.axisV);
}

// Orient a plane normal into the shadow. A lamp in the sheet's plane
// produces no positive-solid-angle shadow. IEEE doubles and the relative
// tolerance here suit room-scale, well-conditioned geometry. Grazing or
// nearly coincident inputs within rounding error have no exact guarantee.
function orientedPlane(normal, origin, toward) {
  const length = Math.hypot(...normal);
  if (!length) return null;
  const offset = dot(normal, toward);
  if (Math.abs(offset) <= roundoff * length * Math.hypot(...toward)) return null;
  const scale = Math.sign(offset) / length;
  return { normal: normal.map(value => value * scale), origin };
}

function shadowVolume(rect, lamp) {
  const { center, axisU, axisV } = rect;
  const toCenter = subtract(center, lamp);
  // The first plane rules out all receivers in front of the blocker.
  const back = orientedPlane(cross(axisU, axisV), center, toCenter);
  if (!back) return null;
  const corners = square.map(([u, v]) =>
    toCenter.map((d, i) => d + u * axisU[i] + v * axisV[i]));
  const planes = [back];
  for (let i = 0; i < 4; i++) {
    const edge = subtract(corners[(i + 1) % 4], corners[i]);
    const side = orientedPlane(cross(corners[i], edge), lamp, toCenter);
    if (!side) return null;
    planes.push(side);
  }
  return planes;
}

function hasArea(vertices) {
  if (vertices.length < 3) return false;
  const ox = vertices[0][0], oy = vertices[0][1];
  let area = 0;
  for (let i = 1; i < vertices.length - 1; i++) {
    const p = vertices[i], q = vertices[i + 1];
    area += (p[0] - ox) * (q[1] - oy) - (p[1] - oy) * (q[0] - ox);
  }
  return area > 0;
}

// Split both sides together so they get literally the same intersection.
// Marking a whole polygon coplanar assigns it to the positive half-space.
function split(vertices, a, b, c, tolerance, keepOutside = true) {
  let positive = false, negative = false;
  for (const p of vertices) {
    const d = a * p[0] + b * p[1] + c;
    positive ||= d > tolerance;
    negative ||= d < -tolerance;
    if (positive && negative) break;
  }
  if (!negative) return [vertices, []];
  if (!positive) return [[], vertices];
  const inside = [], outside = [];
  for (let i = 0; i < vertices.length; i++) {
    const j = (i + 1) % vertices.length;
    const p = vertices[i], q = vertices[j];
    let dp = a * p[0] + b * p[1] + c, dq = a * q[0] + b * q[1] + c;
    if (Math.abs(dp) <= tolerance) dp = 0;
    if (Math.abs(dq) <= tolerance) dq = 0;
    if (dp >= 0) inside.push(p);
    if (keepOutside && dp <= 0) outside.push(p);
    if ((dp > 0 && dq < 0) || (dp < 0 && dq > 0)) {
      const t = dp / (dp - dq);
      const crossing = [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])];
      inside.push(crossing);
      if (keepOutside) outside.push(crossing);
    }
  }
  return [inside, outside];
}

function projectedFootprint(receiver, planes) {
  if (!planes) return null;
  let uv = square, previousOrigin, delta, tolerance;
  const size = Math.hypot(...receiver.axisU) + Math.hypot(...receiver.axisV);
  for (let i = 0; i < planes.length; i++) {
    const { normal, origin } = planes[i];
    if (origin !== previousOrigin) {
      delta = subtract(receiver.center, origin);
      tolerance = roundoff * (Math.hypot(...delta) + size);
      previousOrigin = origin;
    }
    const a = dot(normal, receiver.axisU), b = dot(normal, receiver.axisV);
    const c = dot(normal, delta), radius = Math.abs(a) + Math.abs(b);
    // Errors in dot products depend on the input lengths, including for a
    // receiver exactly coplanar with an adjacent panel.
    // Points on the blocker's own plane only touch the end of the light
    // segment. Coplanar receiver sheets must never shadow one another.
    if (i === 0 && c + radius <= tolerance) return null;
    if (c + radius < -tolerance) return null;
    if (c - radius >= -tolerance) continue;
    const [inside] = split(uv, a, b, c, tolerance, false);
    if (inside !== uv && !hasArea(inside)) return null;
    uv = inside;
  }
  return uv;
}

function sameFootprint(a, b) {
  if (!a || !b) return !a && !b;
  return a === b || (a.length === b.length && a.every((p, i) => sameVec(p, b[i])));
}

function footprintEdges(polygon) {
  return polygon.map(([x, y], i) => {
    const [nx, ny] = polygon[(i + 1) % polygon.length];
    const a = y - ny, b = nx - x, c = -(a * x + b * y);
    return [a, b, c, roundoff * (Math.abs(a) + Math.abs(b) + Math.abs(c))];
  });
}

const polygonBounds = new WeakMap();
export function boundsOf(uv) {
  let box = polygonBounds.get(uv);
  if (!box) {
    let left = Infinity, right = -Infinity, bottom = Infinity, top = -Infinity;
    for (const p of uv) {
      const u = p[0], v = p[1];
      left = Math.min(left, u);
      right = Math.max(right, u);
      bottom = Math.min(bottom, v);
      top = Math.max(top, v);
    }
    box = [left, right, bottom, top];
    polygonBounds.set(uv, box);
  }
  return box;
}
function partitionReceiver(footprints, initial) {
  let lit = initial ? initial.filter(c => c.lit).map(c => c.uv) : [square];
  const dark = initial ? initial.filter(c => !c.lit).map(c => c.uv) : [];
  for (const polygon of footprints.values()) {
    if (!lit.length) break;
    const edges = footprintEdges(polygon), next = [], mask = boundsOf(polygon);
    for (const original of lit) {
      const b = boundsOf(original);
      if (mask[0] >= b[1] || mask[1] <= b[0] || mask[2] >= b[3] || mask[3] <= b[2]) {
        next.push(original);
        continue;
      }
      const uncovered = [];
      let intersection = original;
      for (const edge of edges) {
        const [inside, outside] = split(intersection, edge[0], edge[1], edge[2], edge[3]);
        if (inside === intersection) continue;
        if (!hasArea(inside)) {
          intersection = null;
          break;
        }
        intersection = inside;
        if (hasArea(outside)) uncovered.push(outside);
      }
      if (intersection) {
        dark.push(intersection);
        next.push(...uncovered);
      } else {
        // A disjoint footprint should not leave artificial cuts behind.
        next.push(original);
      }
    }
    lit = next;
  }
  return [...lit.map(uv => ({ uv, lit: true })), ...dark.map(uv => ({ uv, lit: false }))];
}

// Sheets are immutable once submitted. Stable ids allow scene editors to
// replace a sheet object without discarding unrelated visibility work.
// Retained pair data consists only of nonempty footprints. Changing one
// blocker tests its footprint on each receiver plus all blockers on the
// changed receiver. Full-scene builds still test quadratically many pairs.
export class VisibilityCompiler {
  constructor() {
    this.records = new Map();
    this.lamp = null;
  }

  update(rectangles, lightPosition) {
    const lightMoved = !sameVec(this.lamp, lightPosition);
    const changedBlockers = [], nextRecords = new Map();
    const changedReceivers = new Set();
    for (const rect of rectangles) {
      if (nextRecords.has(rect.id)) throw new Error('Duplicate rectangle id: ' + rect.id);
      const old = this.records.get(rect.id);
      const geometryChanged = !old || !sameGeometry(old.rect, rect);
      const blockerChanged = lightMoved || geometryChanged;
      if (geometryChanged || lightMoved) changedReceivers.add(rect.id);
      if (blockerChanged) changedBlockers.push(rect.id);
      nextRecords.set(rect.id, {
        rect, volume: blockerChanged ? shadowVolume(rect, lightPosition) : old.volume,
        footprints: old?.footprints ?? new Map(), cells: old?.cells,
      });
    }
    for (const id of this.records.keys()) if (!nextRecords.has(id)) changedBlockers.push(id);

    const surfaces = [], stats = { receiversRebuilt: 0, pairsTested: 0, cellCount: 0 };
    for (const [id, record] of nextRecords) {
      const full = changedReceivers.has(id);
      let rebuild = full;
      if (full) record.footprints = new Map();
      const candidates = full ? nextRecords.keys() : changedBlockers;
      for (const other of candidates) {
        if (other === id) continue;
        const volume = nextRecords.get(other)?.volume;
        if (volume) stats.pairsTested++;
        const footprint = projectedFootprint(record.rect, volume);
        if (sameFootprint(record.footprints.get(other), footprint)) continue;
        rebuild = true;
        if (footprint) record.footprints.set(other, footprint);
        else record.footprints.delete(other);
      }
      if (rebuild) {
        record.cells = partitionReceiver(record.footprints);
        stats.receiversRebuilt++;
      }
      surfaces.push({ rect: record.rect, cells: record.cells });
      stats.cellCount += record.cells.length;
    }
    this.lamp = [...lightPosition];
    this.records = nextRecords;
    return { surfaces, stats };
  }
}

// CPU projections used by camera visibility and diffuse transfer.
export { shadowVolume, projectedFootprint, partitionReceiver, split };
