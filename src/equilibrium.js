// Diffuse power transfer by reciprocal projected area. The unknowns are mean
// incident RGB on each side. Lamp flux integrates over actual lit polygons.
import { contactReceivers } from './contact-transfer.js';
import { compileParallel, activeParallel } from './parallel-transfer.js';
import { boundsOf, split, VisibilityCompiler } from './visibility.js';
import { dot, sub, cross, unit, point, broadFaces, volume, regions } from './cpu-geometry.js';
function solidAngle(vertices, eye) {
  const dirs = vertices.map((p) => unit(sub(p, eye))),
    a = dirs[0];
  let sum = 0;
  for (let i = 1; i + 1 < dirs.length; i++) {
    const b = dirs[i],
      c = dirs[i + 1];
    sum += 2 * Math.atan2(Math.abs(dot(a, cross(b, c))), 1 + dot(a, b) + dot(a, c) + dot(b, c));
  }
  return sum;
}
// Nearby parallel sheets enclose a narrow cavity. Resolve its footprint in the
// transport basis; its edge transition has a length scale set by the gap.
function nearSheets(rect, others) {
  const normal = unit(cross(rect.axisU, rect.axisV)),
    hits = [];
  for (const other of others) {
    if (other === rect) continue;
    const n = unit(cross(other.axisU, other.axisV));
    if (Math.abs(dot(n, normal)) < 1 - 1e-8) continue;
    const offset = dot(sub(other.center, rect.center), normal),
      gap = Math.abs(offset);
    if (gap < 1e-7 || gap > 0.14 * Math.min(Math.hypot(...other.axisU), Math.hypot(...other.axisV))) continue;
    const corners = [
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, 1],
    ].map(([u, v]) => sub(point(other, u, v), rect.center));
    const bounds = [rect.axisU, rect.axisV]
      .map((axis) => corners.map((v) => dot(v, axis) / dot(axis, axis)))
      .map((values) => [Math.min(...values), Math.max(...values)]);
    if (bounds.some(([lo, hi]) => lo < -1.00001 || hi > 1.00001)) continue;
    if (![rect.axisU, rect.axisV].some((a) => Math.abs(dot(unit(a), unit(other.axisU))) > 0.999999)) continue;
    hits.push({
      sheet: other,
      bounds,
      side: Number(offset < 0),
      gap,
      center: other.center,
      axes: [other.axisU, other.axisV].map((a) => [unit(a), Math.hypot(...a)]),
    });
  }
  const keep = [];
  for (const h of hits.sort((a, b) => a.gap - b.gap)) {
    if (
      !keep.some(
        (q) =>
          q.side === h.side && h.bounds.every(([lo, hi], i) => lo >= q.bounds[i][0] && hi <= q.bounds[i][1]),
      )
    )
      keep.push(h);
  }
  return keep;
}
export function cavityWeight(p, h) {
  let edge = Infinity;
  for (const [axis, length] of h.axes) edge = Math.min(edge, length - Math.abs(dot(sub(p, h.center), axis)));
  // Smooth plane-edge coverage for cosine-weighted diffuse directions.
  return 0.5 + (0.5 * edge) / Math.hypot(edge, h.gap);
}
function elements(rect, covers, lamp, edgeGaps) {
  const lampDelta = sub(lamp, rect.center);
  const axes = [rect.axisU, rect.axisV],
    normal = unit(cross(...axes));
  const distance = Math.abs(dot(lampDelta, normal));
  const duals = [cross(rect.axisV, normal), cross(normal, rect.axisU)];
  const area = dot(rect.axisU, duals[0]);
  const projected = duals.map((axis) => dot(lampDelta, axis) / area);
  // The inverse-square field varies on the scale of the closest lamp-to-sheet
  // distance. Plane distance alone introduces thin rows across a distant
  // receiver when the lamp projection lies outside its finite footprint.
  let featureScale = distance;
  if (!rect.owner && projected.some((t) => Math.abs(t) > 1)) {
    featureScale = Infinity;
    for (const [d, axis] of axes.entries()) {
      const edge = axes[1 - d];
      for (const sign of [-1, 1]) {
        const delta = lampDelta.map((x, i) => x - sign * axis[i]);
        const t = Math.max(-1, Math.min(1, dot(delta, edge) / dot(edge, edge)));
        featureScale = Math.min(featureScale, Math.hypot(...delta.map((x, i) => x - t * edge[i])));
      }
    }
  }
  const knots = axes.map((axis, d) => {
    const count = Math.min(5, Math.ceil(Math.hypot(...axis) / 0.82));
    const cuts = Array.from({ length: count + 1 }, (_, i) => (2 * i) / count - 1);
    const length = Math.hypot(...axis);
    if (!rect.owner && distance > 1e-5 && length / count > featureScale) {
      const center = projected[d];
      for (const k of [-3, -1, 0, 1, 3]) {
        const t = center + (k * featureScale) / length;
        if (t > -0.99999 && t < 0.99999) cuts.push(t);
      }
    }
    for (const gap of edgeGaps) {
      const delta = (2 * gap) / length;
      if (delta < 0.49) cuts.push(-1 + delta, 1 - delta);
    }
    for (const h of covers)
      for (const edge of h.bounds[d])
        for (const k of h.gap < 0.015 * Math.min(...h.axes.map((a) => a[1])) ? [0] : [-2, 0, 2]) {
          const t = edge + (k * h.gap) / length;
          if (t > -0.99999 && t < 0.99999) cuts.push(t);
        }
    return cuts.sort((a, b) => a - b).filter((v, i, list) => !i || v - list[i - 1] > 1e-5);
  });
  const [us, vs] = knots,
    nu = us.length - 1,
    nv = vs.length - 1,
    list = [];
  for (let v = 0; v < nv; v++)
    for (let u = 0; u < nu; u++) {
      const u0 = us[u],
        u1 = us[u + 1],
        v0 = vs[v],
        v1 = vs[v + 1];
      const r = {
        center: point(rect, (u0 + u1) / 2, (v0 + v1) / 2),
        axisU: rect.axisU.map((x) => (x * (u1 - u0)) / 2),
        axisV: rect.axisV.map((x) => (x * (v1 - v0)) / 2),
        color: rect.color,
        owner: rect.owner || rect,
        rect,
        nu,
        nv,
        bounds: [u0, u1, v0, v1],
      };
      const axes = [r.axisU, r.axisV],
        lengths = axes.map((a) => Math.hypot(...a));
      let axis = lengths[0] < lengths[1] ? 0 : 1;
      if (!rect.owner) {
        // Prefer the axis along which the lamp term changes most across this mode.
        const delta = sub(lamp, r.center);
        const scores = axes.map((a) => Math.abs(dot(delta, a)) / Math.max(1e-12, dot(delta, delta)));
        const coords = [(u0 + u1) / 2, (v0 + v1) / 2],
          half = [Math.hypot(...rect.axisU), Math.hypot(...rect.axisV)];
        function edge(bounds, gap) {
          const d = bounds.map(([lo, hi], k) => Math.min(coords[k] - lo, hi - coords[k]) * half[k]);
          const k = d[0] < d[1] ? 0 : 1,
            z = d[k] / gap;
          scores[k] = Math.max(scores[k], lengths[k] / (gap * Math.pow(1 + z * z, 1.5)));
        }
        for (const h of covers) edge(h.bounds, h.gap);
        for (const gap of edgeGaps)
          edge(
            [
              [-1, 1],
              [-1, 1],
            ],
            gap,
          );
        axis = scores[0] > scores[1] ? 0 : 1;
      }
      const direction = axes[axis];
      r.momentAxis = direction.map((x) => x / dot(direction, direction));
      r.normal = unit(cross(r.axisU, r.axisV));
      r.area = 4 * Math.hypot(...cross(r.axisU, r.axisV));
      list.push(r);
    }
  return list;
}
function lampFlux(mode, lit, lamp) {
  const [u0, u1, v0, v1] = mode.bounds;
  let flux = 0,
    moment = 0;
  const lampOffset = sub(lamp, mode.center),
    offset = dot(lampOffset, mode.momentAxis),
    height = Math.abs(dot(lampOffset, mode.normal));
  for (const cell of lit) {
    if (!cell.lit) continue;
    const [left, right, bottom, top] = (cell.bounds ??= boundsOf(cell.uv));
    if (right <= u0 || left >= u1 || top <= v0 || bottom >= v1) continue;
    let poly = cell.uv;
    if (left < u0) [poly] = split(poly, 1, 0, -u0, 1e-12, false);
    if (poly.length >= 3 && right > u1) [poly] = split(poly, -1, 0, u1, 1e-12, false);
    if (poly.length >= 3 && bottom < v0) [poly] = split(poly, 0, 1, -v0, 1e-12, false);
    if (poly.length >= 3 && top > v1) [poly] = split(poly, 0, -1, v1, 1e-12, false);
    if (poly.length < 3) continue;
    const vertices = poly.map(([u, v]) => point(mode.rect, u, v));
    const omega = solidAngle(vertices, lamp);
    flux += omega;
    moment += offset * omega;
    // Integrate the first moment of h/r³ by the boundary integral of 1/r.
    for (let i = 0; i < vertices.length; i++) {
      const a = vertices[i],
        b = vertices[(i + 1) % vertices.length],
        edge = sub(b, a),
        length = Math.hypot(...edge);
      if (!length || !height) continue;
      const radii = Math.hypot(...sub(a, lamp)) + Math.hypot(...sub(b, lamp));
      moment -=
        ((height * dot(cross(edge, mode.normal), mode.momentAxis)) / length) *
        Math.log1p((2 * length) / (radii - length));
    }
  }
  return [flux / mode.area, Math.max(-flux, Math.min(flux, 3 * moment)) / mode.area];
}
function sameVector(a, b) {
  return a.length === b.length && a.every((v, i) => v === b[i]);
}
function sameShape(a, b) {
  if (a === b) return true;
  if (!a || a.id !== b.id || !sameVector(a.center, b.center)) return false;
  if (a.axes || b.axes)
    return (
      a.axes &&
      b.axes &&
      a.radius === b.radius &&
      sameVector(a.halfSize, b.halfSize) &&
      a.axes.every((v, i) => sameVector(v, b.axes[i]))
    );
  return sameVector(a.axisU, b.axisU) && sameVector(a.axisV, b.axisV);
}
export class Equilibrium {
  constructor(sweep) {
    this.sweep = sweep;
    this.visibility = new VisibilityCompiler();
    this.flux = new WeakMap();
    this.elements = new WeakMap();
    this.boxFaces = new WeakMap();
  }
  update(scene, pool, geometryReady) {
    const start = performance.now();
    const lamp = scene.lightPosition;
    const staticChanged =
      !this.sheets ||
      this.sheets.length !== scene.rectangles.length ||
      scene.rectangles.some((r, i) => !sameShape(this.sheets[i], r));
    const boxes = scene.roundedBoxes.map((b, i) => (sameShape(this.boxes?.[i], b) ? this.boxes[i] : b));
    const boxesChanged =
      !this.boxes || this.boxes.length !== boxes.length || boxes.some((b, i) => b !== this.boxes[i]);
    this.boxes = boxes;
    if (boxesChanged) this.shadowState = null;
    if (staticChanged || !this.lamp || lamp.some((x, i) => x !== this.lamp[i])) {
      this.parallel = null;
      this.contacts = null;
      this.shadowState = null;
      this.lampVolumes = new WeakMap();
      this.receiverVisibility = new WeakMap();
      this.lamp = lamp.slice();
      this.fixedLamp = new Map(
        this.visibility.update(scene.rectangles, lamp).surfaces.map((s) => [s.rect, s.cells]),
      );
      this.sheets = scene.rectangles;
      this.elements = new WeakMap();
      this.covers = new Map(scene.rectangles.map((r) => [r, nearSheets(r, scene.rectangles)]));
      this.edges = new Map();
      for (const covers of this.covers.values())
        for (const h of covers) {
          if (!this.edges.has(h.sheet)) this.edges.set(h.sheet, []);
          this.edges.get(h.sheet).push(h.gap);
        }
    }
    const proxy = boxes.map((b) => {
      if (!this.boxFaces.has(b)) this.boxFaces.set(b, broadFaces(b));
      return this.boxFaces.get(b);
    });
    const sheets = [...this.sheets, ...proxy.flat()],
      modes = [],
      groups = new Map();
    for (const rect of sheets) {
      if (!this.elements.has(rect))
        this.elements.set(
          rect,
          elements(rect, this.covers.get(rect) || [], lamp, this.edges.get(rect) || []),
        );
      const group = this.elements.get(rect);
      groups.set(rect, { offset: modes.length, nu: group[0].nu, nv: group[0].nv });
      modes.push(...group);
    }
    this.contacts ??= contactReceivers(this.sheets, groups);
    const contacts = this.contacts;
    const original = new Map(this.sheets.map((r, i) => [r, scene.rectangles[i]]));
    boxes.forEach((b, i) => original.set(b, scene.roundedBoxes[i]));
    const colors = modes.map((m) => (original.get(m.owner) || m).color);
    this.parallel ??= compileParallel(this.sheets, groups, modes, this.covers);
    const near = activeParallel(this.parallel, boxes);
    const prepared = performance.now(),
      exchangeStart = prepared;
    const transfer = this.sweep(modes, near, pool);
    // The angular aperture depends only on geometry. Its GPU integration can
    // overlap both the global sweep and the construction of the lamp input.
    if (geometryReady && transfer.then) geometryReady({ boxes, modes, groups, contacts });
    const visibilityStart = performance.now();
    if (!this.shadowState) {
      const volumes = [...this.sheets, ...boxes].map((shape) => {
        if (!this.lampVolumes.has(shape)) this.lampVolumes.set(shape, volume(shape, lamp));
        return [shape, this.lampVolumes.get(shape)];
      });
      const moving = volumes.slice(this.sheets.length);
      const regionsBySheet = new Map();
      for (const r of sheets) {
        if (r.owner && dot(cross(r.axisU, r.axisV), sub(lamp, r.center)) <= 0) {
          regionsBySheet.set(r, []);
          continue;
        }
        const fixed = this.fixedLamp.get(r);
        let previous = this.receiverVisibility.get(r);
        // Moving faces often live for only one edit. Retain pair results once
        // a receiver actually survives into a second geometry revision.
        if (previous === undefined) this.receiverVisibility.set(r, null);
        else if (!previous) {
          previous = { pairs: new WeakMap() };
          this.receiverVisibility.set(r, previous);
        }
        regionsBySheet.set(r, regions(r, fixed ? moving : volumes, fixed, previous));
      }
      this.shadowState = { volumes, regionsBySheet };
    }
    const { volumes: lampVolumes, regionsBySheet } = this.shadowState,
      lampRegions = new Map(regionsBySheet),
      covers = new Map(this.covers);
    for (const r of this.sheets) {
      const authored = original.get(r);
      if (r === authored) continue;
      lampRegions.set(authored, regionsBySheet.get(r));
      covers.set(authored, covers.get(r));
      groups.set(authored, groups.get(r));
    }
    const visible = performance.now(),
      direct = new Float32Array(6 * modes.length),
      directSlope = new Float32Array(direct.length);
    for (let j = 0; j < modes.length; j++) {
      const r = modes[j],
        side = Number(dot(sub(lamp, r.center), r.normal) < 0);
      const cells = lampRegions.get(r.rect);
      let cache = this.flux.get(cells);
      if (!cache) {
        cache = new WeakMap();
        this.flux.set(cells, cache);
      }
      let flux = cache.get(r);
      if (flux === undefined) {
        flux = lampFlux(r, cells, lamp);
        cache.set(r, flux);
      }
      for (let c = 0; c < 3; c++) {
        const i = (2 * j + side) * 3 + c,
          f = colors[j][c] * scene.lampRGB[c];
        direct[i] = flux[0] * f;
        directSlope[i] = flux[1] * f;
      }
    }
    const lit = performance.now();
    function finish({ solve, bytes, links, workerBytes, workerMs, assemblyMs }) {
      const solveStart = performance.now(),
        result = solve(direct, directSlope, colors, contacts);
      const outgoing = Float32Array.from(
        direct,
        (d, k) => d + result.incident[k] * colors[Math.floor(k / 6)][k % 3],
      );
      return {
        timings: {
          workerMs,
          assemblyMs,
          basis: prepared - start,
          visibility: visible - visibilityStart,
          exchange: pool ? solveStart - exchangeStart : visibilityStart - exchangeStart,
          lamp: lit - visible,
          solve: performance.now() - solveStart,
        },
        contacts,
        modes,
        links,
        workerBytes,
        boxes,
        groups,
        near,
        outgoing,
        proxy,
        covers,
        lampRegions,
        lampVolumes,
        ...result,
        bytes:
          bytes + result.incident.byteLength + result.sidedSlopes.byteLength + result.outSlope.byteLength,
      };
    }
    return transfer.then ? transfer.then(finish) : finish(transfer);
  }
}
