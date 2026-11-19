// Diffuse power transfer by reciprocal projected area. The unknowns are mean
// incident RGB on each side. Lamp flux integrates over actual lit polygons.
import { compileParallel, applyParallel } from './parallel-transfer.js';
import { split, VisibilityCompiler } from './visibility.js';
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
  const lampDelta = sub(lamp, rect.center),
    distance = Math.abs(dot(lampDelta, unit(cross(rect.axisU, rect.axisV))));
  const knots = [rect.axisU, rect.axisV].map((axis, d) => {
    const count = Math.min(5, Math.ceil(Math.hypot(...axis) / 0.82));
    const cuts = Array.from({ length: count + 1 }, (_, i) => (2 * i) / count - 1);
    const length = Math.hypot(...axis);
    if (!rect.owner && distance > 1e-5 && length / count > distance) {
      const center = dot(lampDelta, axis) / (length * length);
      for (const k of [-3, -1, 0, 1, 3]) {
        const t = center + (k * distance) / length;
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
      const short = Math.hypot(...r.axisU) < Math.hypot(...r.axisV) ? r.axisU : r.axisV;
      r.momentAxis = short.map((x) => x / dot(short, short));
      r.normal = unit(cross(r.axisU, r.axisV));
      r.area = 4 * Math.hypot(...cross(r.axisU, r.axisV));
      list.push(r);
    }
  return list;
}
function lampFlux(mode, lit, lamp) {
  const [u0, u1, v0, v1] = mode.bounds;
  let flux = 0;
  for (const cell of lit) {
    if (!cell.lit) continue;
    if (!cell.bounds) {
      const u = cell.uv.map((p) => p[0]),
        v = cell.uv.map((p) => p[1]);
      cell.bounds = [Math.min(...u), Math.max(...u), Math.min(...v), Math.max(...v)];
    }
    const [left, right, bottom, top] = cell.bounds;
    if (right <= u0 || left >= u1 || top <= v0 || bottom >= v1) continue;
    let poly = cell.uv;
    for (const [a, b, c] of [
      [1, 0, -u0],
      [-1, 0, u1],
      [0, 1, -v0],
      [0, -1, v1],
    ]) {
      [poly] = split(poly, a, b, c, 1e-12);
      if (poly.length < 3) break;
    }
    if (poly.length >= 3)
      flux += solidAngle(
        poly.map(([u, v]) => point(mode.rect, u, v)),
        lamp,
      );
  }
  return flux / mode.area;
}
export class Equilibrium {
  constructor(sweep) {
    this.sweep = sweep;
    this.visibility = new VisibilityCompiler();
    this.flux = new WeakMap();
    this.elements = new WeakMap();
    this.boxFaces = new WeakMap();
  }
  update(scene) {
    const shapes = [...scene.rectangles, ...scene.roundedBoxes],
      lamp = scene.lightPosition;
    const staticChanged =
      !this.sheets ||
      this.sheets.length !== scene.rectangles.length ||
      scene.rectangles.some((r, i) => this.sheets[i] !== r);
    if (staticChanged || !this.lamp || lamp.some((x, i) => x !== this.lamp[i])) {
      this.parallel = null;
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
    const proxy = scene.roundedBoxes.map((b) => {
      if (!this.boxFaces.has(b)) this.boxFaces.set(b, broadFaces(b));
      return this.boxFaces.get(b);
    });
    const sheets = [...scene.rectangles, ...proxy.flat()],
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
    const lampVolumes = shapes.map((s) => [s, volume(s, lamp)]);
    const moving = lampVolumes.slice(scene.rectangles.length);
    const lampRegions = new Map(
      sheets.map((r) => [
        r,
        this.fixedLamp.has(r) ? regions(r, moving, this.fixedLamp.get(r)) : regions(r, lampVolumes),
      ]),
    );
    const { direct, K, solve, bytes } = this.sweep(modes);
    this.parallel ??= compileParallel(scene.rectangles, groups, modes, this.covers);
    const near = applyParallel(this.parallel, scene.roundedBoxes, K, modes.length * 2);
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
      for (let c = 0; c < 3; c++) direct[(2 * j + side) * 3 + c] = flux * r.color[c] * scene.lampRGB[c];
    }
    const result = solve();
    const outgoing = Float32Array.from(
      direct,
      (d, k) => d + result.incident[k] * modes[Math.floor(k / 6)].color[k % 3],
    );
    return {
      modes,
      groups,
      near,
      outgoing,
      proxy,
      covers: this.covers,
      lampRegions,
      lampVolumes,
      ...result,
      bytes: bytes + result.incident.byteLength + result.slopes.byteLength,
    };
  }
}
