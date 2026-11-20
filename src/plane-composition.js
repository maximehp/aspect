// Compose transport on an intermediate plane against a finite planar receiver.
// Every wall evaluation sees the entire receiving region. Both ends use the
// authored geometry; unresolved curved apertures use the native strip operator.
import { dot, cross, sub, unit, shadowBounds, canShadow } from './cpu-geometry.js';
import { projectedFootprint, shadowVolume, partitionReceiver, split, square } from './visibility.js';
import { conicImports } from './silhouette.js';
import { writeChart, writeReceiver } from './surface-functions.js';
import { ReceiverAperture } from './receiver-aperture.js';
import { AngularMeasure } from './angular-measure.js';

const compiled = WebAssembly.compileStreaming(fetch(new URL('./function-transport.wasm', import.meta.url)));
const norm = (v) => Math.hypot(...v);
const at = (chart, u, v) => chart.c.map((x, k) => x + chart.ax[k] * u + chart.v[k] * v);
const axes = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

function extent(shape, axis) {
  if (!shape.axes) return Math.abs(dot(shape.axisU, axis)) + Math.abs(dot(shape.axisV, axis));
  return (
    shape.radius * norm(axis) +
    shape.axes.reduce((s, a, i) => s + Math.abs(dot(a, axis)) * (shape.halfSize[i] - shape.radius), 0)
  );
}

// Projection of any connecting segment lies in the convex hull of the two
// projected intervals. A strictly disjoint blocker can be dropped for all w.
function possible(shapes, wall, target) {
  const delta = sub(target.center, wall.center),
    dirs = [
      ...axes,
      cross(wall.axisU, wall.axisV),
      cross(target.axisU, target.axisV),
      ...axes.map((k) => cross(delta, k)),
    ];
  return shapes.filter(
    (s) =>
      s !== wall &&
      s !== target.owner &&
      dirs.every((k) => {
        const end = dot(delta, k),
          r = extent(wall, k),
          t = extent(target, k),
          z = dot(sub(s.center, wall.center), k),
          d = extent(s, k);
        const margin = 1e-12 * (1 + Math.abs(z) + Math.abs(end) + r + t + d);
        return z + d >= Math.min(-r, end - t) - margin && z - d <= Math.max(r, end + t) + margin;
      }),
  );
}

function contour(eye, normal, target, blockers) {
  const center = sub(target.center, eye),
    h = dot(center, normal);
  if (dot(center, target.n) >= 0) return [];
  const [front] = split(square, dot(target.axisU, normal), dot(target.axisV, normal), h, 0, false);
  if (front.length < 3) return [];
  const footprints = [],
    curved = [];
  for (const s of blockers) {
    if (s.axes) {
      const bounds = shadowBounds(s, eye);
      if (!bounds || canShadow(target, bounds)) curved.push(s);
      continue;
    }
    const p = projectedFootprint(target, shadowVolume(s, eye));
    if (p === square) return [];
    if (p) footprints.push(p);
  }
  const lit = partitionReceiver(footprints, [{ uv: front, lit: true }]).filter((c) => c.lit);
  if (!lit.length) return [];
  if (curved.length) return null;
  return lit.map((cell) =>
    cell.uv.map(([u, v]) => center.map((x, k) => x + u * target.axisU[k] + v * target.axisV[k])),
  );
}

function chartFor(region, n, object) {
  return {
    type: 0,
    object,
    slot: 0,
    c: region.center,
    ax: unit(region.axisU),
    v: unit(region.axisV),
    u: [0, 0, 0],
    n,
    width: norm(region.axisU),
    height: norm(region.axisV),
    r: 0,
  };
}

class NativePlane {
  constructor(wasm, geometry, wall, emission, target, tolerance, packetRef) {
    this.was = wasm;
    this.tolerance = tolerance;
    this.wall = wall;
    this.geometry = geometry;
    this.ns = geometry.shapes.length;
    let end = Number(wasm.__heap_base);
    const alloc = (n) => {
      const a = end;
      end += Math.ceil(n / 8) * 8;
      return a;
    };
    const { packet } = geometry,
      ns = this.ns;
    this.src = alloc(packet.shapes.byteLength);
    this.lamp = alloc(24);
    this.ws = alloc(wasm.transportBytes(ns));
    this.wallPtr = alloc(168);
    this.target = alloc(168);
    this.emit = alloc(emission.length * 168);
    this.wallIds = alloc(ns * 4);
    this.catalog = alloc(emission.length * (ns + 1) * 4);
    this.dark = alloc((ns + 1) * 4);
    this.counts = alloc(160);
    this.cutCapacity = 32 + 32 * ns;
    this.cuts = alloc(this.cutCapacity * 8);
    this.ends = alloc(16 * (ns + 1));
    this.output = alloc(20 * Math.max(1, emission.length));
    this.ids = alloc(4 * Math.max(1, emission.length));
    this.points = end;
    this.reserve(1);
    const buf = wasm.memory.buffer;
    new Float64Array(buf, this.src, packet.shapes.length).set(packet.shapes);
    new Float64Array(buf, this.lamp, 3).set(packet.lamp);
    this.emission = emission;
    for (const [p, c] of [
      [this.wallPtr, wall],
      [this.target, target],
      ...emission.map((c, i) => [this.emit + 168 * i, c]),
    ])
      writeChart(buf, p, c);
    packetRef.packet = packet;
    this.reset();
    this.blockers = wasm.lampBlockers(this.ws, this.src, ns, this.wallPtr, this.wallIds);
    const cat = new Int32Array(buf, this.catalog, emission.length * (ns + 1));
    emission.forEach((c, i) => {
      const k = i * (ns + 1);
      cat[k] = wasm.lampBlockers(this.ws, this.src, ns, this.emit + 168 * i, this.catalog + 4 * (k + 1));
    });
    new Int32Array(buf, this.dark, 1)[0] = -1;
  }
  reserve(n) {
    const pages = Math.ceil((this.points + 56 * n - this.was.memory.buffer.byteLength) / 65536);
    if (pages > 0) this.was.memory.grow(pages);
  }
  reset() {
    this.was.prepareTransport(this.ws, this.src, this.ns, this.lamp);
  }
  events() {
    const n = this.was.lampEvents(
      this.ws,
      this.wallPtr,
      this.wallIds,
      this.blockers,
      this.cuts,
      this.cutCapacity,
    );
    if (n === this.cutCapacity) throw Error('Lamp boundary event capacity exhausted');
    return [...new Float64Array(this.was.memory.buffer, this.cuts, n)];
  }
  openings(t) {
    const n = this.was.lampIntervals(
      this.ws,
      this.src,
      this.wallPtr,
      this.lamp,
      this.wallIds,
      this.blockers,
      t,
      this.ends,
      this.counts,
    );
    return new Float64Array(this.was.memory.buffer, this.ends, 2 * n);
  }
  receive(points) {
    this.reserve(points.length);
    this.reset();
    const b = this.was.memory.buffer;
    points.forEach(({ p }, i) =>
      writeReceiver(b, this.points + 56 * i, { p, n: this.wall.n, object: this.wall.object }),
    );
  }
  row(i, emission) {
    const n = emission ? this.emission.length : 1,
      tile = emission ? this.emit : this.target;
    if (!n) return { rgb: [0, 0, 0], weight: 0, gpu: false };
    const count = this.was.transportRow(
      this.ws,
      this.src,
      this.ns,
      tile,
      n,
      this.points,
      i,
      this.lamp,
      this.tolerance,
      this.output,
      this.ids,
      this.counts,
      0,
      0,
      emission ? this.catalog : this.dark,
      0,
      0,
      0,
    );
    const b = this.was.memory.buffer,
      ids = new Int32Array(b, this.ids, count),
      values = new Float32Array(b, this.output, 5 * count);
    let rgb = [0, 0, 0],
      weight = 0,
      gpu = false;
    for (let j = 0; j < count; j++) {
      const k = ids[j] < 0 ? ~ids[j] : ids[j];
      if (emission) {
        const c = this.emission[k],
          color = this.geometry.shapes[c.object].color,
          jac = c.type ? 1 : norm(cross(c.ax, c.v));
        for (let a = 0; a < 3; a++) rgb[a] += jac * values[5 * j + 4] * color[a];
      } else {
        gpu = ids[j] < 0;
        for (let a = 0; a < 4; a++) weight += values[5 * j + a];
      }
    }
    return { rgb, weight, gpu };
  }
}

// Gauss points never coincide with a physical shadow edge. A lamp strip is
// split at the true interval ends before either incident function is used.
const rule = [
  [0.21132486540518713, 0.5],
  [0.7886751345948129, 0.5],
];
function integrateIntervals(lo, hi, span, visit) {
  const n = Math.max(1, Math.ceil((hi - lo) / span)),
    width = (hi - lo) / n;
  for (let j = 0; j < n; j++) for (const [x, w] of rule) visit(lo + (j + x) * width, width * w);
}
export class PlaneComposition {
  static async create(gl, geometry, wall, region, sourceObjects, { stripTolerance = 2e-5 } = {}) {
    if (wall?.type !== 0 || !region?.owner?.axisU || !region?.n)
      throw TypeError('The intermediate surface and receiver region must be planar charts');
    if (!Number.isFinite(stripTolerance) || stripTolerance <= 0)
      throw RangeError('The strip tolerance must be positive and finite');
    const ctx = {},
      w = new WebAssembly.Instance(await compiled, { math: Math, geometry: conicImports(() => ctx) }).exports;
    ctx.memory = w.memory;
    const target = chartFor(region, region.n, geometry.shapes.indexOf(region.owner));
    const selected = new Set(sourceObjects);
    const native = new NativePlane(
      w,
      geometry,
      wall,
      geometry.tiles.filter((c) => selected.has(geometry.shapes[c.object])),
      target,
      stripTolerance,
      ctx,
    );
    return new PlaneComposition(gl, native, region, target);
  }
  constructor(gl, native, region, target) {
    this.native = native;
    this.region = region;
    this.target = target;
    this.edges = new AngularMeasure(gl, { boundary: true });
    this.rect = new AngularMeasure(gl);
    this.blockers = possible(native.geometry.shapes, native.geometry.shapes[native.wall.object], region);
    this.family = new ReceiverAperture(native.wall, region, this.blockers);
  }
  dispose() {
    this.edges.dispose();
    this.rect.dispose();
  }
  solve({ relative = 0.005, absolute = 1e-9, initialPanels = 4, maxPanels = 64 } = {}) {
    if (
      !Number.isFinite(relative) ||
      relative < 0 ||
      !Number.isFinite(absolute) ||
      absolute <= 0 ||
      !Number.isSafeInteger(initialPanels) ||
      initialPanels < 1 ||
      !Number.isSafeInteger(maxPanels) ||
      maxPanels < initialPanels
    )
      throw RangeError('Invalid plane integral tolerances or panel limits');
    let previous = null,
      streak = 0;
    const estimates = [];
    for (let n = initialPanels; n <= maxPanels; n *= 2) {
      const sample = this.integrate(n);
      estimates.push(sample);
      const rgb = sample.directWallRGB.concat(sample.reflectedWallRGB);
      if (previous) {
        const met = rgb.every((x, i) => Math.abs(x - previous[i]) <= absolute + relative * Math.abs(x));
        streak = met ? streak + 1 : 0;
        if (streak === 2) return { converged: true, estimates };
      }
      previous = rgb;
    }
    return { converged: false, estimates };
  }
  integrate(panels) {
    if (!Number.isSafeInteger(panels) || panels < 1) throw RangeError('Expected a positive number of panels');
    const native = this.native,
      wall = native.wall,
      region = this.region,
      points = [];
    const t0 = performance.now();
    native.reset();
    const cuts = [-wall.height, ...native.events(), ...this.family.cuts, wall.height].sort((a, b) => a - b);
    let rows = 0,
      ends = 0,
      hiddenArea = 0,
      blockerTests = 0;
    const jac = norm(cross(wall.ax, wall.v));
    for (let z = 1; z < cuts.length; z++)
      if (cuts[z] > cuts[z - 1] + 1e-12)
        integrateIntervals(cuts[z - 1], cuts[z], (2 * wall.height) / panels, (t, wt) => {
          rows++;
          const visible = native.openings(t);
          ends += visible.length;
          const row = this.family.row(t),
            pieces = row.cuts.concat([...visible]).sort((a, b) => a - b);
          let j = 0;
          for (let k = 1; k < pieces.length; k++) {
            const a = Math.max(-wall.width, pieces[k - 1]),
              b = Math.min(wall.width, pieces[k]);
            if (b <= a) continue;
            const mid = (a + b) / 2,
              blockers = row.at(mid);
            if (!blockers) {
              hiddenArea += (b - a) * wt * jac;
              continue;
            }
            while (j < visible.length && visible[j + 1] < mid) j += 2;
            const lit = j < visible.length && visible[j] <= mid;
            integrateIntervals(a, b, (2 * wall.width) / panels, (u, wu) => {
              points.push({ p: at(wall, u, t), w: wt * wu * jac, lit, blockers });
              blockerTests += blockers.length;
            });
          }
        });
    const afterLamp = performance.now(),
      values = new Float64Array(points.length),
      indices = [],
      edges = [],
      jobs = [],
      fallback = [];
    const orientation = Math.sign(dot(cross(region.axisU, region.axisV), region.n)) * -1;
    for (let i = 0; i < points.length; i++) {
      const cells = contour(points[i].p, wall.n, region, points[i].blockers);
      if (cells === null) {
        fallback.push(i);
        continue;
      }
      for (const c of cells)
        for (let j = 0; j < c.length; j++) {
          indices.push(i);
          edges.push(c[j], c[(j + 1) % c.length]);
        }
    }
    const afterVisibility = performance.now();
    this.edges.reserve(indices.length);
    for (let j = 0; j < indices.length; j++)
      this.edges.putEdge(j, edges[2 * j], edges[2 * j + 1], wall.n, orientation);
    this.edges.integrate(indices.length);
    for (let j = 0; j < indices.length; j++) values[indices[j]] += this.edges.values[j];
    native.receive(points);
    for (const i of fallback) {
      const x = native.row(i, false);
      values[i] = x.weight * norm(cross(this.target.ax, this.target.v));
      if (x.gpu) jobs.push(i);
    }
    this.rect.reserve(jobs.length);
    for (let j = 0; j < jobs.length; j++) this.rect.put(j, this.target, { p: points[jobs[j]].p, n: wall.n });
    this.rect.integrate(jobs.length);
    for (let j = 0; j < jobs.length; j++) values[jobs[j]] += this.rect.values[j];
    const afterAperture = performance.now();
    // The emitting charts have their own source indices and generator caches.
    native.reset();
    const E = [0, 0, 0],
      B = [0, 0, 0],
      lamp = native.geometry.packet.lamp;
    let min = 1,
      max = 0,
      emittingQueries = 0;
    for (let i = 0; i < points.length; i++) {
      const aperture = values[i];
      min = Math.min(min, aperture);
      max = Math.max(max, aperture);
      if (aperture <= 0) continue;
      const { p, w, lit } = points[i],
        d = sub(lamp, p),
        l = norm(d),
        direct = lit ? Math.max(0, dot(d, wall.n)) / (l * l * l) : 0;
      const emission = native.row(i, true).rgb;
      emittingQueries++;
      for (let c = 0; c < 3; c++) {
        E[c] += w * aperture * direct;
        B[c] += w * aperture * emission[c];
      }
    }
    const end = performance.now(),
      area = 4 * norm(cross(region.axisU, region.axisV));
    const factor = region.owner.color.map(
      (r, c) => (r * native.geometry.shapes[wall.object].color[c] * native.geometry.packet.lampRGB[c]) / area,
    );
    return {
      panels,
      hiddenArea,
      blockerTests,
      wholeBundleBlockers: this.family.records.length,
      wallRows: rows,
      wallPoints: points.length,
      lampIntervalEnds: ends,
      edges: indices.length,
      curvedFallback: fallback.length,
      rectangularJobs: jobs.length,
      emittingQueries,
      minAperture: min,
      maxAperture: max,
      directWallRGB: E.map((x, c) => x * factor[c]),
      reflectedWallRGB: B.map((x, c) => x * factor[c]),
      ms: {
        lamp: afterLamp - t0,
        visibility: afterVisibility - afterLamp,
        apertures: afterAperture - afterVisibility,
        incident: end - afterAperture,
        total: end - t0,
      },
    };
  }
}
