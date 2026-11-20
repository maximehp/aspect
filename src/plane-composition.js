// Compose transport on an intermediate plane against a finite planar receiver.
// Every wall evaluation sees the entire receiving region. Both ends use the
// authored geometry; unresolved curved apertures use the native strip operator.
import { dot, cross, sub, unit, shadowBounds, canShadow } from './cpu-geometry.js';
import { projectedFootprint, shadowVolume, partitionReceiver, split, square } from './visibility.js';
import { conicImports } from './silhouette.js';
import { writeChart, writeReceiver } from './surface-functions.js';
import { ReceiverAperture, intersect, extent } from './receiver-aperture.js';
import {
  generatorSteps,
  weightedApertureVertex,
  composedApertureVertex,
  apertureBands,
  boundedBand,
} from './weighted-aperture.js';
import { planarObserver } from './parallel-observer.js';
import { AngularMeasure } from './angular-measure.js';

const compiled = WebAssembly.compileStreaming(fetch(new URL('./function-transport.wasm', import.meta.url)));
const norm = (v) => Math.hypot(...v);
const at = (chart, u, v) => chart.c.map((x, k) => x + chart.ax[k] * u + chart.v[k] * v);
const axes = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];

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
  return lit.map((cell) => cell.uv);
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
    const tileId = new Map(geometry.tiles.map((t, i) => [t, i]));
    const raw = packet.lampEvents;
    const records = emission.map((t) => {
      const k = raw?.[tileId.get(t)];
      return k ? raw.subarray(k, k + 2 + raw[k] + 4 * raw[k + 1]) : null;
    });
    const events = new Float64Array(emission.length + records.reduce((s, f) => s + (f?.length || 0), 0));
    let eventEnd = emission.length;
    records.forEach((f, k) => {
      if (f) {
        events[k] = eventEnd;
        events.set(f, eventEnd);
        eventEnd += f.length;
      }
    });
    this.eventsPtr = alloc(events.byteLength);
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
    new Float64Array(buf, this.eventsPtr, events.length).set(events);
    new Float64Array(buf, this.src, packet.shapes.length).set(packet.shapes);
    new Float64Array(buf, this.lamp, 3).set(packet.lamp);
    this.emission = emission;
    for (const [p, c] of [
      [this.wallPtr, wall],
      [this.target, target],
      ...emission.map((c, i) => [this.emit + 168 * i, c]),
    ])
      writeChart(buf, p, c);
    this.context = packetRef;
    packetRef.packet = packet;
    this.reset();
    this.blockers = wasm.lampBlockers(this.ws, this.src, ns, this.wallPtr, this.wallIds);
    const cat = new Int32Array(buf, this.catalog, emission.length * (ns + 1));
    emission.forEach((c, i) => {
      const k = i * (ns + 1);
      if (packet.lampIds) {
        const j = tileId.get(c) * (ns + 1);
        cat.set(packet.lampIds.subarray(j, j + ns + 1), k);
      } else
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
  // The ray origin is a geometric input to both the fast cones and the
  // high-precision conic sign. Rebind both together for each new eye.
  lookFrom(p) {
    const { shapes, definitions } = this.geometry.packet;
    this.context.packet = { shapes, definitions, lamp: p };
    new Float64Array(this.was.memory.buffer, this.lamp, 3).set(p);
    this.reset();
    this.blockers = this.was.lampBlockers(this.ws, this.src, this.ns, this.wallPtr, this.wallIds);
    return this.events().map((v) => v / this.wall.height);
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
  receive(points, surface = this.wall) {
    this.reserve(points.length);
    this.reset();
    const b = this.was.memory.buffer;
    points.forEach(({ p }, i) =>
      writeReceiver(b, this.points + 56 * i, { p, n: surface.n, object: surface.object }),
    );
  }
  row(i, emission) {
    const n = emission ? this.emission.length : 1,
      tile = emission ? this.emit : this.target;
    if (!n) return { rgb: [0, 0, 0], weights: [0, 0, 0, 0], gpu: false };
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
      emission ? this.eventsPtr : 0,
      0,
    );
    const b = this.was.memory.buffer,
      ids = new Int32Array(b, this.ids, count),
      values = new Float32Array(b, this.output, 5 * count);
    let rgb = [0, 0, 0],
      weights = [0, 0, 0, 0],
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
        for (let a = 0; a < 4; a++) weights[a] += values[5 * j + a];
      }
    }
    return { rgb, weights, gpu };
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
// Bounded staging space independent of the number of clipped intervals.
// When a row needs refinement its new estimate replaces the complete row.
class BandBatch {
  constructor(gpu, output, consume) {
    this.consume = consume;
    this.gpu = gpu;
    this.output = output;
    this.rows = new Uint32Array(8192);
    this.count = 0;
    this.jobs = 0;
    this.ms = 0;
  }
  put(row, origin, tilt, lower, upper) {
    if (this.count === this.rows.length) this.flush();
    this.gpu.reserve(this.rows.length);
    const k = 12 * this.count,
      p = this.gpu.packed;
    p.set(origin, k);
    p.set(tilt, k + 3);
    p.set(lower, k + 6);
    p.set(upper, k + 9);
    this.rows[this.count++] = row;
    this.jobs++;
  }
  flush() {
    if (!this.count) return;
    const t = performance.now(),
      { gpu } = this,
      channels = gpu.channels;
    if (this.consume) this.consume(this.rows, this.count);
    else {
      gpu.integrate(this.count);
      for (let i = 0; i < this.count; i++)
        for (let k = 0; k < channels; k++) {
          const x = gpu.values[i * channels + k];
          this.output[this.rows[i] * channels + k] += x >= 0 && Number.isFinite(x) ? x : NaN;
        }
    }
    this.ms += performance.now() - t;
    this.count = 0;
  }
}
export class PlaneComposition {
  static async create(gl, geometry, wall, region, sourceObjects, { stripTolerance = 2e-5 } = {}) {
    if (wall?.type !== 0 || !region?.owner?.axisU || !region?.n)
      throw TypeError('The intermediate surface and receiver region must be planar charts');
    if (!Number.isFinite(stripTolerance) || stripTolerance <= 0)
      throw RangeError('The strip tolerance must be positive and finite');
    const module = await compiled;
    function createNative(source, emit, destination) {
      const ctx = {},
        w = new WebAssembly.Instance(module, { math: Math, geometry: conicImports(() => ctx) }).exports;
      ctx.memory = w.memory;
      return new NativePlane(w, geometry, source, emit, destination, stripTolerance, ctx);
    }
    const target = chartFor(region, region.n, geometry.shapes.indexOf(region.owner));
    const selected = new Set(sourceObjects);
    const ids = geometry.packet.lampIds,
      stride = geometry.shapes.length + 1;
    let excluded = 0;
    const emit = geometry.tiles.filter((c, k) => {
      if (!selected.has(geometry.shapes[c.object])) return false;
      if (ids?.[k * stride] < 0) {
        excluded++;
        return false;
      }
      return true;
    });
    const native = createNative(wall, emit, target);
    native.excluded = excluded;
    return new PlaneComposition(gl, native, region, target, () => createNative(target, [], target));
  }
  constructor(gl, native, region, target, createEye) {
    this.createEye = createEye;
    this.native = native;
    this.region = region;
    this.target = target;
    this.bands = new AngularMeasure(gl, { apertureVertex: weightedApertureVertex, components: 4 });
    const along = target.ax,
      normal = unit(cross(target.ax, target.v)),
      across = cross(normal, along);
    this.axes = [along, across, normal];
    this.extent = [target.width, dot(region.axisV, along), dot(region.axisV, across)];
    gl.useProgram(this.bands.program);
    gl.uniform3fv(gl.getUniformLocation(this.bands.program, 'extent'), this.extent);
    this.rect = new AngularMeasure(gl);
    this.blockers = possible(native.geometry.shapes, native.geometry.shapes[native.wall.object], region);
    this.family = new ReceiverAperture(native.wall, region, this.blockers);
  }
  dispose() {
    this.joint?.dispose();
    this.bands.dispose();
    this.rect.dispose();
  }
  // Compile a function only after the entire floor-to-observer bundle is clear.
  // The same certification applies to every underlying path feeding this floor.
  observerRegion(target, tolerance, queries) {
    const floor = this.region,
      shapes = this.native.geometry.shapes;
    const object = shapes.indexOf(target.owner);
    if (object < 0) throw Error('Observer must belong to the current scene');
    const family = new ReceiverAperture(
      this.target,
      target,
      possible(
        shapes.filter((s) => s !== floor.owner),
        floor,
        target,
      ),
    );
    if (family.records.length) return null;
    const field = planarObserver(floor, target, tolerance, queries);
    if (!field) return null;
    const observers = field.nodes.map((p) => ({ p, n: target.n, object }));
    this.clearQueries ??= new WeakMap();
    this.clearQueries.set(observers, field.minimumDistance);
    return {
      orders: field.orders,
      relativeBound: field.relativeBound,
      solve: (panels) => {
        const appearance = this.integrate(panels, observers).composedObservers;
        if (appearance.some((s) => !s.complete)) return { complete: false };
        const samples = appearance.map((s) => [...s.directWallRGB, ...s.reflectedWallRGB]);
        return {
          complete: true,
          at: (u, v) => field.at(samples, u, v),
          moments: field.moments(samples),
          sharedBands: appearance[0].sharedBands || 0,
        };
      },
    };
  }
  receiverWeights(next) {
    const native = this.native,
      g = this.target;
    native.receive([next], next);
    const { weights, gpu } = native.row(0, false);
    const jac = norm(cross(g.ax, g.v));
    for (let j = 0; j < 4; j++) weights[j] *= jac;
    if (gpu) {
      this.rect.reserve(1);
      this.rect.put(0, g, next);
      this.rect.integrate(1);
      weights[0] += this.rect.values[0];
    }
    return weights;
  }
  // For a verified planar observer domain, the separation bounds the second
  // distance at every integration point. The incoming visibility and band
  // geometry can be generated once for all samples of the output function.
  composeBundle(points, observers, gap) {
    if (points.some((p) => p.emission && !p.cells)) return null;
    const { native, region, axes, extent } = this,
      { geometry, wall } = native,
      gl = this.bands.gl;
    this.joint ??= new AngularMeasure(gl, { apertureVertex: composedApertureVertex });
    const gpu = this.joint;
    const set = (name, v) => gl.uniform3fv(gl.getUniformLocation(gpu.program, name), v);
    gl.useProgram(gpu.program);
    set('extent', extent);
    set('halfSteps', generatorSteps(native.tolerance * 0.4));
    const frame = (p, n) => {
      const d = sub(region.center, p);
      return [
        [dot(d, axes[0]), dot(d, axes[1]), Math.abs(dot(d, axes[2]))],
        [dot(n, axes[0]), dot(n, region.axisV), dot(n, d)],
      ];
    };
    const dst = observers.map((o) => frame(o.p, o.n)),
      sums = observers.map(() => new Float64Array(4));
    let ok = true,
      finite = true;
    const batch = new BandBatch(gpu, null, (rows, count) => {
      for (let j = 0; j < dst.length; j++) {
        const [origin, tilt] = dst[j],
          total = sums[j];
        set('nextDisplacement', origin);
        set('nextTilt', tilt);
        gpu.integrate(count);
        for (let i = 0; i < count; i++) {
          const x = points[rows[i]],
            w = x.w * gpu.values[i];
          if (!(w >= 0 && Number.isFinite(w))) {
            finite = false;
            return;
          }
          total[0] += w * x.direct;
          for (let c = 0; c < 3; c++) total[c + 1] += w * x.emission[c];
        }
      }
    });
    for (let i = 0; i < points.length && ok && finite; i++) {
      const x = points[i];
      if (!x.emission || (!x.direct && x.emission.every((v) => v === 0))) continue;
      const [origin, tilt] = frame(x.p, wall.n);
      for (const cell of x.cells)
        apertureBands(cell, (l, h) => {
          if (ok)
            ok = boundedBand(
              l,
              h,
              origin,
              extent,
              native.tolerance * 0.4,
              (a, b) => batch.put(i, origin, tilt, a, b),
              gap,
            );
        });
    }
    if (!ok || !finite) return null;
    batch.flush();
    if (!ok || !finite) return null;
    const gain = region.owner.color.map(
      (r, c) =>
        r *
        geometry.shapes[wall.object].color[c] *
        geometry.packet.lampRGB[c] *
        geometry.shapes[observers[0].object].color[c],
    );
    return sums.map((s) => ({
      complete: true,
      sharedBands: batch.jobs,
      directWallRGB: gain.map((g) => g * s[0]),
      reflectedWallRGB: gain.map((g, c) => g * s[c + 1]),
    }));
  }
  compose(points, next) {
    const started = performance.now(),
      { region, native, axes, extent } = this,
      { geometry, wall } = native;
    const obstacles = possible(geometry.shapes, geometry.shapes[next.object], region);
    const fixed = contour(next.p, next.n, region, obstacles);
    if (fixed && !fixed.length)
      return {
        complete: true,
        directWallRGB: [0, 0, 0],
        reflectedWallRGB: [0, 0, 0],
        bands: 0,
        darkPoints: points.length,
        curved: 0,
        lineJobs: 0,
        limited: 0,
        largestDelta: 0,
        ms: performance.now() - started,
      };

    const displacement = (p) => {
      const d = sub(region.center, p);
      return [dot(d, axes[0]), dot(d, axes[1]), Math.abs(dot(d, axes[2]))];
    };
    const tilt = (p, n) => [dot(n, axes[0]), dot(n, region.axisV), dot(n, sub(region.center, p))];
    const q = displacement(next.p),
      aperture = new Float64Array(points.length);
    const gl = this.bands.gl;
    this.joint ??= new AngularMeasure(gl, { apertureVertex: composedApertureVertex });
    const gpu = this.joint;
    gl.useProgram(gpu.program);
    for (const [name, v] of [
      ['nextDisplacement', q],
      ['nextTilt', tilt(next.p, next.n)],
      ['extent', extent],
      ['halfSteps', generatorSteps(native.tolerance * 0.4)],
    ])
      gl.uniform3fv(gl.getUniformLocation(gpu.program, name), v);
    const curved = [];
    let darkPoints = 0;
    const linear = new BandBatch(gpu, aperture);
    for (let i = 0; i < points.length; i++) {
      const x = points[i];
      if (!x.emission || (!x.direct && x.emission.every((v) => v === 0))) {
        darkPoints++;
        continue;
      }
      if (!x.cells || !fixed) {
        curved.push(i);
        continue;
      }
      let ok = true;
      const origin = displacement(x.p),
        slope = tilt(x.p, wall.n);
      for (const a of x.cells)
        for (const b of fixed) {
          const visible = b === square ? a : intersect(a, b);
          apertureBands(visible, (lo, hi) => {
            if (!ok) return;
            ok = boundedBand(
              lo,
              hi,
              origin,
              extent,
              native.tolerance * 0.4,
              (l, h) => {
                linear.put(i, origin, slope, l, h);
              },
              q,
            );
          });
        }
      if (!ok) curved.push(i);
    }
    linear.flush();
    for (let i = 0; i < aperture.length; i++)
      if (!Number.isFinite(aperture[i]) && !curved.includes(i)) curved.push(i);
    let lineJobs = 0,
      limited = 0,
      largestDelta = 0;
    if (curved.length) {
      this.apertureEyes ??= [this.createEye(), this.createEye()];
      const [moving, stationary] = this.apertureEyes;
      const horizon = (p, n) => [dot(n, region.axisU), dot(n, region.axisV), dot(n, sub(region.center, p))];
      const fixedH = horizon(next.p, next.n);
      const fixedCuts = stationary.lookFrom(next.p);
      const visible = (eye, v, h) => {
        const raw = eye.openings(v * this.target.height),
          intervals = [];
        const g = h[1] * v + h[2],
          a = h[0];
        for (let j = 0; j < raw.length; j += 2) {
          let l = raw[j] / this.target.width,
            r = raw[j + 1] / this.target.width;
          if (!a && g <= 0) continue;
          if (a > 0) l = Math.max(l, -g / a);
          if (a < 0) r = Math.min(r, -g / a);
          if (l < r) intervals.push([l, r]);
        }
        return intervals;
      };
      let remaining = curved,
        previous = null;
      for (let n = 4; n <= 2048 && remaining.length; n *= 2) {
        const estimate = new Float64Array(points.length),
          batch = new BandBatch(gpu, estimate);
        for (const i of remaining) {
          const x = points[i],
            origin = displacement(x.p),
            slope = tilt(x.p, wall.n);
          const h = horizon(x.p, wall.n),
            cuts = [-1, ...fixedCuts, ...moving.lookFrom(x.p), 1];
          for (const g of [h, fixedH])
            if (g[1])
              for (const u of [-1, 1]) {
                const v = -(g[2] + g[0] * u) / g[1];
                if (v > -1 && v < 1) cuts.push(v);
              }
          cuts.sort((a, b) => a - b);
          for (let j = 1; j < cuts.length; j++)
            if (cuts[j] > cuts[j - 1]) {
              integrateIntervals(cuts[j - 1], cuts[j], 2 / n, (v, w) => {
                const a = visible(moving, v, h),
                  b = visible(stationary, v, fixedH);
                for (const [l, r] of a)
                  for (const [L, R] of b) {
                    const lo = Math.max(l, L),
                      hi = Math.min(r, R);
                    if (hi > lo) {
                      batch.put(i, origin, slope, [lo, hi, v], [w, 0, -3]);
                    }
                  }
              });
            }
        }
        batch.flush();
        lineJobs += batch.jobs;
        if (remaining.some((i) => !Number.isFinite(estimate[i])))
          return { complete: false, reason: 'Angular precision or work limit' };
        if (previous)
          remaining = remaining.filter((i) => {
            const delta = Math.abs(estimate[i] - previous[i]);
            if (delta <= 1e-10 + native.tolerance * estimate[i] || n === 2048) {
              aperture[i] = estimate[i];
              largestDelta = Math.max(largestDelta, delta);
              if (delta > 1e-10 + native.tolerance * estimate[i]) limited++;
              return false;
            }
            return true;
          });
        previous = estimate;
      }
    }
    const direct = [0, 0, 0],
      reflected = [0, 0, 0];
    const gain = region.owner.color.map(
      (r, c) =>
        r *
        geometry.shapes[wall.object].color[c] *
        geometry.packet.lampRGB[c] *
        geometry.shapes[next.object].color[c],
    );
    for (let i = 0; i < points.length; i++) {
      if (!aperture[i]) continue;
      const p = points[i],
        value = aperture[i];
      for (let c = 0; c < 3; c++) {
        direct[c] += gain[c] * p.w * p.direct * value;
        reflected[c] += gain[c] * p.w * p.emission[c] * value;
      }
    }
    return {
      complete: !limited,
      directWallRGB: direct,
      reflectedWallRGB: reflected,
      bands: linear.jobs,
      darkPoints,
      curved: curved.length,
      lineJobs,
      limited,
      largestDelta,
      ms: performance.now() - started,
    };
  }
  solve({ relative = 0.005, absolute = 1e-9, initialPanels = 4, maxPanels = 64 } = {}, observers = []) {
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
      const sample = this.integrate(n, observers);
      estimates.push(sample);
      const rgb = sample.directMoments.flat().concat(sample.reflectedMoments.flat());
      if (observers.length)
        for (const entry of sample.composedObservers)
          if (entry.complete) rgb.push(...entry.directWallRGB, ...entry.reflectedWallRGB);
      if (sample.composedObservers.some((x) => !x.complete)) {
        previous = null;
        streak = 0;
        continue;
      }
      if (previous && rgb.length === previous.length) {
        const met = rgb.every((x, i) => Math.abs(x - previous[i]) <= absolute + relative * Math.abs(x));
        streak = met ? streak + 1 : 0;
        if (streak === 2) return { converged: true, estimates };
      }
      previous = rgb;
    }
    return { converged: false, estimates };
  }
  integrate(panels, observers = []) {
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
      values = new Float64Array(4 * points.length),
      batch = new BandBatch(this.bands, values),
      jobs = [],
      fallback = new Set();
    const tilt = [dot(wall.n, this.axes[0]), dot(wall.n, region.axisV)];
    let bandCount = 0;
    for (let i = 0; i < points.length; i++) {
      const cells = contour(points[i].p, wall.n, region, points[i].blockers);
      if (observers.length) points[i].cells = cells;
      if (!cells) {
        fallback.add(i);
        continue;
      }
      const delta = sub(region.center, points[i].p),
        displacement = this.axes.map((a) => dot(a, delta));
      displacement[2] = Math.abs(displacement[2]);
      const cosine = dot(wall.n, delta),
        slope = [...tilt, cosine];
      const save = (lo, hi) => {
        batch.put(i, displacement, slope, lo, hi);
      };
      let ok = true;
      for (const cell of cells)
        apertureBands(cell, (lo, hi) => {
          bandCount++;
          if (ok) ok = boundedBand(lo, hi, displacement, this.extent, native.tolerance, save);
        });
      if (!ok) {
        fallback.add(i);
      }
    }
    batch.flush();
    const afterVisibility = performance.now() - batch.ms;
    for (let i = 0; i < points.length; i++)
      for (let k = 0; k < 4; k++) if (!(values[4 * i + k] >= 0 && values[4 * i + k] <= 1)) fallback.add(i);
    native.receive(points);
    const jacobian = norm(cross(this.target.ax, this.target.v));
    for (const i of fallback) {
      const x = native.row(i, false);
      for (let k = 0; k < 4; k++) values[4 * i + k] = x.weights[k] * jacobian;
      if (x.gpu) jobs.push(i);
    }
    this.rect.reserve(jobs.length);
    for (let j = 0; j < jobs.length; j++) this.rect.put(j, this.target, { p: points[jobs[j]].p, n: wall.n });
    this.rect.integrate(jobs.length);
    for (let j = 0; j < jobs.length; j++) values[4 * jobs[j]] += this.rect.values[j];
    const afterAperture = performance.now();
    // The emitting charts have their own source indices and generator caches.
    native.reset();
    const E = Array.from({ length: 4 }, () => [0, 0, 0]),
      B = Array.from({ length: 4 }, () => [0, 0, 0]),
      lamp = native.geometry.packet.lamp;
    let min = 1,
      max = 0,
      emittingQueries = 0;
    for (let i = 0; i < points.length; i++) {
      const aperture = values[4 * i] + values[4 * i + 1] + values[4 * i + 2] + values[4 * i + 3];
      min = Math.min(min, aperture);
      max = Math.max(max, aperture);
      if (aperture <= 0) continue;
      const { p, w, lit } = points[i],
        d = sub(lamp, p),
        l = norm(d),
        direct = lit ? Math.max(0, dot(d, wall.n)) / (l * l * l) : 0;
      const emission = native.row(i, true).rgb;
      if (observers.length) {
        points[i].direct = direct;
        points[i].emission = emission;
      }
      emittingQueries++;
      for (let k = 0; k < 4; k++)
        for (let c = 0; c < 3; c++) {
          E[k][c] += w * values[4 * i + k] * direct;
          B[k][c] += w * values[4 * i + k] * emission[c];
        }
    }
    const gap = this.clearQueries?.get(observers);
    const composedObservers =
      (gap && this.composeBundle(points, observers, gap)) ||
      observers.map((next) => this.compose(points, next));
    const end = performance.now(),
      area = 4 * norm(cross(region.axisU, region.axisV));
    const factor = region.owner.color.map(
      (r, c) => (r * native.geometry.shapes[wall.object].color[c] * native.geometry.packet.lampRGB[c]) / area,
    );
    const scale = (row) => row.map((x, c) => x * factor[c]);
    const directMoments = E.map(scale),
      reflectedMoments = B.map(scale);
    const total = (m) => m.reduce((a, b) => a.map((x, c) => x + b[c]), [0, 0, 0]);
    return {
      composedObservers,
      panels,
      hiddenArea,
      blockerTests,
      wholeBundleBlockers: this.family.records.length,
      wallRows: rows,
      wallPoints: points.length,
      lampIntervalEnds: ends,
      visibleBands: bandCount,
      boundedBands: batch.jobs,
      nativeApertures: fallback.size,
      rectangularJobs: jobs.length,
      emittingQueries,
      darkSourceCharts: native.excluded,
      emittingCharts: native.emission.length,
      minAperture: min,
      maxAperture: max,
      directMoments,
      reflectedMoments,
      directWallRGB: total(directMoments),
      reflectedWallRGB: total(reflectedMoments),
      ms: {
        lamp: afterLamp - t0,
        visibility: afterVisibility - afterLamp,
        apertures: afterAperture - afterVisibility,
        incidentAndComposition: end - afterAperture,
        total: end - t0,
      },
    };
  }
}
