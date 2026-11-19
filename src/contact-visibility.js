// Compile immutable receiver/source geometry once per chart. A moving frame
// supplies the solid transforms and the shape indices in each surviving bundle.
import { shadowAzimuths } from './cpu-geometry.js';
import { contactTextureWidth } from './gpu-contact.js';
import { square } from './visibility.js';

const angles = Float64Array.from(shadowAzimuths.flat());

// Each block is aligned for double-precision coordinates. Static chart blocks
// precede the shared high-water scratch space in the module's linear memory.
function block(start = 0) {
  return {
    end: start,
    reserve(bytes) {
      const at = this.end;
      this.end += Math.ceil(bytes / 8) * 8;
      return at;
    },
  };
}

function compileChart(sites, kernel, base) {
  const groups = new Map(),
    rects = new Map(),
    polygons = new Map();
  let count = 0,
    pieces = 0,
    vertices = 0;
  for (const site of sites)
    for (const sample of site.samples) {
      count++;
      if (!groups.has(sample.bundle)) groups.set(sample.bundle, groups.size);
      for (const rect of [sample.m, sample.region]) if (!rects.has(rect)) rects.set(rect, rects.size);
      for (const { uv } of sample.lit) {
        pieces++;
        if (uv !== square && !polygons.has(uv)) {
          polygons.set(uv, vertices);
          vertices += uv.length;
        }
      }
    }
  const space = block(),
    eyes = space.reserve(sites.length * 24),
    rectangles = space.reserve(rects.size * 72),
    addresses = space.reserve(count * 32),
    ranges = space.reserve(pieces * 8),
    fixed = space.reserve(vertices * 16);
  ensure(kernel.memory, base + space.end);
  const buffer = kernel.memory.buffer;
  const coordinates = new Float64Array(buffer, base + eyes, sites.length * 3);
  const planes = new Float64Array(buffer, base + rectangles, rects.size * 9);
  const ints = new Int32Array(buffer, base + addresses, count * 8);
  const spans = new Int32Array(buffer, base + ranges, pieces * 2),
    points = new Float64Array(buffer, base + fixed, vertices * 2);
  for (const [rect, i] of rects) {
    planes.set(rect.center, i * 9);
    planes.set(rect.axisU, i * 9 + 3);
    planes.set(rect.axisV, i * 9 + 6);
  }
  for (const [poly, start] of polygons) {
    for (let j = 0; j < poly.length; j++) points.set(poly[j], 2 * (start + j));
  }
  let row = 0,
    piece = 0;
  for (const [i, site] of sites.entries()) {
    coordinates.set(site.p, i * 3);
    const cell = site.cell !== undefined;
    for (const sample of site.samples) {
      const first = piece;
      for (const { uv } of sample.lit) {
        spans.set([uv === square ? -1 : polygons.get(uv), uv.length], 2 * piece++);
      }
      ints.set(
        [
          i,
          groups.get(sample.bundle),
          first,
          piece,
          rects.get(sample.m),
          rects.get(sample.region),
          Number(cell),
          sample.coefficientRow,
        ],
        row * 8,
      );
      row++;
    }
  }
  return {
    base,
    bytes: space.end,
    eyes,
    rectangles,
    addresses,
    ranges,
    fixed,
    count,
    groups: [...groups.keys()],
  };
}

function ensure(memory, bytes) {
  if (memory.buffer.byteLength < bytes) memory.grow(Math.ceil((bytes - memory.buffer.byteLength) / 65536));
}

export class ContactVisibility {
  constructor(module) {
    this.kernel = new WebAssembly.Instance(module).exports;
    this.layouts = new Map();
    this.directions = Number(this.kernel.__heap_base);
    this.firstChart = this.directions + angles.byteLength;
    this.end = this.firstChart;
    ensure(this.kernel.memory, this.firstChart);
    new Float64Array(this.kernel.memory.buffer, this.directions, angles.length).set(angles);
    this.scratchBytes = 16384;
    this.jobCapacity = 256;
    this.vertexCapacity = 2048;
  }
  retain(charts) {
    const active = new Set();
    for (const contact of charts.values()) active.add(contact.stationary);
    const memory = new Uint8Array(this.kernel.memory.buffer);
    let end = this.firstChart;
    for (const [sites, chart] of this.layouts) {
      if (!active.has(sites)) {
        this.layouts.delete(sites);
        continue;
      }
      if (chart.base !== end) {
        memory.copyWithin(end, chart.base, chart.base + chart.bytes);
        chart.base = end;
      }
      end += chart.bytes;
    }
    this.end = end;
  }
  packets(sites, moving) {
    let chart = this.layouts.get(sites);
    if (!chart) {
      chart = compileChart(sites, this.kernel, this.end);
      this.end += chart.bytes;
      this.layouts.set(sites, chart);
    }
    const { groups, count } = chart;
    const candidates = groups.reduce((n, group) => n + group.candidates.length, 0);
    const kernel = this.kernel,
      memory = kernel.memory;
    const space = block(this.end);
    const shapes = space.reserve(moving.length * 16 * 8),
      groupsAt = space.reserve((groups.length + 1) * 4),
      indices = space.reserve(candidates * 4),
      cones = space.reserve(kernel.coneBytes(shadowAzimuths.length) * moving.length),
      counts = space.reserve(16),
      cached = space.reserve(count * 4),
      workspace = space.end;
    let jobs, points;
    ensure(memory, workspace);
    const packed = new Float64Array(memory.buffer, shapes, moving.length * 16);
    for (const [i, s] of moving.entries()) {
      const base = i * 16;
      packed.set(s.center, base);
      for (let k = 0; k < 3; k++) packed.set(s.axes[k], base + 3 + 3 * k);
      packed.set(s.halfSize, base + 12);
      packed[base + 15] = s.radius;
    }
    const offsets = new Int32Array(memory.buffer, groupsAt, groups.length + 1);
    const members = new Int32Array(memory.buffer, indices, candidates);
    let at = 0;
    for (const [i, group] of groups.entries()) {
      offsets[i] = at;
      members.set(group.candidates, at);
      at += group.candidates.length;
    }
    offsets[groups.length] = at;
    for (;;) {
      space.end = workspace;
      const scratch = space.reserve(this.scratchBytes);
      jobs = space.reserve(this.jobCapacity * 16);
      points = space.reserve(this.vertexCapacity * 8);
      ensure(memory, space.end);
      const status = kernel.contactPackets(
        chart.base + chart.eyes,
        chart.base + chart.rectangles,
        chart.base + chart.addresses,
        count,
        shapes,
        moving.length,
        chart.base + chart.ranges,
        chart.base + chart.fixed,
        groupsAt,
        indices,
        this.directions,
        shadowAzimuths.length,
        cones,
        scratch,
        this.scratchBytes,
        cached,
        jobs,
        this.jobCapacity,
        points,
        this.vertexCapacity,
        counts,
      );
      if (!status) break;
      if (status === -1) this.scratchBytes *= 2;
      else if (status === -2) this.jobCapacity *= 2;
      else if (status === -3) this.vertexCapacity *= 2;
      else throw Error('Invalid contact visibility result.');
    }
    // A frame may retain these packets while the arena prepares another
    // chart or pose. Only copy the live prefix, padding the texture's last row.
    const buffer = memory.buffer;
    const [cachedCount, changedCount, vertices, maxVertices] = new Int32Array(buffer, counts, 4);
    const stride = 4 * contactTextureWidth;
    const positions = new Float32Array(stride * Math.max(1, Math.ceil((vertices * 2) / stride)));
    positions.set(new Float32Array(buffer, points, vertices * 2));
    return {
      cachedJobs: new Float32Array(buffer, cached, cachedCount).slice(),
      changed: { positions, jobs: new Float32Array(buffer, jobs, changedCount * 4).slice(), maxVertices },
    };
  }
}
