import { incidentDirections } from './contact-transfer.js';
import { cross, unit, point } from './cpu-geometry.js';

export const directionCount = 96;
const stripCount = 192,
  frames = [];
const corners = [
  [-1, -1],
  [1, -1],
  [1, 1],
  [-1, 1],
];
for (let i = 0; i < directionCount; i++) {
  const z = (i + 0.5) / directionCount,
    angle = i * 2.399963229728653,
    r = Math.sqrt(1 - z * z);
  const w = [r * Math.cos(angle), r * Math.sin(angle), z],
    u = unit(cross(w, [0, 0, 1]));
  frames.push(...u, ...cross(w, u), ...w);
}
function pack(modes) {
  const count = modes.length,
    packet = new Uint8Array(count * 120),
    ids = new Int32Array(packet.buffer),
    values = new Float32Array(packet.buffer),
    owners = new Map();
  for (let j = 0; j < count; j++) {
    const r = modes[j],
      offset = j * 18;
    if (!owners.has(r.owner)) owners.set(r.owner, owners.size);
    ids.set([4 * j, 4, j, owners.get(r.owner), Number(!!r.owner.axes)], offset);
    values.set([r.area, ...r.normal, ...r.center, ...r.color, ...r.momentAxis], offset + 5);
    let dst = 18 * count + j * 12;
    for (const [u, v] of corners) {
      values.set(point(r, u, v), dst);
      dst += 3;
    }
  }
  return packet;
}
// Rewindable linear-memory arena. Only observed side pairs get link records.
export class BeamWorkspace {
  constructor(exports) {
    this.wasm = exports;
    this.capacity = 1024;
  }
  reserve(bytes) {
    this.cursor = Math.ceil(this.cursor / 8) * 8;
    const p = this.cursor;
    this.cursor += bytes;
    return p;
  }
  grow() {
    const memory = this.wasm.memory;
    if (this.cursor > memory.buffer.byteLength)
      memory.grow(Math.ceil((this.cursor - memory.buffer.byteLength) / 65536));
    this.bytes = new Uint8Array(memory.buffer);
    this.ints = new Int32Array(memory.buffer);
    this.floats = new Float32Array(memory.buffer);
  }
  start(packet) {
    const np = packet.length / 120;
    this.cursor = Number(this.wasm.__heap_base);
    this.records = this.reserve(packet.length);
    this.vertices = this.records + 72 * np;
    this.axes = this.reserve(frames.length * 4);
    this.op = this.reserve(20);
    this.scratch = this.reserve(np * 140 + 16);
    this.grow();
    this.bytes.set(packet, this.records);
    this.floats.set(frames, this.axes / 4);
  }
  table(fn, base) {
    const mark = this.cursor;
    while (true) {
      this.cursor = mark;
      this.links = this.reserve(24 * this.capacity);
      const slots = this.reserve(8 * this.capacity);
      this.grow();
      if (base) this.bytes.set(base, this.links);
      this.wasm.openOperator(this.op, this.links, slots, this.capacity, base ? base.length / 24 : 0);
      fn();
      if (!this.ints[this.op / 4 + 2]) return this.ints[this.op / 4];
      this.capacity *= 2;
    }
  }
  integrate(packet, fixed, first, end, buildStatic) {
    const started = performance.now();
    this.start(packet);
    const np = packet.length / 120,
      mark = this.cursor;
    const run = (count, correction) => {
      this.cursor = mark;
      const used = this.table(() =>
        this.wasm.sweep(
          this.records,
          count,
          this.vertices,
          4 * count,
          this.axes,
          directionCount,
          stripCount,
          this.op,
          correction,
          this.scratch,
          first,
          end,
        ),
      );
      return this.bytes.slice(this.links, this.links + 24 * used);
    };
    const stationary = buildStatic ? run(fixed, 0) : null;
    const change = run(np, 1);
    return {
      stationary,
      change,
      kernelMs: performance.now() - started,
      workingBytes: this.wasm.memory.buffer.byteLength,
    };
  }
  merge(packet, parts, base) {
    this.start(packet);
    // Incoming partials can be copied in turn into the same scratch space.
    const inputs = parts.filter(Boolean),
      size = Math.max(0, ...inputs.map((p) => p.length)),
      temp = this.reserve(size);
    this.grow();
    return this.table(() => {
      for (const p of inputs) {
        this.bytes.set(p, temp);
        this.wasm.mergeLinks(this.op, temp, p.length / 24);
      }
    }, base);
  }
  assemble(modes, near, packet, base, parts) {
    const n = 2 * modes.length;
    this.start(packet);
    const temp = this.reserve(Math.max(0, ...parts.map((p) => p.length)));
    let exact, mass;
    const used = this.table(() => {
      exact = this.reserve(4 * this.capacity);
      mass = this.reserve(4 * n);
      this.grow();
      this.ints.fill(0, exact / 4, exact / 4 + this.capacity);
      this.floats.fill(0, mass / 4, mass / 4 + n);
      for (const part of parts) {
        this.bytes.set(part, temp);
        this.wasm.mergeLinks(this.op, temp, part.length / 24);
      }
      for (const b of near) {
        if (b.offset > b.otherOffset) continue;
        for (let y = 0; y < b.size; y++)
          for (let x = 0; x < b.count; x++) {
            const a = b.offset + y,
              j = b.otherOffset + x,
              i = 2 * a + b.side,
              k = 2 * j + b.otherSide,
              t = y * b.count + x;
            const area = modes[a].area * b.mean[t],
              idx = this.wasm.installExact(this.op, i, k, area, b.m1[t], b.m2[t], b.m12[t]);
            if (idx >= 0) {
              this.ints[exact / 4 + idx] = 1;
              this.floats[mass / 4 + i] += b.mean[t];
              this.floats[mass / 4 + k] += area / modes[j].area;
            }
          }
      }
    }, base);
    const offset = this.reserve(4 * (n + 1)),
      next = this.reserve(4 * n),
      rows = this.reserve(used * 40),
      direct = this.reserve(12 * n),
      directSlope = this.reserve(12 * n),
      incoming = this.reserve(24 * n),
      outgoing = this.reserve(24 * n),
      slope = this.reserve(24 * n),
      outSlope = this.reserve(24 * n),
      bounds = this.reserve(16),
      raw = this.reserve(4 * n);
    this.grow();
    this.floats.fill(0, raw / 4, raw / 4 + n);
    this.wasm.prepareRows(this.records, n, this.op, offset, next, rows, exact, mass, raw, bounds);
    return {
      links: used,
      bytes: this.wasm.memory.buffer.byteLength + base.byteLength,
      solve: (lamp, gradient, colors, contacts = []) => {
        if (colors)
          for (let i = 0; i < modes.length; i++) this.floats.set(colors[i], this.records / 4 + i * 18 + 12);
        this.floats.set(lamp, direct / 4);
        this.floats.set(gradient, directSlope / 4);
        const iterations = this.wasm.solveLinks(
          this.records,
          n,
          offset,
          rows,
          direct,
          directSlope,
          incoming,
          outgoing,
          slope,
          outSlope,
          bounds,
        );
        if (!iterations) throw Error('Diffuse equilibrium failed to converge.');
        const memory = this.wasm.memory.buffer;
        const [residual, maxRow] = new Float64Array(memory, bounds, 2);
        const directional = incidentDirections(
          contacts,
          modes,
          this.ints.subarray(offset / 4),
          this.ints.subarray(rows / 4),
          this.floats.subarray(rows / 4),
          new Float64Array(memory, outgoing, 3 * n),
          new Float64Array(memory, outSlope, 3 * n),
        );
        return {
          directional,
          iterations,
          residual,
          maxRow,
          sidedSlopes: Float32Array.from(new Float64Array(memory, slope, 3 * n)),
          outSlope: Float32Array.from(new Float64Array(memory, outSlope, 3 * n)),
          incident: new Float64Array(memory, incoming, 3 * n).slice(),
        };
      },
    };
  }
}

export async function createSweep(binary) {
  const { instance } = await WebAssembly.instantiate(binary);
  return instanceSweep(instance);
}
export function instanceSweep(instance) {
  const core = new BeamWorkspace(instance.exports);
  let previous = [],
    base,
    lastModes,
    lastTransfer;
  return (modes, near, pool) => {
    if (lastTransfer && lastModes.length === modes.length && modes.every((m, i) => m === lastModes[i])) {
      return { ...lastTransfer, workerMs: 0, assemblyMs: 0 };
    }
    lastTransfer = null;
    const fixed = modes.filter((r) => !r.owner.axes),
      packet = pack(modes);
    const changed = !base || previous.length !== fixed.length || fixed.some((r, i) => r !== previous[i]);

    const snapshot = base;
    function finish(parts) {
      const start = performance.now();
      if (changed) {
        const used = core.merge(
          packet,
          parts.map((p) => p.stationary),
        );
        base = core.bytes.slice(core.links, core.links + used * 24);
        previous = fixed;
      }
      const result = core.assemble(
        modes,
        near,
        packet,
        changed ? base : snapshot,
        parts.map((p) => p.change),
      );
      lastModes = modes;
      lastTransfer = result;
      result.assemblyMs = performance.now() - start;
      result.workerMs = Math.max(...parts.map((p) => p.kernelMs));
      result.workerBytes = pool ? parts.reduce((s, p) => s + p.workingBytes, 0) : 0;
      return result;
    }
    if (pool) return pool.integrate(packet, fixed.length, changed).then(finish);
    return finish([core.integrate(packet, fixed.length, 0, directionCount, changed)]);
  };
}
