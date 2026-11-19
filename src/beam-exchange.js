import { cross, unit, point } from './cpu-geometry.js';
const dirs = 96,
  bins = 192;
const frames = [];
for (let i = 0; i < dirs; i++) {
  const z = (i + 0.5) / dirs,
    angle = i * 2.399963229728653,
    r = Math.sqrt(1 - z * z);
  const w = [r * Math.cos(angle), r * Math.sin(angle), z],
    u = unit(cross(w, [0, 0, 1]));
  frames.push(...u, ...cross(w, u), ...w);
}
export async function createSweep(binary) {
  const { instance } = await WebAssembly.instantiate(binary);
  const { memory, sweep, equilibrium, __heap_base } = instance.exports;
  let staticModes = [],
    base;
  return (modes) => {
    const n = 2 * modes.length,
      np = modes.length,
      nv = 4 * np;
    let cursor = Number(__heap_base);
    const reserve = (bytes) => {
      const p = cursor;
      cursor += bytes;
      return p;
    };
    const fixed = modes.filter((r) => !r.owner.axes);
    const records = reserve(np * 72),
      vertices = reserve(12 * nv),
      axes = reserve(frames.length * 4);
    const weights = reserve(4 * n * n),
      source = reserve(12 * n),
      moments = reserve((np - fixed.length) * n * 4),
      slopes = reserve(np * 12),
      scratch = reserve(8 * nv + 88 * np);
    cursor = Math.ceil(cursor / 8) * 8;
    const incident = reserve(24 * n),
      emitted = reserve(24 * n),
      bounds = reserve(16);
    if (cursor > memory.buffer.byteLength)
      memory.grow(Math.ceil((cursor - memory.buffer.byteLength) / 65536));
    const F = new Float32Array(memory.buffer),
      I = new Int32Array(memory.buffer);
    F.set(frames, axes / 4);
    F.fill(0, source / 4, scratch / 4);
    const owners = new Map();
    let face = records / 4,
      vert = 0;
    for (const [j, r] of modes.entries()) {
      if (!owners.has(r.owner)) owners.set(r.owner, owners.size);
      const uv = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ];
      I.set([vert, uv.length, j, owners.get(r.owner), +!!r.owner.axes], face);
      F.set([r.area, ...r.normal, ...r.center, ...r.color, ...r.momentAxis], face + 5);
      for (const [u, v] of uv) {
        F.set(point(r, u, v), vertices / 4 + 3 * vert);
        vert++;
      }
      face += 18;
    }
    const K = new Float32Array(memory.buffer, weights, n * n),
      G = new Float32Array(memory.buffer, source, 3 * n);

    if (
      !base ||
      base.length !== K.length ||
      fixed.length !== staticModes.length ||
      fixed.some((r, i) => r !== staticModes[i])
    ) {
      K.fill(0);
      sweep(
        records,
        fixed.length,
        vertices,
        fixed.length * 4,
        axes,
        dirs,
        bins,
        n,
        weights,
        0,
        scratch,
        moments,
        fixed.length,
      );
      base = K.slice();
      staticModes = fixed;
    } else K.set(base);
    sweep(records, np, vertices, nv, axes, dirs, bins, n, weights, 1, scratch, moments, fixed.length);
    return {
      K,
      direct: G,
      bytes: memory.buffer.byteLength + base.byteLength,
      solve() {
        const iterations = equilibrium(
          records,
          n,
          weights,
          source,
          incident,
          emitted,
          scratch,
          scratch + 4 * n,
          bounds,
          moments,
          fixed.length,
          slopes,
        );
        if (!iterations) throw Error('Diffuse equilibrium failed to converge.');
        const [residual, maxRow] = new Float64Array(memory.buffer, bounds, 2);
        return {
          slopes: new Float32Array(memory.buffer, slopes, np * 3).slice(),
          iterations,
          residual,
          maxRow,
          incident: new Float64Array(memory.buffer, incident, 3 * n).slice(),
        };
      },
    };
  };
}
