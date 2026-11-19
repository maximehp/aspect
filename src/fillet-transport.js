import { cross, dot, unit, sub, volume } from './cpu-geometry.js';
import { painter } from './painter.js';
import { roundedBoxDistance } from './rounded-box.js';

const mul = (v, s) => v.map((x) => x * s);
export function shapeRecord(s) {
  if (s.axes)
    return [
      ...s.center,
      ...s.axes.flat(),
      ...s.halfSize,
      s.radius,
      ...s.bounds.max.map((v, i) => v - s.center[i]),
    ];
  const normal = unit(cross(s.axisU, s.axisV));
  const u = cross(s.axisV, normal),
    v = cross(normal, s.axisU);
  return [
    ...s.center,
    ...mul(u, 1 / dot(u, s.axisU)),
    ...mul(v, 1 / dot(v, s.axisV)),
    ...normal,
    0,
    0,
    0,
    0,
    ...s.axisU.map((v, i) => Math.abs(v) + Math.abs(s.axisV[i])),
  ];
}
export class Arena {
  constructor(module) {
    this.wasm = new WebAssembly.Instance(module).exports;
    this.reset();
  }
  planes(eye, shapes = this.shapes, cones) {
    const ranges = [],
      planes = [];
    for (const [i, s] of shapes.entries()) {
      const p = cones ? cones[i] : volume(s, eye) || [];
      ranges.push(planes.length / 4, p.length);
      for (const h of p) planes.push(...h.normal, dot(h.normal, h.origin));
    }
    return [this.put(Float64Array, planes), this.put(Int32Array, ranges)];
  }
  reset() {
    this.end = Number(this.wasm.__heap_base);
  }
  space(bytes) {
    const at = Math.ceil(this.end / 8) * 8;
    this.end = at + bytes;
    const memory = this.wasm.memory;
    if (this.end > memory.buffer.byteLength)
      memory.grow(Math.ceil((this.end - memory.buffer.byteLength) / 65536));
    return at;
  }
  put(Type, data) {
    const p = this.space(Type.BYTES_PER_ELEMENT * data.length);
    new Type(this.wasm.memory.buffer, p, data.length).set(data);
    return p;
  }
}
// A normal belongs to its largest signed component. This partitions the true
// surface into six integration charts, including half of each adjacent quarter
// cylinder and one third of each adjacent spherical octant.
const gauss = [(1 - 1 / Math.sqrt(3)) / 2, (1 + 1 / Math.sqrt(3)) / 2];
function quadrature(box, group, modes, object, sites, areas) {
  const h = box.halfSize,
    r = box.radius,
    core = h.map((x) => x - r);
  const world = (local, origin) =>
    origin.map((c, k) => c + local.reduce((s, x, i) => s + x * box.axes[i][k], 0));
  for (let f = 0; f < 6; f++) {
    const d = f >> 1,
      sign = f % 2 ? -1 : 1,
      a = (d + 1) % 3,
      b = (d + 2) % 3;
    const { offset, nu, nv } = group[f];
    function site(p, n, area) {
      const u = (sign * p[a]) / h[a],
        v = p[b] / h[b];
      let m = offset;
      for (; m < offset + nu * nv - 1; m++) {
        const [l, z, lo, hi] = modes[m].bounds;
        if (u >= l - 1e-10 && u <= z + 1e-10 && v >= lo - 1e-10 && v <= hi + 1e-10) break;
      }
      sites.push(...world(p, box.center), ...world(n, [0, 0, 0]), area, object, m);
      areas[m] += area;
    }
    const normal = [0, 0, 0];
    normal[d] = sign;
    for (let i = offset; i < offset + nu * nv; i++) {
      const [l, z, lo, hi] = modes[i].bounds;
      const x0 = Math.max(-core[a], Math.min(sign * l, sign * z) * h[a]),
        x1 = Math.min(core[a], Math.max(sign * l, sign * z) * h[a]);
      const y0 = Math.max(-core[b], lo * h[b]),
        y1 = Math.min(core[b], hi * h[b]);
      if (x1 <= x0 || y1 <= y0) continue;
      for (const u of gauss)
        for (const v of gauss) {
          const p = [0, 0, 0];
          p[d] = sign * h[d];
          p[a] = x0 + (x1 - x0) * u;
          p[b] = y0 + (y1 - y0) * v;
          site(p, normal, ((x1 - x0) * (y1 - y0)) / 4);
        }
    }
    for (const edge of [a, b])
      for (const side of [-1, 1]) {
        const along = edge === a ? b : a;
        const breaks = new Set([-core[along], core[along]]);
        for (let i = offset; i < offset + nu * nv; i++)
          for (const t of modes[i].bounds.slice(along === a ? 0 : 2, along === a ? 2 : 4)) {
            const x = t * h[along] * (along === a ? sign : 1);
            if (Math.abs(x) < core[along]) breaks.add(x);
          }
        const xs = [...breaks].sort((a, b) => a - b);
        for (let j = 1; j < xs.length; j++)
          for (const t of gauss)
            for (const u of gauss) {
              const theta = (t * Math.PI) / 4,
                n = [0, 0, 0],
                p = [0, 0, 0];
              n[d] = sign * Math.cos(theta);
              n[edge] = side * Math.sin(theta);
              p[d] = sign * core[d] + r * n[d];
              p[edge] = side * core[edge] + r * n[edge];
              p[along] = xs[j - 1] + u * (xs[j] - xs[j - 1]);
              site(p, n, (((r * Math.PI) / 4) * (xs[j] - xs[j - 1])) / 4);
            }
      }
    for (const sa of [-1, 1])
      for (const sb of [-1, 1]) {
        const samples = gauss.flatMap((u) =>
          gauss.map((v) => {
            const n = [0, 0, 0];
            n[d] = sign;
            n[a] = sa * u;
            n[b] = sb * v;
            return [unit(n), (1 + u * u + v * v) ** -1.5];
          }),
        );
        const sum = samples.reduce((n, s) => n + s[1], 0);
        for (const [n, w] of samples)
          site(
            n.map((x, k) => Math.sign(x) * core[k] + r * x),
            n,
            (Math.PI * r * r * w) / (6 * sum),
          );
      }
  }
}
const directions = Array.from({ length: 96 }, (_, i) => {
  const z = (i + 0.5) / 96,
    t = i * 2.399963229728653,
    h = Math.sqrt(1 - z * z);
  return [h * Math.cos(t), h * Math.sin(t), z];
}).flat();
export class FilletTransport extends Arena {
  recover(vertices, mode, light) {
    this.end = this.mark;
    const data = this.put(Float64Array, [
      ...light,
      ...mode.center,
      ...mode.momentAxis,
      ...mode.normal,
      ...vertices.flat(),
    ]);
    const out = this.space(16);
    this.wasm.recoverFlux(this.rounded, this.boxCount, ...this.cones, data, vertices.length, out);
    return new Float64Array(this.wasm.memory.buffer, out, 2);
  }
  compute(scene, modes, groups, charts, parallel) {
    this.reset();
    const shapes = [...scene.rectangles, ...scene.roundedBoxes],
      fixed = scene.rectangles.length;
    const shape = this.put(Float64Array, shapes.flatMap(shapeRecord));
    const groupData = [],
      sites = [],
      areas = new Float64Array(modes.length);
    for (let i = 0; i < shapes.length; i++) {
      const faces = i < fixed ? [shapes[i]] : charts[i - fixed],
        layout = faces.map((face) => groups.get(face));
      for (let k = 0; k < 6; k++)
        groupData.push(layout[k]?.offset || 0, layout[k] ? layout[k].nu * layout[k].nv : 0);
      if (i >= fixed) quadrature(shapes[i], layout, modes, i, sites, areas);
    }
    const mode = this.put(
      Float64Array,
      modes.flatMap((m) => [...m.center, ...m.momentAxis, ...m.bounds]),
    );
    const index = this.put(Int32Array, groupData),
      count = sites.length / 9;
    const data = this.space(64 * count);
    const b = this.wasm.memory.buffer;
    const values = new Float64Array(b, data, 8 * count),
      ids = new Int32Array(b, data, 16 * count);
    for (let i = 0; i < count; i++) {
      const k = 9 * i;
      for (let j = 0; j < 7; j++) values[8 * i + j] = sites[k + j];
      ids[16 * i + 14] = sites[k + 7];
      ids[16 * i + 15] = sites[k + 8];
    }
    this.rounded = shape + 152 * fixed;
    this.boxCount = shapes.length - fixed;
    const angles = this.put(Float64Array, directions),
      light = this.put(Float64Array, scene.lightPosition);
    const flux = this.put(Float64Array, new Float64Array(2 * modes.length));
    this.wasm.directSites(shape, shapes.length, data, count, light, mode, flux);
    this.cones = this.planes(scene.lightPosition, scene.roundedBoxes);
    this.mark = this.end;
    const args = [shape, shapes.length, fixed, data, count, index, mode, angles, 96];
    const memory = this.wasm.memory.buffer;
    const result = { areas, flux: new Float64Array(memory, flux, areas.length * 2).slice(), sites: count };
    if (parallel) {
      const heap = Number(this.wasm.__heap_base);
      result.packet = { args, bytes: new Uint8Array(memory, heap, this.mark - heap).slice() };
    } else result.change = this.links(args, 0, 1);
    return result;
  }
  integrate(packet, first, stride) {
    this.reset();
    this.put(Uint8Array, packet.bytes);
    return this.links(packet.args, first, stride);
  }
  links(args, first, stride) {
    let size = this.capacity || 2048,
      mark = this.end;
    for (;;) {
      this.end = mark;
      const links = this.space(24 * size),
        slots = this.space(8 * size);
      const used = this.wasm.surfaceExchange(...args, links, slots, size, first, stride);
      if (used < 0) {
        size *= 2;
        continue;
      }
      this.capacity = size;
      return new Uint8Array(this.wasm.memory.buffer, links, used * 24).slice();
    }
  }
}
export class Scanlines extends Arena {
  prepare(scene) {
    this.reset();
    this.scene = scene;
    this.shapes = [...scene.rectangles, ...scene.roundedBoxes];
    this.objects = this.put(Float64Array, this.shapes.flatMap(shapeRecord));
    const cones = this.shapes.map((s) => volume(s, scene.lightPosition) || []);
    this.lamp = this.planes(scene.lightPosition, this.shapes, cones);
    const blocks = [],
      ids = [];
    this.shapes.forEach((s, id) => {
      const corners = s.axes
        ? Array.from({ length: 8 }, (_, i) =>
            s.center.map((c, k) =>
              s.axes.reduce((x, a, j) => x + ((i >> j) & 1 ? 1 : -1) * s.halfSize[j] * a[k], c),
            ),
          )
        : Array.from({ length: 4 }, (_, i) =>
            s.center.map((c, k) => c + (i & 1 ? 1 : -1) * s.axisU[k] + (i & 2 ? 1 : -1) * s.axisV[k]),
          );
      const start = ids.length;
      let dark = false;
      for (let i = 0; i < this.shapes.length; i++) {
        if (i === id) continue;
        const cone = cones[i],
          block = this.shapes[i];
        const bound = (h, side) => {
          const d = dot(s.center, h.normal) - dot(h.origin, h.normal);
          const r = s.axes
            ? s.radius +
              s.axes.reduce((a, v, j) => a + Math.abs(dot(v, h.normal)) * (s.halfSize[j] - s.radius), 0)
            : Math.abs(dot(s.axisU, h.normal)) + Math.abs(dot(s.axisV, h.normal));
          return side * d + r < -1e-9;
        };
        if (cone.some((h) => bound(h, 1))) continue;
        if (
          cone.length &&
          cone.every((h) => bound(h, -1)) &&
          (!block.axes ||
            corners.every(
              (p) =>
                roundedBoxDistance(block, scene.lightPosition, sub(p, scene.lightPosition), 1e-8, 1 - 1e-9) <
                1,
            ))
        ) {
          dark = true;
          break;
        }
        ids.push(i);
      }
      if (dark) ids.length = start;
      blocks.push(start, dark ? -1 : ids.length - start);
    });
    this.blockers = this.put(Int32Array, ids);
    this.blocks = this.put(Int32Array, blocks);
    this.mark = this.end;
  }
  render(camera, w, h) {
    this.end = this.mark;
    const view = this.planes(camera.position);
    const ordered = painter(this.shapes, camera.position).reverse(),
      ids = [],
      groups = [];
    ordered.forEach((group, i) => {
      for (const id of Array.isArray(group) ? group : [group]) {
        ids.push(id);
        groups.push(i);
      }
    });
    const order = this.put(Int32Array, ids),
      group = this.put(Int32Array, groups);
    const cx = Math.cos(camera.yaw),
      sx = Math.sin(camera.yaw),
      cy = Math.cos(camera.pitch),
      sy = Math.sin(camera.pitch);
    const step = (2 * Math.tan(camera.fov / 2)) / h,
      dx = mul([cx, 0, sx], step),
      dy = mul([-sx * sy, cy, cx * sy], step);
    const base = [sx * cy, sy, -cx * cy].map((x, k) => x + (0.5 - w / 2) * dx[k] + (0.5 - h / 2) * dy[k]);
    const frame = this.put(Float64Array, [
      ...camera.position,
      ...this.scene.lightPosition,
      ...base,
      ...dx,
      ...dy,
    ]);
    const forward = [sx * cy, sy, -cx * cy],
      near = [],
      far = [];
    for (const s of this.shapes) {
      const c = dot(sub(s.center, camera.position), forward);
      const extent = s.axes
        ? s.radius + s.axes.reduce((a, v, i) => a + Math.abs(dot(v, forward)) * (s.halfSize[i] - s.radius), 0)
        : Math.abs(dot(s.axisU, forward)) + Math.abs(dot(s.axisV, forward));
      near.push(c - extent);
      far.push(c + extent);
    }
    const depth = this.put(Float64Array, [...near, ...far]);
    const stats = this.space(16),
      mark = this.end;
    let cap = this.capacity || 4 * h;
    for (;;) {
      this.end = mark;
      const out = this.space(20 * cap);
      new Int32Array(this.wasm.memory.buffer, stats, 4).fill(0);
      const used = this.wasm.renderRuns(
        this.objects,
        ids.length,
        ...view,
        ...this.lamp,
        order,
        group,
        frame,
        this.blockers,
        this.blocks,
        depth,
        w,
        h,
        out,
        cap,
        stats,
      );
      if (used > cap) {
        cap = Math.max(cap * 2, used);
        continue;
      }
      this.capacity = cap;
      return {
        runs: new Int32Array(this.wasm.memory.buffer, out, 5 * used),
        stats: Array.from(new Int32Array(this.wasm.memory.buffer, stats, 4)),
        base,
        dx,
        dy,
      };
    }
  }
}
