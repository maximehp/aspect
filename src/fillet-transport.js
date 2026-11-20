import { hitImport } from './exact-hit.js';
import { dot, sub, volume } from './cpu-geometry.js';
import { painter } from './painter.js';
import { roundedBoxDistance } from './rounded-box.js';

const mul = (v, s) => v.map((x) => x * s);
export class Arena {
  constructor(module) {
    this.wasm = new WebAssembly.Instance(module, {
      geometry: { hit: hitImport((id) => this.shapes[id]) },
    }).exports;
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
export class Scanlines extends Arena {
  prepare(scene) {
    this.reset();
    this.scene = scene;
    this.shapes = scene.shapes;
    this.objects = this.put(Float64Array, scene.records);
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
