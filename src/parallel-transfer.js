// Twice-integrated parallel-rectangle kernel. The mixed derivative with
// respect to the four endpoint coordinates gives h² / (π r⁴).
import { dot, sub, unit, cross } from './cpu-geometry.js';

function primitive(x, y, h) {
  const a = Math.hypot(x, h),
    b = Math.hypot(y, h);
  return (
    (y * a * Math.atan(y / a) + x * b * Math.atan(x / b) - 0.5 * h * h * Math.log(x * x + y * y + h * h)) /
    (2 * Math.PI)
  );
}
function extent(shape, axis) {
  return shape.axes
    ? shape.axes.reduce((sum, a, i) => sum + Math.abs(dot(a, axis)) * shape.halfSize[i], 0)
    : Math.abs(dot(shape.axisU, axis)) + Math.abs(dot(shape.axisV, axis));
}
function interval(shape, axis) {
  const c = dot(shape.center, axis),
    d = extent(shape, axis);
  return [c - d, c + d];
}
function intersects(shape, axes, bounds) {
  return axes.every((axis, k) => {
    const [lo, hi] = interval(shape, axis);
    return hi > bounds[k][0] + 1e-8 && lo < bounds[k][1] - 1e-8;
  });
}
function coefficient(a, b, axes, h) {
  const x = interval(a, axes[0]),
    y = interval(a, axes[1]),
    u = interval(b, axes[0]),
    v = interval(b, axes[1]);
  let sum = 0;
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++)
      for (let k = 0; k < 2; k++)
        for (let l = 0; l < 2; l++)
          sum += ((i + j + k + l) % 2 ? -1 : 1) * primitive(x[i] - u[j], y[k] - v[l], h);
  return Math.max(0, sum);
}
export function compileParallel(sheets, groups, modes, covers) {
  const pairs = [];
  for (const a of sheets)
    for (const h of covers.get(a) || []) {
      const b = h.sheet,
        axes = [unit(a.axisU), unit(a.axisV), unit(cross(a.axisU, a.axisV))];
      if (Math.abs(dot(axes[0], axes[1])) > 1e-8 || Math.abs(dot(unit(b.axisU), unit(b.axisV))) > 1e-8)
        continue;
      if (
        ![axes[0], axes[1]].some((axis) => Math.hypot(...cross(axis, unit(b.axisU))) < 1e-9) ||
        Math.hypot(...cross(axes[2], unit(cross(b.axisU, b.axisV)))) > 1e-9
      )
        continue;
      const bounds = axes.map((axis) => {
        const [a0, a1] = interval(a, axis),
          [b0, b1] = interval(b, axis);
        return [Math.min(a0, b0), Math.max(a1, b1)];
      });
      if (sheets.some((s) => s !== a && s !== b && intersects(s, axes, bounds))) continue;
      if (pairs.some((p) => p.a === b && p.b === a)) continue;
      const ga = groups.get(a),
        gb = groups.get(b),
        sideB = Number(dot(sub(a.center, b.center), modes[gb.offset].normal) < 0);
      const rows = [];
      for (let i = ga.offset; i < ga.offset + ga.nu * ga.nv; i++)
        for (let j = gb.offset; j < gb.offset + gb.nu * gb.nv; j++) {
          const value = coefficient(modes[i], modes[j], axes, h.gap);
          rows.push([2 * i + h.side, 2 * j + sideB, value / modes[i].area, value / modes[j].area]);
        }
      const sides = [
        [a, b, h.side, sideB, ga, gb],
        [b, a, sideB, h.side, gb, ga],
      ].map(([receiver, other, side, otherSide, g, t], index) => {
        const count = t.nu * t.nv,
          size = g.nu * g.nv,
          mean = new Float32Array(size * count),
          mass = new Float64Array(size);
        for (const [i, j, wa, wb] of rows) {
          const row = (index ? j : i) / 2,
            col = (index ? i : j) / 2;
          const y = Math.floor(row) - g.offset,
            x = Math.floor(col) - t.offset,
            w = index ? wb : wa;
          mean[y * count + x] = w;
          mass[y] += w;
        }
        return {
          receiver,
          other,
          side,
          otherSide,
          count,
          size,
          mean,
          mass,
          offset: g.offset,
          otherOffset: t.offset,
          axes,
          height: h.gap,
        };
      });
      pairs.push({ a, b, axes, bounds, sides });
    }
  return pairs;
}
export function applyParallel(pairs, movers, K, n) {
  const active = pairs
    .filter((p) => !movers.some((s) => intersects(s, p.axes, p.bounds)))
    .flatMap((p) => p.sides);
  const byRow = new Map();
  for (const side of active)
    for (let y = 0; y < side.size; y++) {
      const i = 2 * (y + side.offset) + side.side;
      if (!byRow.has(i)) byRow.set(i, []);
      byRow.get(i).push([side, y]);
      for (let x = 0; x < side.count; x++) K[i * n + 2 * (side.otherOffset + x) + side.otherSide] = 0;
    }
  for (const [i, blocks] of byRow) {
    const row = K.subarray(i * n, (i + 1) * n),
      fixed = blocks.reduce((s, [b, y]) => s + b.mass[y], 0);
    let remaining = 0;
    for (let j = 0; j < n; j++) remaining += Math.max(0, row[j]);
    if (remaining > 1 - fixed && remaining > 0) {
      const scale = Math.max(0, 1 - fixed) / remaining;
      for (let j = 0; j < n; j++) row[j] *= scale;
    }
    for (const [b, y] of blocks)
      for (let x = 0; x < b.count; x++) row[2 * (b.otherOffset + x) + b.otherSide] = b.mean[b.count * y + x];
  }
  return active;
}
// Point-to-rectangle transfer evaluates a continuous function on the receiver.
// The unobstructed portion uses the same outgoing solution as the equilibrium.
export function nearField(block, grid, width, rows, solution) {
  const { modes, incident, outgoing } = solution,
    { axes, height: h, count, size } = block;
  if (!block.texels || block.texels.grid !== grid) {
    const weights = new Float32Array(width * rows * count),
      open = new Float32Array(width * rows);
    const rects = Array.from({ length: count }, (_, i) =>
      axes.slice(0, 2).map((axis) => interval(modes[block.otherOffset + i], axis)),
    );
    const term = (x, y) => {
      const a = Math.hypot(x, h),
        b = Math.hypot(y, h);
      return ((x / a) * Math.atan(y / a) + (y / b) * Math.atan(x / b)) / (2 * Math.PI);
    };
    for (let j = 0; j < rows; j++)
      for (let i = 0; i < width; i++) {
        const p = grid.origin.map((t, k) => t + (grid.u[k] * i) / (width - 1) + (grid.v[k] * j) / (rows - 1)),
          u = dot(p, axes[0]),
          v = dot(p, axes[1]),
          k = j * width + i;
        let sum = 0;
        for (let x = 0; x < count; x++) {
          const [[l, r], [b, t]] = rects[x];
          sum += weights[k * count + x] = Math.max(
            0,
            term(l - u, b - v) + term(r - u, t - v) - term(r - u, b - v) - term(l - u, t - v),
          );
        }
        open[k] = Math.max(0, 1 - sum);
      }
    block.texels = { grid, weights, open };
  }
  const emitted = new Float64Array(count * 3);
  for (let j = 0; j < count; j++)
    emitted.set(
      outgoing.subarray(
        6 * (block.otherOffset + j) + 3 * block.otherSide,
        6 * (block.otherOffset + j) + 3 * block.otherSide + 3,
      ),
      j * 3,
    );
  const other = new Float64Array(size * 3);
  for (let y = 0; y < size; y++)
    for (let c = 0; c < 3; c++) {
      let reflected = 0;
      for (let j = 0; j < count; j++) reflected += block.mean[y * count + j] * emitted[j * 3 + c];
      const full = incident[6 * (block.offset + y) + 3 * block.side + c];
      other[y * 3 + c] = Math.max(0, full - reflected) / Math.max(1e-8, 1 - block.mass[y]);
    }
  return { ...block.texels, other, emitted, count };
}
