// Twice-integrated parallel-rectangle kernel. The mixed derivative with
// respect to the four endpoint coordinates gives h² / (π r⁴).
import { chartPoint } from './irradiance-layout.js';
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
// Summation by parts: each interior edge gets the difference between
// neighboring linear basis values. The constant integral stays analytic.
function footprint(mode, axes) {
  const d = Math.abs(dot(mode.momentAxis, axes[0])) > Math.abs(dot(mode.momentAxis, axes[1])) ? 0 : 1;
  const steps = 8,
    sign = Math.sign(dot(mode.momentAxis, axes[d]));
  const [us, vs] = axes.slice(0, 2).map((axis, k) => {
    const [lo, hi] = interval(mode, axis),
      n = k === d ? steps : 1;
    return Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
  });
  const result = [];
  for (let i = 0; i < us.length; i++)
    for (let j = 0; j < vs.length; j++) {
      const k = d === 0 ? i : j,
        t = d === 0 ? j : i,
        edge = t === 0 ? -1 : 1;
      const constant = k === 0 ? -1 : k === steps ? 1 : 0;
      const gradient = k === 0 || k === steps ? 1 - 1 / steps : -2 / steps;
      result.push({ x: us[i], y: vs[j], w: edge * constant, m: edge * gradient * sign });
    }
  return result;
}
function coefficients(a, b, kernel) {
  let area = 0,
    ma = 0,
    mb = 0,
    mixed = 0;
  for (const u of a)
    for (const v of b) {
      const f = kernel(u.x - v.x, u.y - v.y);
      area += f * u.w * v.w;
      ma += f * u.m * v.w;
      mb += f * u.w * v.m;
      mixed += f * u.m * v.m;
    }
  return [Math.max(0, area), ma, mb, mixed];
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
      const values = new Map();
      function kernel(x, y) {
        let col = values.get(x);
        if (!col) {
          col = new Map();
          values.set(x, col);
        }
        let f = col.get(y);
        if (f === undefined) {
          f = primitive(x, y, h.gap);
          col.set(y, f);
        }
        return f;
      }
      const functions = new Map();
      for (const g of [ga, gb])
        for (let i = g.offset; i < g.offset + g.nu * g.nv; i++) functions.set(i, footprint(modes[i], axes));
      for (let i = ga.offset; i < ga.offset + ga.nu * ga.nv; i++)
        for (let j = gb.offset; j < gb.offset + gb.nu * gb.nv; j++) {
          const [value, ma, mb, mm] = coefficients(functions.get(i), functions.get(j), kernel);
          rows.push([
            2 * i + h.side,
            2 * j + sideB,
            value / modes[i].area,
            value / modes[j].area,
            ma,
            mb,
            mm,
          ]);
        }
      const sides = [
        [a, b, h.side, sideB, ga, gb],
        [b, a, sideB, h.side, gb, ga],
      ].map(([receiver, other, side, otherSide, g, t], index) => {
        const count = t.nu * t.nv,
          size = g.nu * g.nv,
          mean = new Float32Array(size * count),
          mass = new Float64Array(size),
          m1 = new Float32Array(size * count),
          m2 = new Float32Array(size * count),
          m12 = new Float32Array(size * count);
        for (const [i, j, wa, wb, ma, mb, mm] of rows) {
          const row = (index ? j : i) / 2,
            col = (index ? i : j) / 2;
          const y = Math.floor(row) - g.offset,
            x = Math.floor(col) - t.offset,
            w = index ? wb : wa;
          mean[y * count + x] = w;
          m1[y * count + x] = index ? mb : ma;
          m2[y * count + x] = index ? ma : mb;
          m12[y * count + x] = mm;
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
          m1,
          m2,
          m12,
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
export function activeParallel(pairs, movers) {
  return pairs.filter((p) => !movers.some((s) => intersects(s, p.axes, p.bounds))).flatMap((p) => p.sides);
}
// Plate edges create narrow transport rows even far outside the projected plate.
// The exterior background uses a continuous area-weighted footprint at each
// display site. Covered modes carry less information about this background.
function backgroundFilter(block, modes, grid, width, rows) {
  const { axes, receiver, other, offset, size } = block;
  const radii = axes.slice(0, 2).map((axis) => {
    const half = extent(receiver, axis);
    return half / Math.min(5, Math.ceil(half / 0.82));
  });
  const shapes = modes.slice(offset, offset + size).map((mode) => ({
    mode,
    bounds: axes.slice(0, 2).map((a) => interval(mode, a)),
  }));
  function moments(lo, hi, x, radius) {
    const a = Math.max(-1, (lo - x) / radius),
      b = Math.min(1, (hi - x) / radius);
    if (a >= b) return [0, 0];
    const mass = (t) => t - 0.5 * t * Math.abs(t);
    const first = (t) => t * t * (0.5 - Math.abs(t) / 3);
    const w = mass(b) - mass(a);
    return [w, radius * (first(b) - first(a)) + (x - 0.5 * (lo + hi)) * w];
  }
  const plate = axes.slice(0, 2).map((axis) => interval(other, axis));
  const blend = new Float32Array(width * rows);
  const taps = [],
    weights = [],
    gradients = [],
    ends = [];
  for (let j = 0; j < rows; j++)
    for (let i = 0; i < width; i++) {
      const p = chartPoint(grid, i / (width - 1), j / (rows - 1));
      const xy = axes.slice(0, 2).map((a) => dot(p, a));
      const clearance = Math.hypot(...xy.map((x, k) => Math.max(0, plate[k][0] - x, x - plate[k][1])));
      const t = Math.min(1, clearance / (2 * block.height));
      blend[j * width + i] = t * t * (3 - 2 * t);
      if (t === 0) {
        ends.push(taps.length);
        continue;
      }
      const entries = [];
      let sum = 0;
      for (const [j, src] of shapes.entries()) {
        const [u, v] = src.bounds.map(([lo, hi], k) => moments(lo, hi, xy[k], radii[k]));
        const exposed = Math.max(0, 1 - block.mass[j]);
        const w = u[0] * v[0] * exposed;
        if (!w) continue;
        const m =
          u[1] * v[0] * dot(src.mode.momentAxis, axes[0]) + v[1] * u[0] * dot(src.mode.momentAxis, axes[1]);
        entries.push([j, w, exposed * m]);
        sum += w;
      }
      if (!entries.length) blend[j * width + i] = 0;
      for (const [id, w, m] of entries) {
        taps.push(id);
        weights.push(w / sum);
        gradients.push(m / sum);
      }
      ends.push(taps.length);
    }
  return {
    blend,
    ends: Uint32Array.from(ends),
    taps: Uint32Array.from(taps),
    weights: Float32Array.from(weights),
    gradients: Float32Array.from(gradients),
  };
}
// Point-to-rectangle transfer evaluates a continuous function on the receiver.
// The unobstructed portion uses the same outgoing solution as the equilibrium.
export function nearField(block, grid, width, rows, solution) {
  const { modes, incident, outgoing } = solution,
    { axes, height: h, count, size } = block;
  if (!block.texels || block.texels.grid !== grid) {
    const weights = new Float32Array(width * rows * count),
      open = new Float32Array(width * rows);
    const rects = Array.from({ length: count }, (_, i) => {
      const m = modes[block.otherOffset + i];
      return {
        bounds: axes.slice(0, 2).map((a) => interval(m, a)),
        axis: axes.slice(0, 2).map((a) => dot(a, m.momentAxis)),
      };
    });
    const gradients = new Float32Array(weights.length);
    const term = (x, y) => {
      const a = Math.hypot(x, h),
        b = Math.hypot(y, h);
      return ((x / a) * Math.atan(y / a) + (y / b) * Math.atan(x / b)) / (2 * Math.PI);
    };
    for (let j = 0; j < rows; j++)
      for (let i = 0; i < width; i++) {
        const p = chartPoint(grid, i / (width - 1), j / (rows - 1)),
          u = dot(p, axes[0]),
          v = dot(p, axes[1]),
          k = j * width + i;
        let sum = 0;
        for (let x = 0; x < count; x++) {
          const {
              bounds: [[l, r], [b, t]],
              axis: [du, dv],
            } = rects[x],
            L = l - u,
            R = r - u,
            B = b - v,
            T = t - v;
          const w = Math.max(0, term(L, B) + term(R, T) - term(R, B) - term(L, T));
          // Integrating the linear source coordinate by parts leaves integrals
          // of 1/r² along the four edges of the source rectangle.
          const edge = (d, a, b) => {
            const q = Math.hypot(d, h);
            return (Math.atan(b / q) - Math.atan(a / q)) / q;
          };
          let m = -(du * (L + R) + dv * (B + T)) * 0.5 * w;
          if (du) m += ((du * h * h) / (2 * Math.PI)) * (edge(L, B, T) - edge(R, B, T));
          if (dv) m += ((dv * h * h) / (2 * Math.PI)) * (edge(B, L, R) - edge(T, L, R));
          sum += weights[k * count + x] = w;
          gradients[k * count + x] = Math.max(-w, Math.min(w, m));
        }
        open[k] = Math.max(0, 1 - sum);
      }
    block.texels = { grid, weights, gradients, open };
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
  const slopes = new Float32Array(count * 3);
  for (let j = 0; j < count; j++) {
    const p = 6 * (block.otherOffset + j) + 3 * block.otherSide;
    slopes.set(solution.outSlope.subarray(p, p + 3), 3 * j);
  }
  const other = new Float64Array(size * 3),
    otherSlope = new Float64Array(size * 3);
  for (let y = 0; y < size; y++)
    for (let c = 0; c < 3; c++) {
      const area = modes[block.offset + y].area;
      let reflected = 0,
        reflectedSlope = 0,
        blockedSlope = 0;
      for (let j = 0; j < count; j++) {
        const k = y * count + j,
          e = emitted[j * 3 + c],
          b = slopes[j * 3 + c];
        reflected += block.mean[k] * e + (block.m2[k] * b) / area;
        reflectedSlope += (3 * (block.m1[k] * e + block.m12[k] * b)) / area;
        blockedSlope += (3 * block.m1[k]) / area;
      }
      const i = 6 * (block.offset + y) + 3 * block.side + c;
      const open = Math.max(1e-8, 1 - block.mass[y]);
      // Factor the residual as open solid angle times background radiance.
      // Use the open-area centroid so its reconstructed slope preserves flux.
      const mean = Math.max(0, incident[i] - reflected) / open;
      const centroid = Math.max(-1, Math.min(1, -blockedSlope / (3 * open)));
      const gradient = (solution.sidedSlopes[i] - reflectedSlope + blockedSlope * mean) / open;
      const slope = Math.max(
        -mean / (1 - centroid || 1e-8),
        Math.min(mean / (1 + centroid || 1e-8), gradient),
      );
      other[y * 3 + c] = mean - centroid * slope;
      otherSlope[y * 3 + c] = slope;
    }
  if (block.background?.grid !== grid)
    block.background = { grid, ...backgroundFilter(block, modes, grid, width, rows) };
  const filter = block.background,
    background = new Float32Array(width * rows * 3);
  let tap = 0;
  for (let i = 0; i < width * rows; i++) {
    for (; tap < filter.ends[i]; tap++) {
      const j = filter.taps[tap] * 3,
        w = filter.weights[tap],
        m = filter.gradients[tap];
      for (let c = 0; c < 3; c++) background[3 * i + c] += other[j + c] * w + otherSlope[j + c] * m;
    }
  }
  return { ...block.texels, other, otherSlope, background, blend: filter.blend, emitted, slopes, count };
}
