import { parallelBilinearExchange } from './rectangle-exchange.js';
function box(o) {
  return o.center.map((c, k) => {
    const h = o.axes
      ? o.axes.reduce((n, a, d) => n + Math.abs(a[k]) * o.halfSize[d], 0)
      : Math.abs(o.axisU[k]) + Math.abs(o.axisV[k]);
    return [c - h, c + h];
  });
}
function axes(s) {
  if (s.axes) return;
  let axes = [s.axisU, s.axisV].map((a) => a.findIndex((x) => x !== 0));
  if (
    axes.includes(-1) ||
    axes[0] === axes[1] ||
    [s.axisU, s.axisV].some((a) => a.filter((x) => x !== 0).length !== 1)
  )
    return;
  return [...axes, 3 - axes[0] - axes[1]];
}
function oneSide(g, axis, k, cell) {
  const v = k ? g.v : g.ax,
    w = k ? g.height : g.width;
  return g.knots[k]
    .slice(cell, cell + 2)
    .map((u) => g.c[axis] + (2 * u - 1) * w * v[axis])
    .sort((a, b) => a - b);
}
function elements(g, uv) {
  const tiles = [],
    w = [g.ax, g.v],
    ix = uv.map((a) => w.findIndex((v) => v[a] !== 0));
  for (let j = 0; j < g.rows; j++)
    for (let i = 0; i < g.cols; i++) {
      const ij = [i, j];
      const nodes = Array.from({ length: 4 }, (_, n) => {
        const c = [i, j];
        for (let k = 0; k < 2; k++) c[ix[k]] += ((n >> k) & 1) ^ Number(w[ix[k]][uv[k]] < 0);
        return g.ids[c[0] + c[1] * (g.cols + 1)];
      });
      tiles.push({ rect: uv.map((a, k) => oneSide(g, a, ix[k], ij[ix[k]])), nodes });
    }
  return tiles;
}
export function selectPairs(geometry) {
  const { shapes, maps, receivers } = geometry,
    boxes = shapes.map(box),
    normal = shapes.map(axes),
    pairs = [];
  for (let i = 0; i < shapes.length; i++)
    for (let j = i + 1; j < shapes.length; j++) {
      if (!normal[i] || normal[i][2] !== normal[j]?.[2]) continue;
      const [a, b, n] = normal[i],
        A = boxes[i],
        B = boxes[j],
        gap = Math.abs(A[n][0] - B[n][0]);
      if (!gap || gap > 0.35 * Math.min(...[a, b].flatMap((k) => [A[k][1] - A[k][0], B[k][1] - B[k][0]])))
        continue;
      const hull = A.map(([l, h], k) => [Math.min(l, B[k][0]), Math.max(h, B[k][1])]);
      if (
        boxes.some((c, k) => k !== i && k !== j && c.every(([l, h], d) => h > hull[d][0] && l < hull[d][1]))
      )
        continue;
      const sign = Math.sign(B[n][0] - A[n][0]),
        f = maps[i].find((g) => g.n?.[n] === sign),
        t = maps[j].find((g) => g.n?.[n] === -sign);
      if (!f || !t) continue;
      pairs.push({ i, j, f, t, a, b, gap });
    }
  return pairs;
}
export function weakPairs(geometry, selected) {
  const { shapes } = geometry,
    pairs = [];
  for (const { i, j, f, t, a, b, gap } of selected) {
    const fx = elements(f, [a, b]),
      tx = elements(t, [a, b]),
      op = parallelBilinearExchange(gap);
    const F = f.ids,
      T = t.ids,
      li = new Map(F.map((v, k) => [v, k])),
      lj = new Map(T.map((v, k) => [v, k]));
    const massA = new Float64Array(F.length),
      massB = new Float64Array(T.length),
      weights = new Float64Array(F.length * T.length);
    function mass(c, m, index) {
      for (let e of c) {
        const ar = e.rect.reduce((p, [l, h]) => p * (h - l), 1);
        for (let id of e.nodes) m[index.get(id)] += ar / 4;
      }
    }
    mass(fx, massA, li);
    mass(tx, massB, lj);
    function layout(chart) {
      const k = [a, b].map((axis) => (chart.ax[axis] ? 0 : 1)),
        axes = [a, b].map((axis, i) =>
          chart.knots[k[i]].map(
            (u) =>
              chart.c[axis] +
              (2 * u - 1) * (k[i] ? chart.height : chart.width) * (k[i] ? chart.v : chart.ax)[axis],
          ),
        );
      const reverse = axes.map((x) => x[0] > x.at(-1));
      axes.forEach((x, i) => {
        if (reverse[i]) x.reverse();
      });
      const order = [];
      for (let j = 0; j < axes[1].length; j++)
        for (let i = 0; i < axes[0].length; i++) {
          const p = [0, 0];
          for (let d = 0; d < 2; d++) p[k[d]] = reverse[d] ? axes[d].length - 1 - [i, j][d] : [i, j][d];
          order.push(p[1] * (chart.cols + 1) + p[0]);
        }
      return { axes, order };
    }
    const x = layout(f),
      y = layout(t),
      W = op.grid(y.axes, x.axes);
    let negative = 0,
      min = 0;
    for (let row = 0; row < x.order.length; row++)
      for (let col = 0; col < y.order.length; col++) {
        const w = W[row * T.length + col];
        weights[x.order[row] * T.length + y.order[col]] = w;
        if (w < 0) {
          negative++;
          min = Math.min(min, w);
        }
      }
    pairs.push({
      a: shapes[i].id,
      b: shapes[j].id,
      F,
      T,
      massA,
      massB,
      weights,
      negative,
      min,
      corners: op.evaluations,
      cells: fx.length * tx.length,
      area: weights.reduce((a, b) => a + b, 0),
      f,
      t,
    });
  }
  return pairs;
}
// Audit assembly on the instrumented solver. The strong operator obtains
// coefficients by inverting the consistent tensor-product FE mass, or by
// dividing by the integrated hat. Both use the same underlying bilinear form.
function massInverse(g, vec, stride, offset) {
  const n = g.cols + 1;
  for (const d of [0, 1]) {
    const knots = g.knots[d],
      size = knots.length,
      lines = d ? n : g.rows + 1,
      step = d ? n : 1;
    let lo = new Float64Array(size),
      diag = new Float64Array(size),
      width = (d ? g.height : g.width) * 2;
    for (let p = 1; p < size; p++) {
      lo[p] = ((knots[p] - knots[p - 1]) * width) / 6;
      diag[p - 1] += 2 * lo[p];
      diag[p] += 2 * lo[p];
    }
    const mult = new Float64Array(size);
    for (let p = 1; p < size; p++) {
      mult[p] = lo[p] / diag[p - 1];
      diag[p] -= mult[p] * lo[p];
    }
    for (let line = 0; line < lines; line++) {
      const base = offset + (d ? line : line * n) * stride,
        inc = step * stride;
      for (let i = 1; i < size; i++) vec[base + i * inc] -= mult[i] * vec[base + (i - 1) * inc];
      vec[base + (size - 1) * inc] /= diag[size - 1];
      for (let i = size - 2; i >= 0; i--)
        vec[base + i * inc] = (vec[base + i * inc] - lo[i + 1] * vec[base + (i + 1) * inc]) / diag[i];
    }
  }
}
export function projectedRows(geometry, pairs, method) {
  const consistent = method !== 'lumped';
  const N = geometry.receivers.length,
    subs = Array.from({ length: N }, () => []);
  for (const p of pairs)
    for (const forward of [true, false]) {
      const rows = forward ? p.F : p.T,
        cols = forward ? p.T : p.F,
        mass = forward ? p.massA : p.massB;
      const m = new Float64Array(rows.length * cols.length);
      for (let i = 0; i < rows.length; i++)
        for (let j = 0; j < cols.length; j++)
          m[i * cols.length + j] =
            p.weights[forward ? i * cols.length + j : j * rows.length + i] / (consistent ? 1 : mass[i]);
      if (consistent)
        for (let j = 0; j < cols.length; j++) massInverse(forward ? p.f : p.t, m, cols.length, j);
      if (method === 'positive')
        for (let j = 0; j < cols.length; j++) {
          let target = 0,
            neg = false;
          for (let i = 0; i < rows.length; i++) {
            const v = m[i * cols.length + j];
            target += mass[i] * v;
            neg ||= v < 0;
          }
          if (!neg) continue;
          // Project onto nonnegative nodal values, retaining integral of the column.
          let shift = 0;
          for (let k = 0; k < rows.length; k++) {
            let total = 0,
              active = 0;
            for (let i = 0; i < rows.length; i++) {
              const v = m[i * cols.length + j];
              if (v > shift) {
                total += mass[i] * v;
                active += mass[i];
              }
            }
            if (!active) break;
            const next = (total - Math.max(0, target)) / active;
            if (next <= shift + 3e-15) {
              shift = next;
              break;
            }
            shift = next;
          }
          for (let i = 0; i < rows.length; i++)
            m[i * cols.length + j] = Math.max(0, m[i * cols.length + j] - shift);
        }
      rows.forEach((r, i) =>
        subs[r].push({ cols, weights: m.subarray(i * cols.length, (i + 1) * cols.length) }),
      );
    }
  const starts = new Uint32Array(N + 1);
  for (let row = 0; row < N; row++) {
    let count = 0;
    for (const t of subs[row]) for (const w of t.weights) if (w) count++;
    starts[row + 1] = starts[row] + count;
  }
  const col = new Uint32Array(starts[N]),
    weights = new Float64Array(col.length);
  for (let row = 0; row < N; row++) {
    let k = starts[row];
    for (const t of subs[row])
      for (let j = 0; j < t.cols.length; j++)
        if (t.weights[j]) {
          col[k] = t.cols[j];
          weights[k++] = t.weights[j];
        }
  }
  return { starts, col, weights };
}
export function mergeParts(parts, extra) {
  for (const p of parts) {
    const starts = new Uint32Array(p.starts.length);
    for (let r = 0; r + 1 < starts.length; r++) {
      const i = p.first + p.stride * r;
      starts[r + 1] = starts[r] + extra.starts[i + 1] - extra.starts[i] + p.starts[r + 1] - p.starts[r];
    }
    const col = new Uint32Array(starts.at(-1)),
      weights = new Float64Array(col.length);
    for (let r = 0; r + 1 < starts.length; r++) {
      const i = p.first + p.stride * r,
        a = p.starts[r],
        b = p.starts[r + 1],
        l = extra.starts[i],
        h = extra.starts[i + 1],
        k = starts[r];
      col.set(p.col.subarray(a, b), k);
      weights.set(p.weights.subarray(a, b), k);
      col.set(extra.col.subarray(l, h), k + b - a);
      weights.set(extra.weights.subarray(l, h), k + b - a);
    }
    p.starts = starts;
    p.weights = weights;
    p.col = col;
  }
}

export function markPairs(geometry, pairs) {
  const { groups, tiles, packet } = geometry;
  for (const { f, t } of pairs)
    for (const [x, y] of [
      [f, t],
      [t, f],
    ]) {
      const sources = [];
      tiles.forEach((s, k) => {
        if (s.offset === x.offset) sources.push(k);
      });
      groups.forEach((g, j) => {
        if (g.object === y.object && g.n === y.n)
          for (const k of sources) packet.regions[j * tiles.length + k] |= 128;
      });
    }
}
