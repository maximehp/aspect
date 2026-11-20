// Install one measured weak exchange block in both directions and re-solve
// the actual coupled room. This comparison is loaded only by optical probes.
import { parallelBilinearExchange } from '../src/rectangle-exchange.js';
import { parallelLineDomain, parallelLineExchange } from '../src/parallel-line-exchange.js';
import { modalEquilibrium } from '../src/modal-equilibrium.js';
import { cross } from '../src/cpu-geometry.js';

function mass(chart, shape) {
  const area = 4 * Math.hypot(...cross(shape.axisU, shape.axisV));
  const [u, v] = chart.knots,
    mass = new Float64Array(chart.ids.length);
  for (let j = 0; j < chart.rows; j++)
    for (let i = 0; i < chart.cols; i++) {
      const w = (area * (u[i + 1] - u[i]) * (v[j + 1] - v[j])) / 4,
        k = j * (chart.cols + 1) + i;
      for (const d of [0, 1, chart.cols + 1, chart.cols + 2]) mass[k + d] += w;
    }
  return mass;
}

export async function compareLineEquilibrium(
  renderer,
  scene,
  from,
  to,
  { angles = 16, offsetPanels = 1 } = {},
) {
  const before = await renderer.prepareScene(scene),
    { geometry, rows, firstBounce } = before;
  const chart = ({ id, side }) => geometry.maps[geometry.shapes.findIndex((s) => s.id === id)]?.[side];
  const src = chart(from),
    dest = chart(to);
  if (src?.type !== 0 || dest?.type !== 0 || src.object === dest.object)
    throw Error('Expected distinct planar surface sides');
  const start = performance.now(),
    source = geometry.shapes[src.object],
    receiver = geometry.shapes[dest.object];
  const srcMass = mass(src, source),
    dstMass = mass(dest, receiver);
  const area = dstMass.reduce((a, b) => a + b, 0);
  const domain = parallelLineDomain(geometry, src, dest);
  if (!domain) return { qualified: false };
  const directions = [source.axisU, source.axisV].map((v) => v.findIndex((x) => x !== 0));
  const aligned = [source, receiver].every((s) =>
    [s.axisU, s.axisV].every(
      (v, i) => v[directions[i]] > 0 && v.every((x, k) => k === directions[i] || x === 0),
    ),
  );
  const clear = domain.edges.length === 0 && aligned;
  const positions = (s, chart) =>
    [s.axisU, s.axisV].map((v, i) =>
      chart.knots[i].map((t) => s.center[directions[i]] + (2 * t - 1) * v[directions[i]]),
    );
  const block = clear
    ? parallelBilinearExchange(domain.h).grid(positions(source, src), positions(receiver, dest))
    : new Float64Array(src.ids.length * dest.ids.length);
  const [u, v] = dest.knots;
  let jobs = 0,
    lines = 0,
    peak = 0;
  for (let j = 0; !clear && j < dest.rows; j++)
    for (let i = 0; i < dest.cols; i++) {
      const bounds = [u[i], u[i + 1], v[j], v[j + 1]],
        op = parallelLineExchange(renderer.gl, geometry, src, dest, bounds, { moments: true });
      if (!op) return { qualified: false };
      try {
        const f = op.integrate(angles, offsetPanels),
          k = j * (dest.cols + 1) + i;
        const ids = [k, k + 1, k + dest.cols + 1, k + dest.cols + 2],
          cellArea = area * (u[i + 1] - u[i]) * (v[j + 1] - v[j]);
        for (let s = 0; s < src.ids.length; s++)
          for (let r = 0; r < 4; r++)
            block[ids[r] * src.ids.length + s] += cellArea * f.coefficients[4 * s + r];
        jobs += f.jobs;
        lines += f.lines;
        peak = Math.max(peak, f.angularPayloadBytes);
      } finally {
        op.dispose();
      }
    }
  const elapsed = performance.now() - start;
  const srcIndex = new Map(src.ids.map((id, k) => [id, k])),
    dstIndex = new Map(dest.ids.map((id, k) => [id, k]));
  let maxMass = 0;
  const parts = rows.map((p) => {
    const starts = [0],
      col = [],
      weights = [];
    for (let line = 0; line + 1 < p.starts.length; line++) {
      const id = p.first + line * p.stride,
        isDest = dstIndex.has(id),
        isSrc = srcIndex.has(id);
      let total = 0;
      for (let j = p.starts[line]; j < p.starts[line + 1]; j++) {
        const k = p.col[j];
        if ((isDest && srcIndex.has(k)) || (isSrc && dstIndex.has(k))) continue;
        col.push(k);
        weights.push(p.weights[j]);
        total += p.weights[j];
      }
      if (isDest || isSrc) {
        const row = (isDest ? dstIndex : srcIndex).get(id),
          other = isDest ? src.ids : dest.ids;
        for (let i = 0; i < other.length; i++) {
          const w = isDest
            ? block[row * src.ids.length + i] / dstMass[row]
            : block[i * src.ids.length + row] / srcMass[row];
          if (w < 0) throw Error('Negative weak exchange coefficient');
          if (w) {
            col.push(other[i]);
            weights.push(w);
            total += w;
          }
        }
      }
      maxMass = Math.max(maxMass, total);
      starts.push(col.length);
    }
    return {
      first: p.first,
      stride: p.stride,
      starts: Uint32Array.from(starts),
      col: Uint32Array.from(col),
      weights: Float64Array.from(weights),
    };
  });
  const color = Float64Array.from(geometry.receivers.flatMap((r) => geometry.shapes[r.object].color));
  const after = await modalEquilibrium(geometry, parts, firstBounce, color);
  return { qualified: true, before, after, clear, maxMass, elapsed, jobs, lines, peak, block };
}
