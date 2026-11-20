// On mutually visible parallel planes the numerator of the diffuse kernel
// is constant. These bounds hold for any nonnegative incoming floor field,
// including fields with sharp shadows. Coordinates in the observer use [-1,1].
import { cross, dot, sub } from './cpu-geometry.js';
import { sheetVertices } from './lamp-visibility.js';

function tail(n, step, scale) {
  let best = Infinity;
  for (let rho = 1.25; ; rho *= 1.25) {
    const separation = 1 - ((rho + 1 / rho) * step) / 2;
    if (separation <= 0 || !Number.isFinite(rho)) return best;
    best = Math.min(best, (4 * scale) / (rho ** n * (1 - 1 / rho) * separation ** 4));
  }
}
function axis(n) {
  const nodes = Array.from({ length: n }, (_, i) => Math.cos((Math.PI * (i + 0.5)) / n));
  const barycentric = nodes.map((_, i) => (i % 2 ? -1 : 1) * Math.sin((Math.PI * (i + 0.5)) / n));
  const mean = (k) => (k % 2 ? 0 : 1 / (1 - k * k));
  const hat = [-1, 1].map((sign) =>
    nodes.map((_, i) => {
      let weight = 0.5;
      for (let k = 1; k < n; k++) {
        const moment = mean(k) + (sign * (mean(k - 1) + mean(k + 1))) / 2;
        weight += moment * Math.cos((Math.PI * k * (i + 0.5)) / n);
      }
      return weight / n;
    }),
  );
  return {
    nodes,
    hat,
    weights(t) {
      const index = nodes.indexOf(t);
      if (index >= 0) return nodes.map((_, i) => Number(i === index));
      const terms = nodes.map((x, i) => barycentric[i] / (t - x)),
        sum = terms.reduce((a, b) => a + b);
      return terms.map((x) => x / sum);
    },
  };
}
export function parallelObserver(source, target, tolerance, maxQueries) {
  if (!(tolerance > 0 && Number.isFinite(tolerance)) || !Number.isSafeInteger(maxQueries) || maxQueries < 1)
    throw RangeError('Expected a positive tolerance and observation count');
  if (cross(source.n, target.n).some((x) => x !== 0) || dot(source.n, target.n) >= 0) return null;
  const gap = dot(sub(target.center, source.center), source.n);
  if (!(gap > 0)) return null;
  let far = 0;
  for (const x of sheetVertices(source))
    for (const y of sheetVertices(target)) far = Math.max(far, Math.hypot(...sub(y, x)));
  const spans = [target.axisU, target.axisV].map((v) => Math.hypot(...v) / gap),
    scale = (far / gap) ** 4;
  if (spans.some((x) => !(x > 0 && Number.isFinite(x)))) return null;
  // The tensor interpolation error is bounded by E_u + Lambda_u E_v.
  // DCT cardinal functions give Lambda_u <= 2*n_u-1.
  const extent = Math.min(64, maxQueries);
  const bounds = spans.map((step) => Array.from({ length: extent }, (_, i) => tail(i + 1, step, scale)));
  let cost = maxQueries,
    orders,
    relativeBound;
  for (let u = 1; u <= extent; u++)
    for (let v = 1; v <= extent && u * v < cost; v++) {
      const bound = bounds[0][u - 1] + (2 * u - 1) * bounds[1][v - 1];
      if (bound <= tolerance) {
        cost = u * v;
        orders = [u, v];
        relativeBound = bound;
        break;
      }
    }
  if (!orders) return null;
  const [a, b] = orders.map(axis);
  const nodes = b.nodes.flatMap((v) =>
    a.nodes.map((u) => target.center.map((x, k) => x + u * target.axisU[k] + v * target.axisV[k])),
  );
  function contract(samples, U, V) {
    const value = new Float64Array(samples[0].length);
    for (let j = 0; j < V.length; j++)
      for (let i = 0; i < U.length; i++) {
        const w = U[i] * V[j],
          sample = samples[i + j * U.length];
        for (let k = 0; k < value.length; k++) value[k] += w * sample[k];
      }
    return value;
  }
  return {
    nodes,
    orders,
    relativeBound,
    at(samples, u, v) {
      if (samples.length !== nodes.length || !(Math.abs(u) <= 1 && Math.abs(v) <= 1))
        throw RangeError('Samples or coordinates outside the observer domain');
      return contract(samples, a.weights(u), b.weights(v));
    },
    moments(samples) {
      if (samples.length !== nodes.length) throw RangeError('Expected one sample per interpolation node');
      return b.hat.flatMap((V) => a.hat.map((U) => contract(samples, U, V)));
    },
  };
}
