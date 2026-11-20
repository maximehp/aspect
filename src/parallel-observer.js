// On two planes the emitting cosine numerator is one affine function of the
// observer, shared by every point on the source. Interpolate the remaining
// positive field only across an entirely visible, mutually facing domain.
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
function axis(n, affine) {
  const nodes = Array.from({ length: n }, (_, i) => Math.cos((Math.PI * (i + 0.5)) / n));
  const barycentric = nodes.map((_, i) => (i % 2 ? -1 : 1) * Math.sin((Math.PI * (i + 0.5)) / n));
  const mean = (k) => (k % 2 ? 0 : 1 / (1 - k * k));
  const first = (k) => (mean(k - 1) + mean(k + 1)) / 2;
  function moments(linear) {
    return [-1, 1].map((sign) =>
      nodes.map((_, i) => {
        let weight = linear ? sign / 6 : 0.5;
        for (let k = 1; k < n; k++) {
          const moment = linear
            ? first(k) + (sign * (mean(k - 2) + 2 * mean(k) + mean(k + 2))) / 4
            : mean(k) + sign * first(k);
          weight += moment * Math.cos((Math.PI * k * (i + 0.5)) / n);
        }
        return weight / n;
      }),
    );
  }
  return {
    nodes,
    hat: moments(false),
    linearHat: affine && moments(true),
    weights(t) {
      const index = nodes.indexOf(t);
      if (index >= 0) return nodes.map((_, i) => Number(i === index));
      const terms = nodes.map((x, i) => barycentric[i] / (t - x)),
        sum = terms.reduce((a, b) => a + b);
      return terms.map((x) => x / sum);
    },
  };
}
export function planarObserver(source, target, tolerance, maxQueries) {
  if (!(tolerance > 0 && Number.isFinite(tolerance)) || !Number.isSafeInteger(maxQueries) || maxQueries < 1)
    throw RangeError('Expected a positive tolerance and observation count');
  const delta = sub(target.center, source.center);
  const halfSpan = (region, axis) => Math.abs(dot(region.axisU, axis)) + Math.abs(dot(region.axisV, axis));
  const height = dot(delta, source.n);
  const nearestEmitting = height - halfSpan(target, source.n),
    nearestReceiving = -dot(delta, target.n) - halfSpan(source, target.n);
  const distance = Math.hypot(...delta);
  const margin =
    1e-12 *
    (1 +
      distance +
      Math.hypot(...source.axisU) +
      Math.hypot(...source.axisV) +
      Math.hypot(...target.axisU) +
      Math.hypot(...target.axisV));
  if (!(nearestEmitting > margin && nearestReceiving > margin)) return null;
  const gap =
    Math.max(
      nearestEmitting,
      nearestReceiving,
      distance - (halfSpan(source, delta) + halfSpan(target, delta)) / distance,
    ) - margin;
  const gu = dot(target.axisU, source.n) / height,
    gv = dot(target.axisV, source.n) / height;
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
  const a = axis(orders[0], gu !== 0),
    b = axis(orders[1], gv !== 0);
  const nodes = b.nodes.flatMap((v) =>
    a.nodes.map((u) => target.center.map((x, k) => x + u * target.axisU[k] + v * target.axisV[k])),
  );
  const inverse = (gu || gv) && b.nodes.flatMap((v) => a.nodes.map((u) => 1 / (1 + gu * u + gv * v)));
  function contract(samples, U, V, gain = 1, dU, dV) {
    if (samples.length !== nodes.length) throw RangeError('Expected one sample per interpolation node');
    const value = new Float64Array(samples[0].length);
    for (let j = 0; j < V.length; j++)
      for (let i = 0; i < U.length; i++) {
        const k = i + j * U.length,
          sample = samples[k];
        let weight = U[i] * V[j];
        if (inverse) {
          if (dU) weight += gu * dU[i] * V[j];
          if (dV) weight += gv * U[i] * dV[j];
          weight *= inverse[k] * gain;
        }
        for (let c = 0; c < value.length; c++) value[c] += weight * sample[c];
      }
    return value;
  }
  return {
    nodes,
    orders,
    relativeBound,
    minimumDistance: gap,
    at(samples, u, v) {
      if (!(Math.abs(u) <= 1 && Math.abs(v) <= 1))
        throw RangeError('Coordinates outside the observer domain');
      return contract(samples, a.weights(u), b.weights(v), 1 + gu * u + gv * v);
    },
    moments(samples) {
      return b.hat.flatMap((V, j) =>
        a.hat.map((U, i) => contract(samples, U, V, 1, a.linearHat?.[i], b.linearHat?.[j])),
      );
    },
  };
}

export function parallelObserver(source, target, tolerance, maxQueries) {
  if (cross(source.n, target.n).some((x) => x !== 0)) return null;
  return planarObserver(source, target, tolerance, maxQueries);
}
