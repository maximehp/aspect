import { inverseTangentCoefficients } from './inverse-tangent.js';
// Weak diffuse transfer between aligned parallel rectangles with clear mutual
// visibility. Units are metres. Axes are coordinates in one common orthonormal
// plane. Both radiance and receiver test functions may be bilinear.
const factorial = [1, 1, 2, 6];

// Ti2(z) = integral from 0 to z of atan(t)/t dt. For 0 <= z <= 1,
// expand 1/(1+z²s²) in q(1-s²), q=z²/(1+z²). The nth coefficient
// integrates -log(s)(1-s²)^n on [0,1] and is at most one. The remaining
// absolute error after n terms is at most z*q^n, with q <= 1/2.
function inverseTangentIntegral(tolerance) {
  const count = Math.max(1, 1 + Math.ceil(-Math.log2(tolerance)));
  const coefficients = inverseTangentCoefficients(count);
  return (z) => {
    const d = 1 + z * z,
      q = (z * z) / d;
    let sum = 0;
    for (let i = count - 1; i >= 0; i--) sum = coefficients[i] + q * sum;
    return (z * sum) / d;
  };
}

// Substitution s=h sinh(t) leaves atan((y/h)/cosh(t)). Factoring
// its tangent as a difference of two tangents gives the two Ti2 values.
function logMoment(x, y, h, ti2) {
  const X = Math.asinh(Math.abs(x) / h),
    Y = Math.asinh(Math.abs(y) / h);
  return (
    Math.sign(x) *
    Math.sign(y) *
    ((Math.PI / 2) * Math.min(X, Y) - ti2(Math.exp(-Math.abs(X - Y))) + ti2(Math.exp(-X - Y)))
  );
}

function edgeMoments(t, a) {
  const b0 = Math.atan(t / a) / a,
    b1 = Math.log1p((t / a) ** 2) / 2;
  return [b0, b1, t - a * a * b0, (t * t) / 2 - a * a * b1];
}

function primitives(x, y, h, ti2) {
  const out = new Float64Array(9);
  if (!x || !y) return out;
  const h2 = h * h;
  const bx = edgeMoments(y, Math.hypot(h, x)),
    by = edgeMoments(x, Math.hypot(h, y));
  const bx0 = edgeMoments(y, h),
    by0 = edgeMoments(x, h);
  const px = [1, x, x * x, x * x * x],
    py = [1, y, y * y, y * y * y];
  const M = new Float64Array(16),
    N = new Float64Array(16);
  // M_ij integrates s^i t^j / D², N_ij integrates s^i t^j / D,
  // D=h²+s²+t². Integration by parts supplies M. A divergence identity
  // supplies N from the two outer edges whenever i+j is nonzero.
  for (let i = 0; i <= 3; i++) {
    for (let j = 0; j <= 3; j++) {
      const k = 4 * i + j;
      if (!k) {
        M[k] = (x * bx[0] + y * by[0]) / (2 * h2);
        N[k] = logMoment(x, y, h, ti2);
      } else {
        if (i) M[k] = ((i === 1 ? bx0[j] : (i - 1) * N[k - 8]) - px[i - 1] * bx[j]) / 2;
        else M[k] = ((j === 1 ? by0[i] : (j - 1) * N[k - 2]) - py[j - 1] * by[i]) / 2;
        N[k] = (x * px[i] * bx[j] + y * py[j] * by[i] - 2 * h2 * M[k]) / (i + j);
      }
      const value = (((((i + j) % 2 ? -1 : 1) * h2) / Math.PI) * M[k]) / (factorial[i] * factorial[j]);
      for (let a = Math.max(1, i); a <= 3; a++)
        for (let b = Math.max(1, j); b <= 3; b++)
          out[3 * (a - 1) + b - 1] += (value * px[a - i] * py[b - j]) / (factorial[a - i] * factorial[b - j]);
    }
  }
  return out;
}

// Evaluate each distinct endpoint displacement once per grid. The integrator
// retains no primitives or visibility decisions from a preceding pose.
export function parallelBilinearExchange(gap, primitiveTolerance = 1e-12) {
  if (!(gap > 0 && Number.isFinite(gap) && primitiveTolerance > 0 && Number.isFinite(primitiveTolerance)))
    throw RangeError('Expected a finite positive gap and integration tolerance');
  const ti2 = inverseTangentIntegral(primitiveTolerance);
  let evaluations = 0,
    recovered = 0,
    sumRecoveryError = 0;
  function grid(source, receiver, recoveryTolerance = 1e-8) {
    if (source.length !== 2 || receiver.length !== 2) throw RangeError('Expected two plane coordinates');
    if (!(Number.isFinite(recoveryTolerance) && recoveryTolerance > 0))
      throw RangeError('Expected a positive absolute recovery tolerance in square metres');
    for (const knots of [...source, ...receiver])
      if (knots.length < 2 || !knots.every((x, i) => Number.isFinite(x) && (!i || x > knots[i - 1])))
        throw RangeError('Expected increasing finite plane coordinates');
    const [sx, sy] = source,
      [rx, ry] = receiver;
    const nx = sx.length * rx.length,
      ny = sy.length * ry.length;
    const intermediate = new Float64Array(3 * nx * ny);
    const along = new Float64Array(9 * nx),
      across = new Float64Array(3 * ny);
    const xResult = new Float64Array(3 * nx),
      yResult = new Float64Array(ny);
    const coefficients = new Float64Array(nx * ny);
    function offsets(s, r) {
      const lookup = new Map();
      for (let j = 0; j < r.length; j++)
        for (let i = 0; i < s.length; i++) {
          const d = r[j] - s[i];
          if (!lookup.has(d)) lookup.set(d, []);
          lookup.get(d).push(j * s.length + i);
        }
      return lookup;
    }
    const xs = offsets(sx, rx),
      ys = offsets(sy, ry);
    for (const [y, destinations] of ys) {
      for (const [x, columns] of xs) {
        const value = primitives(x, y, gap, ti2);
        evaluations++;
        for (const column of columns) along.set(value, 9 * column);
      }
      contractAxis(sx, rx, along, 3, xResult);
      for (const dest of destinations) intermediate.set(xResult, 3 * nx * dest);
    }
    const bx = supportBounds(sx, rx),
      by = supportBounds(sy, ry),
      h2 = gap * gap;
    for (let x = 0; x < nx; x++) {
      for (let k = 0; k < ny; k++)
        for (let c = 0; c < 3; c++) across[3 * k + c] = intermediate[3 * (k * nx + x) + c];
      contractAxis(sy, ry, across, 1, yResult);
      const s = x % sx.length,
        r = (x - s) / sx.length;
      for (let j = 0; j < ry.length; j++)
        for (let i = 0; i < sy.length; i++) {
          const y = j * sy.length + i;
          let value = yResult[y];
          const A = (bx[3 * x + 2] * by[3 * y + 2] * h2) / Math.PI;
          const lo = A / (h2 + bx[3 * x + 1] + by[3 * y + 1]) ** 2;
          const hi = A / (h2 + bx[3 * x] + by[3 * y]) ** 2;
          if (!(lo <= value && value <= hi)) {
            const error = (hi - lo) / 2;
            if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(error <= recoveryTolerance))
              throw RangeError('Endpoint cancellation exceeds the recovery tolerance');
            value = lo + error;
            recovered++;
            sumRecoveryError += error;
          }
          coefficients[((j * rx.length + r) * sy.length + i) * sx.length + s] = value;
        }
    }
    return coefficients;
  }
  return {
    grid,
    get recovered() {
      return recovered;
    },
    get sumRecoveryError() {
      return sumRecoveryError;
    },
    get evaluations() {
      return evaluations;
    },
  };
}

// Projected supports give the smallest and largest possible squared offset
// and the exact product of the two integrated 1D hats. These bound a
// coefficient independently of cancellation in endpoint differences.
function supportBounds(source, receiver) {
  const result = new Float64Array(3 * source.length * receiver.length);
  for (let j = 0; j < receiver.length; j++)
    for (let i = 0; i < source.length; i++) {
      const a = source[Math.max(0, i - 1)],
        b = source[Math.min(i + 1, source.length - 1)];
      const c = receiver[Math.max(0, j - 1)],
        d = receiver[Math.min(j + 1, receiver.length - 1)];
      const k = 3 * (j * source.length + i);
      result[k] = Math.max(0, a - d, c - b) ** 2;
      result[k + 1] = Math.max(Math.abs(a - d), Math.abs(c - b)) ** 2;
      result[k + 2] = ((b - a) * (d - c)) / 4;
    }
  return result;
}

// Twice integrating a piecewise-linear hat by parts leaves its value jump V
// (sheet edges only) and its slope jump D at each knot. Apply to every basis
// at once. The three primitive orders have coefficients Vs Vr,
// Ds Vr - Vs Dr, and -Ds Dr. Adjacent elements cancel their shared endpoints.
function contractAxis(s, r, data, channels, out) {
  const ns = s.length,
    nr = r.length,
    stride = 3 * channels;
  const left = new Float64Array(ns * nr * channels),
    right = new Float64Array(left.length);
  for (let j = 0; j < nr; j++)
    for (let i = 0; i < ns; i++) {
      const base = stride * (j * ns + i),
        end = i === 0 ? -1 : Number(i === ns - 1);
      for (let k = 0; k < channels; k++) {
        const q = base + channels + k;
        let d3 = 0,
          d4 = 0;
        if (i) {
          const step = s[i] - s[i - 1];
          d3 += (data[q] - data[q - stride]) / step;
          d4 += (data[q + channels] - data[q + channels - stride]) / step;
        }
        if (i + 1 < ns) {
          const step = s[i + 1] - s[i];
          d3 += (data[q] - data[q + stride]) / step;
          d4 += (data[q + channels] - data[q + channels + stride]) / step;
        }
        const p = channels * (j * ns + i) + k;
        left[p] = end * data[base + k] + d3;
        right[p] = -end * data[q] - d4;
      }
    }
  const step = ns * channels;
  for (let j = 0; j < nr; j++)
    for (let k = 0; k < step; k++) {
      const p = j * step + k;
      let v = (j === 0 ? -1 : Number(j === nr - 1)) * left[p];
      if (j) v += (right[p] - right[p - step]) / (r[j] - r[j - 1]);
      if (j + 1 < nr) v += (right[p] - right[p + step]) / (r[j + 1] - r[j]);
      out[p] = v;
    }
}
