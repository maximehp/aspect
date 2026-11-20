// A lifting step splits a piecewise-linear field into a coarser interpolant and
// a detail function. Updating the two neighboring coarse functions gives each
// detail exactly zero integral and first moment on the authored interval.
function axisBasis(knots) {
  let indices = knots.map((_, i) => i);
  const stages = [];
  const hat = (ids, j) => {
    const c = knots[ids[j]],
      l = knots[ids[Math.max(0, j - 1)]],
      r = knots[ids[Math.min(ids.length - 1, j + 1)]];
    return { area: (r - l) / 2, centroid: (l + c + r) / 3 };
  };
  while (indices.length > 2) {
    const kept = indices.filter((_, i) => i % 2 === 0 || i === indices.length - 1),
      predict = [],
      update = [];
    for (let j = 1; j + 1 < indices.length; j += 2) {
      const i = indices[j],
        l = indices[j - 1],
        r = indices[j + 1],
        t = (knots[i] - knots[l]) / (knots[r] - knots[l]);
      const fine = hat(indices, j),
        left = hat(kept, (j - 1) / 2),
        right = hat(kept, (j + 1) / 2);
      const f = (fine.centroid - left.centroid) / (right.centroid - left.centroid);
      predict.push({ i, l, r, a: 1 - t, b: t, update: 0 });
      update.push({
        i,
        l,
        r,
        a: (fine.area * (1 - f)) / left.area,
        b: (fine.area * f) / right.area,
        update: 1,
      });
    }
    stages.push(...predict, ...update);
    indices = kept;
  }
  // Exact induced row norms of the small 1D transforms. Tensor norms factor.
  const m = knots.length,
    synthesis = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    const v = new Float64Array(m);
    v[j] = 1;
    lift(v, stages, 1, true);
    for (let i = 0; i < m; i++) synthesis[i] += Math.abs(v[i]);
  }
  return { stages, synthesis };
}
function lift(values, steps, channels, inverse) {
  for (let z = 0; z < steps.length; z++) {
    const { i, l, r, a, b, update } = steps[inverse ? steps.length - 1 - z : z];
    for (let c = 0; c < channels; c++) {
      const m = channels * i + c,
        L = channels * l + c,
        R = channels * r + c;
      if (update) {
        const d = inverse ? -values[m] : values[m];
        values[L] += a * d;
        values[R] += b * d;
      } else values[m] += (inverse ? 1 : -1) * (a * values[L] + b * values[R]);
    }
  }
  return values;
}
// Evaluate the synthesized chart at arbitrary coordinates by lifting the
// interpolation functional. Only modes with support at the requested point enter.
function weightsAt(knots, t, steps) {
  let lo = 0,
    hi = knots.length - 1;
  while (hi - lo > 1) {
    const m = (lo + hi) >> 1;
    if (t < knots[m]) hi = m;
    else lo = m;
  }
  const z = Math.max(0, Math.min(1, (t - knots[lo]) / (knots[hi] - knots[lo])));
  const weights = new Float64Array(knots.length);
  weights[lo] = 1 - z;
  weights[hi] = z;
  for (const { i, l, r, update, a, b } of steps) {
    if (update) weights[i] -= a * weights[l] + b * weights[r];
    else {
      weights[l] += a * weights[i];
      weights[r] += b * weights[i];
    }
  }
  return weights;
}
export function surfaceModes(geometry) {
  const axes = new Map(),
    groups = [],
    steps = [],
    n = geometry.receivers.length;
  let maxSynthesis = 1,
    first = 0;
  for (const chart of geometry.charts) {
    if (chart.ids[0] < 0 || geometry.shapes[chart.object].axes) continue;
    const [xs, ys] = chart.knots.map(axisBasis),
      ids = chart.ids,
      nx = chart.cols + 1,
      ny = chart.rows + 1,
      begin = steps.length;
    axes.set(chart, [xs.stages, ys.stages]);
    if (ids[0] !== first || ids.at(-1) !== first + nx * ny - 1) throw Error('Noncontiguous sheet basis');
    const insert = (s, index) => steps.push({ ...s, i: index(s.i), l: index(s.l), r: index(s.r) });
    for (let j = 0; j < ny; j++) for (const s of xs.stages) insert(s, (i) => first + j * nx + i);
    for (let i = 0; i < nx; i++) for (const s of ys.stages) insert(s, (j) => first + j * nx + i);
    maxSynthesis = Math.max(maxSynthesis, Math.max(...xs.synthesis) * Math.max(...ys.synthesis));
    groups.push({ first, length: nx * ny, begin, end: steps.length });
    first += nx * ny;
  }
  // Fillet charts share boundary DOFs, including poles. Their original
  // continuous nodal functions remain the local basis on those surfaces.
  while (first < n) {
    const length = Math.min(64, n - first);
    groups.push({ first, length, begin: steps.length, end: steps.length });
    first += length;
  }
  const packed = new ArrayBuffer(32 * steps.length);
  const ints = new Int32Array(packed),
    doubles = new Float64Array(packed);
  steps.forEach(({ i, l, r, update, a, b }, k) => {
    ints.set([i, l, r, update], 8 * k);
    doubles.set([a, b], 4 * k + 2);
  });
  function at(chart, u, v, coefficients) {
    const rgb = [0, 0, 0];
    if (chart.ids[0] < 0) return rgb;
    const [xSteps, ySteps] = axes.get(chart) || [[], []];
    const x = weightsAt(chart.knots[0], u, xSteps),
      y = weightsAt(chart.knots[1], v, ySteps);
    for (let j = 0; j < y.length; j++)
      if (y[j])
        for (let i = 0; i < x.length; i++)
          if (x[i]) {
            const k = 3 * chart.ids[j * x.length + i],
              w = x[i] * y[j];
            for (let c = 0; c < 3; c++) rgb[c] += w * coefficients[k + c];
          }
    return rgb;
  }
  return {
    groups,
    packed,
    steps,
    maxSynthesis,
    at,
    encode: (v) => lift(v.slice(), steps, 3, false),
    decode: (v) => lift(v.slice(), steps, 3, true),
  };
}
