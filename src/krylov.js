export function equilibriumSolve(direct, apply, tolerance, x = direct.slice()) {
  const n = x.length,
    residual = new Float64Array(n);
  const directions = [],
    images = [],
    norms = [];
  let iterations = 0,
    error;
  while (iterations < 512) {
    apply(x, residual);
    error = 0;
    for (let i = 0; i < n; i++) {
      residual[i] = direct[i] - residual[i];
      error = Math.max(error, Math.abs(residual[i]));
    }
    if (error <= tolerance)
      return { incident: x, iterations, residual: error, bytes: (4 + 2 * directions.length) * x.byteLength };
    for (let k = 0; k < 8; k++) {
      const v = (directions[k] ??= new Float64Array(n)),
        a = (images[k] ??= new Float64Array(n));
      v.set(residual);
      apply(v, a);
      for (let j = 0; j < k; j++) {
        const q = images[j],
          d = directions[j];
        let product = 0;
        for (let i = 0; i < n; i++) product += q[i] * a[i];
        const beta = product / norms[j];
        for (let i = 0; i < n; i++) {
          a[i] -= beta * q[i];
          v[i] -= beta * d[i];
        }
      }
      let dot = 0,
        norm = 0;
      for (let i = 0; i < n; i++) {
        dot += a[i] * residual[i];
        norm += a[i] * a[i];
      }
      if (!(norm > 0 && Number.isFinite(norm)))
        throw Error('Diffuse equilibrium has a singular search direction.');
      norms[k] = norm;
      const alpha = dot / norm;
      error = 0;
      for (let i = 0; i < n; i++) {
        x[i] += alpha * v[i];
        residual[i] -= alpha * a[i];
        error = Math.max(error, Math.abs(residual[i]));
      }
      iterations++;
      if (error <= tolerance) break;
    }
  }
  throw Error('Diffuse equilibrium did not meet its residual tolerance.');
}
