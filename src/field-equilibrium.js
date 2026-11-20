// The Gram matrix integrates each pair of basis functions with the same
// physical area measure. Its inverse reconstructs affine fields exactly.
function areaProjection(g) {
  const n = g.supports.length,
    mass = new Float64Array(n);
  const rows = Array.from({ length: n }, () => new Map()),
    components = new Map();
  g.supports.forEach(({ component }, i) => {
    if (!components.has(component)) components.set(component, []);
    components.get(component).push(i);
  });
  for (const q of g.receivers)
    for (const [i, wi] of q.test) {
      const w = q.area * wi;
      mass[i] += w;
      for (const [j, wj] of q.test) rows[i].set(j, (rows[i].get(j) || 0) + w * wj);
    }
  const start = new Uint32Array(n + 1);
  for (let i = 0; i < n; i++) start[i + 1] = start[i] + rows[i].size;
  const cols = new Uint32Array(start[n]),
    values = new Float64Array(cols.length),
    diag = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let k = start[i];
    for (const [j, w] of rows[i]) {
      cols[k] = j;
      values[k++] = w / Math.sqrt(mass[i] * mass[j]);
      if (i === j) diag[i] = w / mass[i];
    }
    if (!(diag[i] > 0)) throw Error('A field mode has no measured support');
  }
  const fixed = new Uint8Array(3 * n);
  let bytes =
    mass.byteLength +
    start.byteLength +
    cols.byteLength +
    values.byteLength +
    diag.byteLength +
    fixed.byteLength;
  function project(samples, freeze = false, useFixed = false) {
    const result = new Float64Array(3 * n),
      rhs = new Float64Array(n),
      x = new Float64Array(n),
      r = new Float64Array(n),
      p = new Float64Array(n),
      Ap = new Float64Array(n);
    for (let c = 0; c < 3; c++) {
      rhs.fill(0);
      x.fill(0);
      g.receivers.forEach((q, k) => {
        const v = q.area * samples[3 * k + c];
        for (const [i, w] of q.test) rhs[i] += w * v;
      });
      let norm = 0;
      for (let i = 0; i < n; i++) {
        r[i] = rhs[i] / Math.sqrt(mass[i]);
        p[i] = r[i] / diag[i];
        norm += r[i] * p[i];
      }
      const tol = 1e-24 * norm;
      for (let j = 0; j < 120 && norm > tol; j++) {
        let den = 0;
        for (let i = 0; i < n; i++) {
          let sum = 0;
          for (let k = start[i]; k < start[i + 1]; k++) sum += values[k] * p[cols[k]];
          Ap[i] = sum;
          den += sum * p[i];
        }
        const step = norm / den;
        let next = 0;
        for (let i = 0; i < n; i++) {
          x[i] += step * p[i];
          r[i] -= step * Ap[i];
          next += (r[i] * r[i]) / diag[i];
        }
        for (let i = 0; i < n; i++) p[i] = r[i] / diag[i] + (next / norm) * p[i];
        norm = next;
      }
      if (!Number.isFinite(norm) || norm > tol) throw Error('Surface mass solve exceeded its tolerance');
      for (let i = 0; i < n; i++) x[i] /= Math.sqrt(mass[i]);
      for (const ids of components.values()) {
        let flux = 0,
          minimum = Infinity;
        for (const i of ids) {
          flux += rhs[i];
          minimum = Math.min(minimum, x[i]);
        }
        let shift = 0;
        if (useFixed) {
          let total = 0,
            area = 0;
          for (const i of ids)
            if (fixed[3 * i + c]) {
              total += mass[i] * x[i];
              area += mass[i];
            }
          if (area) shift = (total - flux) / area;
        } else if (minimum < 0) {
          for (let k = 0; k <= ids.length; k++) {
            let total = 0,
              area = 0;
            for (const i of ids)
              if (x[i] > shift) {
                total += mass[i] * x[i];
                area += mass[i];
              }
            if (!area) break;
            const next = (total - flux) / area;
            if (next <= shift + 1e-14 * (1 + Math.abs(shift))) {
              shift = Math.max(shift, next);
              break;
            }
            shift = next;
          }
        }
        for (const i of ids) {
          const active = useFixed ? fixed[3 * i + c] : x[i] >= shift;
          if (freeze) fixed[3 * i + c] = active;
          result[3 * i + c] = active ? x[i] - shift : 0;
        }
      }
    }
    return result;
  }
  return { project, bytes: bytes + 8 * n * 5, mass };
}
export function fieldEquilibrium(g, I, tolerance = 0.000025) {
  const P = areaProjection(g),
    { parts } = I,
    n = g.supports.length;
  const light = new Float64Array(3 * g.receivers.length),
    emitted = new Float64Array(3 * g.vertices.length);
  function vertexValues(field) {
    const result = new Float64Array(3 * g.vertices.length);
    g.vertices.forEach((v, j) => {
      for (const [i, w] of v.basis) for (let c = 0; c < 3; c++) result[3 * j + c] += w * field[3 * i + c];
    });
    return result;
  }
  // An incident field has two additive terms. The direct term retains its source
  // and the geometry defining its visible support, even across a shadow edge.
  // The reflected term is a continuous function on the authored surface.
  const fieldOf = (indirect, lamp = null) => ({ geometry: g, indirect, lamp });
  const direct = fieldOf(null, g.packet);
  function measure(input, source = -1) {
    if (input.geometry !== g) throw Error('Cannot reflect a field belonging to another pose');
    const smooth = input.indirect,
      lamp = input.lamp;
    light.fill(0);
    if (smooth)
      g.vertices.forEach((v, i) => {
        const rho = g.shapes[v.object].color;
        for (let c = 0; c < 3; c++)
          emitted[3 * i + c] = v.object === source || source < 0 ? rho[c] * smooth[3 * i + c] : 0;
      });
    for (const part of parts)
      for (let r = 0; r + 1 < part.starts.length; r++) {
        const i = 3 * (part.first + r * part.stride);
        if (lamp) {
          if (source < 0) {
            for (let c = 0; c < 3; c++) light[i + c] = part.direct[3 * r + c];
          } else {
            const w = part.lamp[r * g.shapes.length + source],
              rho = g.shapes[source].color;
            for (let c = 0; c < 3; c++) light[i + c] = w * g.packet.lampRGB[c] * rho[c];
          }
        }
        if (!smooth) continue;
        let a = 0,
          b = 0,
          c = 0;
        for (let j = part.starts[r]; j < part.starts[r + 1]; j++) {
          const k = 3 * part.col[j],
            w = part.weights[j];
          a += w * emitted[k];
          b += w * emitted[k + 1];
          c += w * emitted[k + 2];
        }
        light[i] += a;
        light[i + 1] += b;
        light[i + 2] += c;
      }
    return light;
  }
  let field = P.project(measure(direct)),
    iterations = 0,
    residual = Infinity,
    incident;
  for (; iterations < 180; iterations++) {
    incident = vertexValues(field);
    const next = P.project(measure(fieldOf(incident, g.packet)));
    residual = 0;
    for (let i = 0; i < field.length; i++) residual = Math.max(residual, Math.abs(next[i] - field[i]));
    if (!Number.isFinite(residual)) throw Error('Nonfinite diffuse equilibrium');
    if (residual <= tolerance) break;
    field = next;
  }
  if (residual > tolerance) throw Error('Diffuse equilibrium did not converge: ' + residual);
  // Freeze the active positive modes of the settled solution. All source
  // decompositions now use ONE linear, energy-preserving receiver projection.
  const sample = light.slice();
  P.project(sample, true);
  function reflect(input, source = -1) {
    return fieldOf(vertexValues(P.project(measure(input, source), false, true)));
  }
  const illumination = fieldOf(incident, g.packet),
    first = reflect(direct),
    again = reflect(fieldOf(incident));
  let closure = 0;
  for (let i = 0; i < incident.length; i++)
    closure = Math.max(closure, Math.abs(first.indirect[i] + again.indirect[i] - incident[i]));
  return {
    illumination,
    first,
    direct,
    light: sample,
    iterations: iterations + 1,
    residual,
    closure,
    reflect,
    bytes: P.bytes + 2 * light.byteLength + 5 * incident.byteLength + n * 24,
  };
}
