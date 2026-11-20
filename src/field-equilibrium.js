// The Gram matrix integrates each pair of basis functions with the same
// physical area measure. Its inverse reconstructs affine fields exactly.
function areaProjection(g) {
  const n = g.supports.length,
    mass = new Float64Array(n);
  const rows = Array.from({ length: n }, () => new Map());
  for (const q of g.receivers)
    for (const [i, wi] of q.test) {
      const w = q.area * wi;
      mass[i] += w;
      for (const [j, wj] of q.test) rows[i].set(j, (rows[i].get(j) || 0) + w * wj);
    }
  // A closed visibility contact disconnects the receiver mass matrix. Each
  // connected part owns its flux and its positivity correction independently.
  const components = [],
    seen = new Uint8Array(n);
  for (let i = 0; i < n; i++)
    if (!seen[i]) {
      const ids = [i];
      seen[i] = 1;
      for (let k = 0; k < ids.length; k++)
        for (const j of rows[ids[k]].keys())
          if (!seen[j]) {
            ids.push(j);
            seen[j] = 1;
          }
      components.push(ids);
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
  function project(moments, freeze = false, useFixed = false) {
    const result = new Float64Array(3 * n),
      rhs = new Float64Array(n),
      x = new Float64Array(n),
      r = new Float64Array(n),
      p = new Float64Array(n),
      Ap = new Float64Array(n);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < n; i++) rhs[i] = moments[3 * i + c];
      x.fill(0);
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
      for (const ids of components) {
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
// Integrate the receiver-coordinate test functions before taking another
// bounce. K_ij = sum_q area_q * test_i(q) * sourceWeight_j(q).
// The cubature rows are needed once more for the final spatial residual.
function integrateReceiverModes(g, parts, count) {
  const samples = g.supports.map(() => []);
  parts.forEach((p, part) => {
    for (let row = 0; row < p.starts.length - 1; row++) {
      const q = g.receivers[p.rowIds[row]];
      for (const [i, w] of q.test) samples[i].push([part, row, q.area * w]);
    }
  });
  const active = new Uint8Array(count),
    buffer = new Float64Array(count),
    ids = new Uint32Array(count);
  const weights = [],
    columns = [];
  let bytes = active.byteLength + buffer.byteLength + ids.byteLength,
    links = 0;
  for (const samplesOfMode of samples) {
    let used = 0;
    for (const [part, row, w] of samplesOfMode) {
      const p = parts[part];
      for (let j = p.starts[row]; j < p.starts[row + 1]; j++) {
        const k = p.col[j],
          v = w * p.weights[j];
        if (v === 0) continue;
        if (!active[k]) {
          active[k] = 1;
          ids[used++] = k;
        }
        buffer[k] += v;
      }
      for (let j = p.deltaStarts[row]; j < p.deltaStarts[row + 1]; j++) {
        const k = p.deltaCols[j];
        if (!active[k]) {
          active[k] = 1;
          ids[used++] = k;
        }
        buffer[k] += w * p.deltaWeights[j];
      }
    }
    const source = ids.slice(0, used),
      weight = new Float64Array(used);
    for (let j = 0; j < used; j++) {
      const k = source[j];
      weight[j] = buffer[k];
      buffer[k] = 0;
      active[k] = 0;
    }
    columns.push(source);
    weights.push(weight);
    bytes += source.byteLength + weight.byteLength;
    links += used;
  }
  return { weights, columns, bytes, links };
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
  // The reflected term shares one trace on each side of a physical contact.
  const fieldOf = (indirect, lamp = null) => ({ geometry: g, indirect, lamp });
  const direct = fieldOf(null, g.packet);
  function observeLamp(source = -1) {
    const result = new Float64Array(3 * n);
    for (const part of parts)
      for (let row = 0; row < part.starts.length - 1; row++) {
        const q = g.receivers[part.rowIds[row]];
        for (const [i, t] of q.test) {
          const w = q.area * t;
          if (source < 0) {
            for (let c = 0; c < 3; c++)
              result[3 * i + c] += w * (part.direct[3 * row + c] + part.weakDirect[3 * row + c]);
          } else {
            const flux = w * part.lamp[row * g.shapes.length + source];
            for (let c = 0; c < 3; c++)
              result[3 * i + c] += flux * g.packet.lampRGB[c] * g.shapes[source].color[c];
          }
        }
      }
    return result;
  }
  const K = integrateReceiverModes(g, parts, g.vertices.length),
    directMoments = observeLamp();
  function emit(smooth, source) {
    g.vertices.forEach((v, i) => {
      const rho = g.shapes[v.object].color;
      for (let c = 0; c < 3; c++)
        emitted[3 * i + c] = source < 0 || v.object === source ? rho[c] * smooth[3 * i + c] : 0;
    });
  }
  function measure(input, source = -1) {
    if (input.geometry !== g) throw Error('Cannot reflect a field belonging to another pose');
    const moments = input.lamp
      ? source < 0
        ? directMoments.slice()
        : observeLamp(source)
      : new Float64Array(3 * n);
    if (!input.indirect) return moments;
    emit(input.indirect, source);
    for (let i = 0; i < n; i++) {
      let a = 0,
        b = 0,
        c = 0;
      const cols = K.columns[i],
        weights = K.weights[i];
      for (let j = 0; j < cols.length; j++) {
        const k = 3 * cols[j],
          w = weights[j];
        a += w * emitted[k];
        b += w * emitted[k + 1];
        c += w * emitted[k + 2];
      }
      moments[3 * i] += a;
      moments[3 * i + 1] += b;
      moments[3 * i + 2] += c;
    }
    return moments;
  }
  function spatialResidualSamples(incident) {
    emit(incident, -1);
    for (const part of parts)
      for (let row = 0; row < part.starts.length - 1; row++) {
        const i = 3 * part.rowIds[row];
        let a = 0,
          b = 0,
          c = 0;
        for (let j = part.starts[row]; j < part.starts[row + 1]; j++) {
          const k = 3 * part.col[j],
            w = part.weights[j];
          a += w * emitted[k];
          b += w * emitted[k + 1];
          c += w * emitted[k + 2];
        }
        light[i] = a + part.direct[3 * row];
        light[i + 1] = b + part.direct[3 * row + 1];
        light[i + 2] = c + part.direct[3 * row + 2];
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
  const sample = spatialResidualSamples(incident);
  P.project(measure(fieldOf(incident, g.packet)), true);
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
    modeLinks: K.links,
    bytes: K.bytes + P.bytes + 2 * light.byteLength + 5 * incident.byteLength + n * 24,
  };
}
