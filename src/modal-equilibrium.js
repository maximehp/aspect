import { surfaceModes } from './surface-modes.js';
import { equilibriumSolve } from './krylov.js';

const module = WebAssembly.compileStreaming(fetch(new URL('./modal-operator.wasm', import.meta.url)));

export async function modalEquilibrium(geometry, parts, direct, color) {
  const start = performance.now(),
    basis = surfaceModes(geometry),
    n = geometry.receivers.length;
  const { exports: w } = await WebAssembly.instantiate(await module);
  const mem = w.memory;
  let top = Number(w.__heap_base.value);
  function alloc(bytes) {
    const p = Math.ceil(top / 8) * 8;
    top = p + bytes;
    return p;
  }
  function reserve(end) {
    const pages = Math.ceil((end - mem.buffer.byteLength) / 65536);
    if (pages > 0) mem.grow(pages);
  }
  const D = basis.encode(direct);
  const amplitude = new Float64Array(n);
  for (let i = 0; i < n; i++)
    amplitude[i] = Math.max(0.005, Math.abs(D[3 * i]), Math.abs(D[3 * i + 1]), Math.abs(D[3 * i + 2]));
  const steps = alloc(basis.packed.byteLength),
    norm = alloc(8 * n),
    rows = alloc(4 * (n + 1));
  const block = alloc(8 * n * Math.max(...basis.groups.map((g) => g.length))),
    discard = alloc(8);
  const activePtr = alloc(n),
    subset = alloc(4 * (n + 1)),
    source = alloc(24 * n),
    result = alloc(24 * n);
  const links = alloc(0);
  reserve(top);
  new Uint8Array(mem.buffer, steps, basis.packed.byteLength).set(new Uint8Array(basis.packed));
  new Float64Array(mem.buffer, norm, n).set(amplitude);
  let full = 0;
  // Bound omitted weights in the amplitude-weighted modal norm.
  const budget = 0.0001;
  for (const g of basis.groups) {
    reserve(links + 8 * (full + g.length * n));
    const matrix = new Float64Array(mem.buffer, block, g.length * n);
    matrix.fill(0);
    for (let r = 0; r < g.length; r++) {
      const i = g.first + r,
        p = parts[i % parts.length],
        k = (i - p.first) / p.stride,
        base = r * n;
      for (let j = p.starts[k]; j < p.starts[k + 1]; j++) matrix[base + p.col[j]] += p.weights[j];
    }
    full = w.compileModes(
      block,
      g.first,
      g.length,
      n,
      steps,
      g.begin,
      g.end,
      basis.steps.length,
      norm,
      budget,
      links,
      rows,
      full,
      discard,
    );
  }
  top = links + 8 * full;
  const compact = alloc(8 * full);
  reserve(top);
  const input = new Float64Array(mem.buffer, source, 3 * n),
    output = new Float64Array(mem.buffer, result, 3 * n);
  const active = new Uint8Array(mem.buffer, activePtr, n),
    rhs = D.slice();
  const counts = {
    fullLinks: full,
    activeLinks: 0,
    activeModes: 0,
    allModes: n,
    applications: 0,
    checks: 0,
    linkReads: 0,
    receiverRowReads: 0,
    reconstructions: 0,
  };
  const tolerance = 0.00015;
  for (let i = 0; i < n; i++)
    active[i] = Math.max(Math.abs(D[3 * i]), Math.abs(D[3 * i + 1]), Math.abs(D[3 * i + 2])) > tolerance;
  const buildMs = performance.now() - start;
  function apply(sourceField, out, restricted = true) {
    input.set(sourceField);
    w.applyModes(n, restricted ? subset : rows, restricted ? compact : links, source, result);
    out.set(output);
  }
  const state = new Float64Array(3 * n);
  let iterations = 0,
    peakBytes = 0,
    error = 0;
  function transferState(v, restricted) {
    for (let j = 0; j < 3 * n; j++) input[j] = color[j] * v[j];
    counts.linkReads += w.applyModes(
      n,
      restricted ? subset : rows,
      restricted ? compact : links,
      source,
      result,
    );
  }
  // Two current-pose applications reveal coefficient support reached indirectly.
  // They read the same modal map as subsequent bounces.
  state.set(D);
  for (let k = 0; k < 2; k++) {
    transferState(state, false);
    counts.checks++;
    for (let i = 0; i < n; i++) {
      for (let c = 0; c < 3; c++) {
        const j = 3 * i + c;
        state[j] = D[j] + output[j];
      }
      if (
        Math.max(Math.abs(state[3 * i]), Math.abs(state[3 * i + 1]), Math.abs(state[3 * i + 2])) >
        tolerance / 2
      )
        active[i] = 1;
    }
  }
  for (let i = 0; i < n; i++) if (!active[i]) state.fill(0, 3 * i, 3 * i + 3);
  for (;;) {
    counts.activeLinks = w.restrictModes(n, rows, links, activePtr, subset, compact);
    counts.activeModes = active.reduce((a, b) => a + b, 0);
    for (let j = 0; j < 3 * n; j++) rhs[j] = active[Math.floor(j / 3)] ? D[j] : 0;
    const solve = equilibriumSolve(
      rhs,
      (v, out) => {
        transferState(v, true);
        counts.applications++;
        for (let j = 0; j < 3 * n; j++) out[j] = v[j] - output[j];
      },
      tolerance / 32,
      state,
    );
    iterations += solve.iterations;
    peakBytes = Math.max(peakBytes, solve.bytes);
    transferState(state, false);
    counts.checks++;
    error = 0;
    let added = 0;
    for (let i = 0; i < n; i++) {
      let local = 0;
      for (let j = 3 * i; j < 3 * i + 3; j++) local = Math.max(local, Math.abs(D[j] + output[j] - state[j]));
      error = Math.max(error, local);
      if (!active[i] && local > tolerance) {
        active[i] = 1;
        added++;
      }
    }
    if (!added) break;
  }
  const incident = basis.decode(state);
  counts.reconstructions++;
  let ratio = 0;
  for (let j = 0; j < 3 * n; j++)
    ratio = Math.max(ratio, Math.abs(state[j] * color[j]) / amplitude[Math.floor(j / 3)]);
  const omitted = new Float64Array(mem.buffer, discard, 1)[0] * basis.maxSynthesis;
  // The single residual audit uses the construction data after reconstruction.
  // It captures rounding, pruning and inactive modes together and never drives
  // a reflected-light iteration. The positive nodal operator gives a contraction bound.
  let residual = 0,
    auditLinks = 0;
  for (let j = 0; j < 3 * n; j++) input[j] = incident[j] * color[j];
  for (const p of parts)
    for (let r = 0; r + 1 < p.starts.length; r++) {
      const i = 3 * (p.first + r * p.stride);
      let a = direct[i] - incident[i],
        b = direct[i + 1] - incident[i + 1],
        c = direct[i + 2] - incident[i + 2];
      for (let k = p.starts[r]; k < p.starts[r + 1]; k++) {
        const j = 3 * p.col[k],
          w = p.weights[k];
        a += w * input[j];
        b += w * input[j + 1];
        c += w * input[j + 2];
      }
      residual = Math.max(residual, Math.abs(a), Math.abs(b), Math.abs(c));
      auditLinks += p.starts[r + 1] - p.starts[r];
    }
  counts.finalAuditLinks = auditLinks;
  const modal = {
    basis,
    coefficients: state,
    firstCoefficients: D,
    active,
    counts,
    buildMs,
    solveMs: performance.now() - start - buildMs,
    modalResidual: error,
    physicalResidualBound: basis.maxSynthesis * error + omitted * ratio,
    weightedOperatorBound: omitted,
    inspectLinks(receivers, sources) {
      const boundaries = new Int32Array(mem.buffer, subset, n + 1),
        entries = new Int32Array(mem.buffer, compact, 2 * counts.activeLinks);
      const from = new Set(),
        to = new Set();
      let count = 0;
      for (const i of receivers)
        for (let k = boundaries[i]; k < boundaries[i + 1]; k++) {
          const j = entries[2 * k];
          if (sources.has(j)) {
            from.add(j);
            to.add(i);
            count++;
          }
        }
      return { sourceModes: from.size, receiverModes: to.size, links: count };
    },
    transfer(sourceField, restricted = true) {
      const out = new Float64Array(3 * n);
      apply(sourceField, out, restricted);
      return out;
    },
  };
  return {
    modal,
    incident,
    iterations,
    residual,
    bytes:
      mem.buffer.byteLength + peakBytes + basis.packed.byteLength + amplitude.byteLength + 5 * D.byteLength,
  };
}
