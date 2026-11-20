import { authoredCharts, fieldGeometry } from './surface-functions.js';
import { domain, leaves, split } from './field-domain.js';
import { SourceIntegrals } from './source-integrals.js';
import { fieldEquilibrium } from './field-equilibrium.js';
import { contactLines, splitContacts } from './plane-contact.js';
import { sheetBoundaryForms } from './sheet-boundary.js';
import { poseShadows } from './plane-shadow.js';
import { dot, cross, unit } from './cpu-geometry.js';

const spatialTolerance = { absolute: 0.0015, relative: 0.045 };
const unique = (a) =>
  [...new Set(a)].sort((x, y) => x - y).filter((v, i, all) => !i || v - all[i - 1] > 1e-10);
// Close, parallel occluders can hide between any finite set of cubature points.
// Their footprint provides integration breaks even before a light field exists.
function initialFunctions({ charts, shapes }, records) {
  return charts.map((g) => {
    const root = domain(),
      s = shapes[g.object];
    g.contacts = contactLines(g, shapes);
    splitContacts(root, g.contacts);
    if (g.empty || s.axes) return root;
    const dual = records.subarray(19 * g.object, 19 * (g.object + 1)),
      features = [];
    for (const other of shapes) {
      if (other === s || other.axes) continue;
      const d = other.center.map((v, i) => v - s.center[i]),
        gap = dot(d, g.n);
      if (
        !(
          gap > 0 &&
          gap < 0.25 * Math.min(g.width, g.height, Math.hypot(...other.axisU), Math.hypot(...other.axisV))
        ) ||
        Math.abs(dot(g.n, unit(cross(other.axisU, other.axisV)))) < 0.999999
      )
        continue;
      const box = [0, 1].map((k) => {
        const axis = dual.slice(3 + 3 * k, 6 + 3 * k),
          c = 0.5 + 0.5 * dot(d, axis),
          h = 0.5 * (Math.abs(dot(other.axisU, axis)) + Math.abs(dot(other.axisV, axis)));
        return [c - h, c + h];
      });
      if (box.some(([a, b]) => b <= 0 || a >= 1)) continue;
      // Exchange is confined by both finite planes. A large opposing wall
      // has no edge inside the smaller sheet, yet the return paths change
      // within a gap length of the smaller sheet's own edge.
      features.push({ gap, box: box.map(([lo, hi]) => [Math.max(0, lo), Math.min(1, hi)]) });
    }
    const covered = [];
    for (const { gap, box } of features.sort((a, b) => a.gap - b.gap)) {
      if (covered.some((v) => v.every(([l, h], k) => box[k][0] >= l && box[k][1] <= h))) continue;
      covered.push(box);
      for (let d = 0; d < 2; d++) {
        const w = gap / (2 * (d ? g.height : g.width));
        for (const x of unique(box[d].flatMap((x) => [x - w, x, x + w])))
          if (x > 1e-8 && x < 1 - 1e-8) {
            for (const leaf of leaves(root)) split(leaf, d, x);
          }
      }
    }
    return root;
  });
}
function detail(g, state) {
  const changes = [];
  let maximum = 0,
    failed = 0,
    total = 0,
    area = 0;
  for (const cell of g.cells) {
    const chart = g.charts[cell.g],
      rho = g.shapes[chart.object].color,
      score = [0, 0],
      errors = [];
    for (let j = cell.first; j < cell.end; j++) {
      const q = g.receivers[j],
        err = [];
      for (let c = 0; c < 3; c++) {
        let v = 0;
        for (let k = 0; k < 4; k++) v += q.hat[k] * state.illumination.indirect[3 * q.ids[k] + c];
        const actual = state.light[3 * j + c],
          diff = rho[c] * (actual - v);
        err.push(diff / (spatialTolerance.absolute + spatialTolerance.relative * rho[c] * actual));
        maximum = Math.max(maximum, Math.abs(diff));
        total += q.area * diff * diff;
      }
      area += q.area;
      errors.push(err);
    }
    const n = Math.round(Math.sqrt(errors.length));
    let largest = 0;
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++)
        for (let c = 0; c < 3; c++) {
          const v = errors[j * n + i][c];
          largest = Math.max(largest, Math.abs(v));
          if (i) score[0] = Math.max(score[0], Math.abs(v - errors[j * n + i - 1][c]));
          if (j) score[1] = Math.max(score[1], Math.abs(v - errors[(j - 1) * n + i][c]));
        }
    if (largest <= 1) continue;
    failed++;
    changes.push({
      leaf: cell.leaf,
      axes: score.map((x) => x >= 0.5 * Math.max(...score)),
      score: largest,
    });
  }
  return { changes, maximum, rms: Math.sqrt(total / (3 * area)), failed };
}
export class DiffuseTransport {
  constructor(angular) {
    this.sources = new SourceIntegrals(angular);
  }
  dispose() {
    this.sources.dispose();
  }
  async update(scene) {
    const start = performance.now(),
      surface = authoredCharts(scene),
      shadowOn = poseShadows(surface, scene.lightPosition),
      sheetForms = sheetBoundaryForms(surface, scene.records),
      knots = initialFunctions(surface, scene.records),
      history = [];
    const counts = Array(20).fill(0);
    let solveMs = 0,
      assemblyMs = 0,
      workerMs = 0,
      work = 0,
      peak = 0;
    for (let pass = 0; ; pass++) {
      const g = fieldGeometry(scene, surface, knots, 3),
        I = await this.sources.integrate(g, shadowOn, sheetForms);
      const t = performance.now(),
        state = fieldEquilibrium(g, I),
        estimate = detail(g, state);
      solveMs += performance.now() - t;
      assemblyMs += I.timings.assemblyMs;
      workerMs += I.timings.workerMs;
      work += I.newJobs;
      I.counts.forEach((count, i) => (counts[i] += count));
      peak = Math.max(peak, I.bytes + state.bytes);
      history.push({
        coefficients: g.supports.length,
        observations: g.receivers.length,
        cells: g.cells.length,
        receiverModeLinks: state.modeLinks,
        pointLinks: I.links,
        outlineSections: I.outlineSections,
        sheetSections: I.sheetSections,
        axisWork: I.axisWork,
        failed: estimate.failed,
        maximum: estimate.maximum,
        rms: estimate.rms,
      });
      if (!estimate.failed || pass === 3) {
        return {
          illumination: state.illumination,
          firstBounce: state.first,
          direct: state.direct,
          iterations: state.iterations,
          residual: state.residual,
          maxMass: I.maxMass,
          spatialTolerance,
          history,
          converged: !estimate.failed,
          newJobs: work,
          bytes: peak,
          workerBytes: I.workerBytes,
          links: I.links,
          axisCounts: I.axisCounts,
          counts,
          timings: {
            solve: solveMs,
            assemblyMs,
            workerMs,
            basis: performance.now() - start - solveMs - assemblyMs - workerMs,
            operatorMs: performance.now() - start - solveMs,
          },
          operator: { reflect: state.reflect, closure: state.closure },
        };
      }
      for (const { leaf, axes } of estimate.changes) {
        if (axes[0]) split(leaf, 0, (leaf.bounds[0] + leaf.bounds[1]) / 2);
        if (axes[1])
          for (const child of leaf.children || [leaf])
            split(child, 1, (child.bounds[2] + child.bounds[3]) / 2);
      }
    }
  }
}
