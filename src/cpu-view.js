import { compileContact, drawContact, contactMoving } from './contact-transfer.js';
import { nearField } from './parallel-transfer.js';
import { cavityWeight } from './equilibrium.js';
import { dot, cross, sub, unit } from './cpu-geometry.js';
import { planarCharts, chartPoint } from './irradiance-layout.js';

function sameBlockers(contact, moving) {
  return moving.length === contact.moving.length && moving.every((b, i) => b === contact.moving[i]);
}
function direct(p, n, lamp) {
  const d = sub(lamp, p),
    r = dot(d, d);
  return Math.max(0, dot(n, d)) / (r * Math.sqrt(r));
}
function bracket(knots, t) {
  if (t <= knots[0]) return [0, 0, 0];
  for (let i = 1; i < knots.length; i++)
    if (t <= knots[i]) return [i - 1, i, (t - knots[i - 1]) / (knots[i] - knots[i - 1])];
  return [knots.length - 1, knots.length - 1, 0];
}
function chartFilter(g, sources, width, rows, tex, row, lamp) {
  const rect = sources[0].r.rect,
    oneSheet = sources.every((s) => s.r.rect === rect);
  const covers = [...new Set(sources.flatMap((s) => s.covers))],
    memberships = sources.map((s) => covers.map((h) => cavityWeight(s.r.center, h)));
  const { nu, nv } = sources[0].r;
  const centers = (start, stride, count, axis) =>
    Array.from({ length: count }, (_, j) => {
      const b = sources[start + j * stride].r.bounds;
      return (b[axis] + b[axis + 1]) / 2;
    });
  const us = centers(0, 1, nu, 0),
    vs = centers(0, nu, nv, 2);
  const terms = oneSheet ? 4 : sources.length,
    weights = new Float32Array(rows * width * terms),
    linearWeights = new Float32Array(weights.length),
    indices = oneSheet ? new Uint32Array(weights.length) : null;
  const n = unit(cross(rect.axisU, rect.axisV)),
    area = dot(cross(rect.axisU, rect.axisV), n),
    du = cross(rect.axisV, n),
    dv = cross(n, rect.axisU);
  for (let y = 0; y < rows; y++)
    for (let x = 0; x < width; x++) {
      const p = chartPoint(g, x / (width - 1), y / (rows - 1)),
        i = y * width + x;
      tex[4 * (row * width + i) + 3] = direct(p, g.n, lamp);
      const base = i * terms;
      if (oneSheet) {
        const offset = sub(p, rect.center),
          [u0, u1, ux] = bracket(us, dot(offset, du) / area),
          [v0, v1, vy] = bracket(vs, dot(offset, dv) / area);
        indices.set([v0 * nu + u0, v0 * nu + u1, v1 * nu + u0, v1 * nu + u1], base);
        weights.set([(1 - ux) * (1 - vy), ux * (1 - vy), (1 - ux) * vy, ux * vy], base);
      } else {
        const coverage = covers.map((h) => cavityWeight(p, h));
        let sum = 0;
        for (let j = 0; j < sources.length; j++) {
          const r = sources[j].r,
            d = sub(r.center, p),
            h = dot(d, d) + 0.1 * r.area;
          let w = r.area / (h * h);
          for (let k = 0; k < coverage.length; k++) {
            const a = 1 - Math.abs(coverage[k] - memberships[j][k]);
            w *= a * a;
          }
          sum += weights[base + j] = w;
        }
        for (let j = 0; j < sources.length; j++) weights[base + j] /= sum;
      }
      for (let j = 0; j < terms; j++) {
        const r = sources[indices ? indices[base + j] : j].r;
        linearWeights[base + j] =
          weights[base + j] * Math.max(-1, Math.min(1, dot(sub(p, r.center), r.momentAxis)));
      }
    }
  return {
    weights,
    linearWeights,
    indices,
    terms,
    byteLength: weights.byteLength + linearWeights.byteLength + (indices?.byteLength || 0),
  };
}
export class CpuView {
  constructor() {
    this.cache = {};
  }
  setScene(scene, solution, visibleOnly = false) {
    this.scene = scene;
    this.solution = solution;
    const sheets = planarCharts(scene.rectangles, this.cache, solution.contacts).slots;
    const charts = new Map();
    this.faces = sheets;
    const modeSources = (this.modeSources = (r, s) => {
      const { offset, nu, nv } = solution.groups.get(r);
      const covers = (solution.covers.get(r) || []).filter((h) => h.side === s);
      return solution.modes.slice(offset, offset + nu * nv).map((m, i) => ({
        side: s,
        r: m,
        covers,
        slope: solution.sidedSlopes.subarray(6 * (offset + i) + 3 * s, 6 * (offset + i) + 3 * s + 3),
        rgb: solution.incident.subarray(3 * (2 * (offset + i) + s), 3 * (2 * (offset + i) + s) + 3),
      }));
    });
    for (let i = 0; i < sheets.length; i++)
      for (let side = 0; side < 2; side++) {
        const g = sheets[i][side];
        if (!charts.has(g)) charts.set(g, []);
        charts.get(g).push({ rect: scene.rectangles[i], side });
      }
    const width = 48,
      rows = 24,
      height = Math.max(1, charts.size) * rows;
    if (this.prevCharts !== sheets || !this.lamp || scene.lightPosition.some((x, i) => x !== this.lamp[i])) {
      this.prevCharts = sheets;
      this.lamp = scene.lightPosition.slice();
      this.filters = new Map();
      this.contacts = new Map();
      this.atlas = { tex: new Float32Array(width * height * 4), width, height, rows };
      let row = 0;
      for (const g of charts.keys()) {
        g.displayRow = row;
        row += rows;
      }
    }
    this.contactDraws = this.contactBackend?.beginChart ? new Map() : null;
    this.pendingCharts = charts;
    this.chartMs = 0;
    this.atlasChanged = false;
    if (!visibleOnly) for (const grid of charts.keys()) this.prepareChart(grid);
  }
  // Reuse the last drawn receivers as a visibility hint while the transport
  // workers run. A newly visible chart still compiles against this exact pose.
  prepareContactGeometry(geometry) {
    if (!this.contacts?.size || (this.contactBackend && !this.contactBackend.beginChart)) return;
    const prepared = new Map();
    for (const [grid, contact] of this.contacts) {
      if (this.pendingCharts.has(grid) || !geometry.contacts.includes(contact.feature)) continue;
      const moving = contactMoving(contact.domain, geometry.boxes);
      if (sameBlockers(contact, moving)) continue;
      const { width, rows } = this.atlas;
      prepared.set(
        grid,
        compileContact(
          grid,
          contact.feature,
          geometry,
          width,
          rows,
          this.filters.get(grid),
          contact.stationary,
          this.contactBackend,
        ),
      );
    }
    return prepared;
  }
  // The equilibrium includes the entire scene. Only reconstruct a surface-side
  // texture when a drawable polygon first refers to it in this revision.
  prepareChart(grid) {
    const sheets = this.pendingCharts.get(grid);
    if (!sheets) return;
    const start = performance.now();
    const sources = sheets.flatMap(({ rect, side }) => this.modeSources(rect, side));
    const { tex, width, rows } = this.atlas;
    const solution = this.solution;
    let filter = this.filters.get(grid);
    if (!filter) {
      filter = chartFilter(grid, sources, width, rows, tex, grid.displayRow, this.scene.lightPosition);
      this.filters.set(grid, filter);
    }
    const {
      side,
      r: { rect },
    } = sources[0];
    const blocks = solution.near.filter((b) => b.receiver === rect && b.side === side);
    const local =
      blocks.length === 1 && sources.every((s) => s.r.rect === rect)
        ? nearField(blocks[0], grid, width, rows, solution)
        : null;
    if (local)
      for (let j = 0; j < sources.length; j++) {
        sources[j].rgb = local.other.subarray(3 * j, 3 * j + 3);
        sources[j].slope = local.otherSlope.subarray(3 * j, 3 * j + 3);
      }
    const { weights, linearWeights, indices, terms } = filter;
    const chartBase = grid.displayRow * width * 4;
    let base = chartBase;
    let k = 0;
    for (let i = 0; i < width * rows; i++, base += 4) {
      let r = 0,
        g = 0,
        b = 0;
      for (let j = 0; j < terms; j++) {
        const source = sources[indices ? indices[k] : j];
        const m = linearWeights[k],
          w = weights[k++];
        r += source.rgb[0] * w + source.slope[0] * m;
        g += source.rgb[1] * w + source.slope[1] * m;
        b += source.rgb[2] * w + source.slope[2] * m;
      }
      if (local) {
        const open = local.open[i];
        // In the cavity the opening varies on the gap scale. At the exposed
        // side of the edge use the continuous background across the sheet.
        const mix = local.blend[i];
        r = open * (r + mix * (local.background[3 * i] - r));
        g = open * (g + mix * (local.background[3 * i + 1] - g));
        b = open * (b + mix * (local.background[3 * i + 2] - b));
        for (let j = 0; j < local.count; j++) {
          const w = local.weights[i * local.count + j],
            m = local.gradients[i * local.count + j];
          r += w * local.emitted[3 * j] + m * local.slopes[3 * j];
          g += w * local.emitted[3 * j + 1] + m * local.slopes[3 * j + 1];
          b += w * local.emitted[3 * j + 2] + m * local.slopes[3 * j + 2];
        }
      }
      tex[base] = r;
      tex[base + 1] = g;
      tex[base + 2] = b;
    }
    const feature =
      !local && indices && solution.contacts.find((f) => f.receiver === rect && f.side === side);
    if (feature) {
      let contact = solution.preparedContacts?.get(grid) || this.contacts.get(grid);
      const moving = contact && contactMoving(contact.domain, solution.boxes);
      if (!contact || contact.feature !== feature || !sameBlockers(contact, moving)) {
        contact = compileContact(
          grid,
          feature,
          solution,
          width,
          rows,
          filter,
          contact?.feature === feature ? contact.stationary : undefined,
          this.contactBackend,
        );
      }
      this.contacts.set(grid, contact);
      if (contact.resident) this.contactDraws.set(grid, { contact, filter });
      else drawContact(contact, filter, solution, tex, chartBase);
    }
    this.chartMs += performance.now() - start;
    this.pendingCharts.delete(grid);
    this.atlasChanged = true;
  }
}
