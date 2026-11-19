import { nearField } from './parallel-transfer.js';
import { cavityWeight } from './equilibrium.js';
import { painter } from './painter.js';
import { volume, regions, shell, dot, cross, sub, unit, point } from './cpu-geometry.js';
import { split } from './visibility.js';
import { planarCharts, chartPoint, chartCoordinate, chartPolygons } from './irradiance-layout.js';

function intersect(poly, mask) {
  for (let i = 0; i < mask.length && poly.length; i++) {
    const a = mask[i],
      b = mask[(i + 1) % mask.length],
      x = a[1] - b[1],
      y = b[0] - a[0];
    [poly] = split(poly, x, y, -x * a[0] - y * a[1], 1e-11);
  }
  return poly;
}
function signed(p, plane) {
  return dot(sub(p, plane.origin), plane.normal);
}
function splitFace(poly, plane) {
  const pos = [],
    neg = [];
  let a = poly.at(-1),
    da = signed(a, plane);
  for (const b of poly) {
    const db = signed(b, plane);
    if ((da < 0 && db > 0) || (db < 0 && da > 0)) {
      const t = da / (da - db),
        p = a.map((x, k) => x + t * (b[k] - x));
      pos.push(p);
      neg.push(p);
    }
    if (db >= 0) pos.push(b);
    if (db <= 0) neg.push(b);
    a = b;
    da = db;
  }
  return [pos, neg];
}
function trim(polys, blockers, shadows) {
  const dark = [];
  for (const planes of blockers) {
    const next = [];
    for (const poly of polys) {
      let inside = poly,
        out = [];
      for (const plane of planes) {
        if (!inside.length) break;
        let lo = Infinity,
          hi = -Infinity;
        for (const p of inside) {
          const v = signed(p, plane);
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
        if (hi <= 1e-9) {
          inside = [];
          break;
        }
        if (lo >= -1e-9) continue;
        const [a, b] = splitFace(inside, plane);
        inside = a;
        if (b.length >= 3) out.push(b);
      }
      if (inside.length >= 3) {
        next.push(...out);
        if (shadows) dark.push(inside);
      } else next.push(poly);
    }
    polys = next;
    if (!polys.length) break;
  }
  return [polys, dark];
}
function relevant(box, volumes) {
  const answer = [];
  for (const [owner, planes] of volumes) {
    if (owner === box || !planes) continue;
    let keep = true;
    const candidates = [];
    for (const p of planes) {
      const distance = signed(box.center, p);
      const extent = box.axes.reduce((s, a, i) => s + Math.abs(dot(a, p.normal)) * box.halfSize[i], 0);
      if (distance + extent <= 1e-9) {
        keep = false;
        break;
      }
      if (distance - extent < 1e-9) candidates.push(p);
    }
    if (keep) answer.push(candidates);
  }
  return answer;
}
function direct(p, n, lamp) {
  const d = sub(lamp, p),
    r = dot(d, d);
  return Math.max(0, dot(n, d)) / (r * Math.sqrt(r));
}
function field(p, sources, out) {
  let sum = 0;
  out.fill(0);
  for (const { r, rgb, slope } of sources) {
    const d = sub(r.center, p),
      distance = dot(d, d) + 0.1 * r.area;
    const w = r.area / (distance * distance);
    sum += w;
    const local = Math.max(-1, Math.min(1, -dot(d, r.momentAxis)));
    for (let c = 0; c < 3; c++) out[c] += w * (rgb[c] + local * slope[c]);
  }
  for (let c = 0; c < 3; c++) out[c] /= sum;
  return out;
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
    this.shells = new WeakMap();
    this.vertices = new Float32Array(0);
  }
  setScene(scene, solution) {
    this.scene = scene;
    this.solution = solution;
    this.shapes = [...scene.rectangles, ...scene.roundedBoxes];
    const sheets = planarCharts(scene.rectangles, this.cache).slots;
    const charts = new Map();
    this.faces = sheets;
    const modeSources = (r, s) => {
      const { offset, nu, nv } = solution.groups.get(r);
      return solution.modes.slice(offset, offset + nu * nv).map((m, i) => ({
        side: s,
        r: m,
        covers: (solution.covers.get(r) || []).filter((h) => h.side === s),
        slope: solution.sidedSlopes.subarray(6 * (offset + i) + 3 * s, 6 * (offset + i) + 3 * s + 3),
        rgb: solution.incident.subarray(3 * (2 * (offset + i) + s), 3 * (2 * (offset + i) + s) + 3),
      }));
    };
    for (let i = 0; i < sheets.length; i++)
      for (let side = 0; side < 2; side++) {
        const g = sheets[i][side];
        if (!charts.has(g)) charts.set(g, []);
        charts.get(g).push(...modeSources(scene.rectangles[i], side));
      }
    const width = 48,
      rows = 24,
      height = charts.size * rows,
      rgb = [0, 0, 0];
    if (this.prevCharts !== sheets || !this.lamp || scene.lightPosition.some((x, i) => x !== this.lamp[i])) {
      this.prevCharts = sheets;
      this.lamp = scene.lightPosition.slice();
      this.filters = [];
      const tex = new Float32Array(width * height * 4);
      this.atlas = { tex, width, height, rows };
      let row = 0;
      for (const [g, sources] of charts) {
        g.displayRow = row;
        this.filters.push(chartFilter(g, sources, width, rows, tex, row, scene.lightPosition));
        row += rows;
      }
    }
    let chart = 0,
      base = 0;
    const { tex } = this.atlas;
    for (const [grid, sources] of charts) {
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
        for (let j = 0; j < sources.length; j++) sources[j].rgb = local.other.subarray(3 * j, 3 * j + 3);
      const { weights, linearWeights, indices, terms } = this.filters[chart++];
      let k = 0;
      for (let i = 0; i < width * rows; i++, base += 4) {
        let r = 0,
          g = 0,
          b = 0;
        for (let j = 0; j < terms; j++) {
          const source = sources[indices ? indices[k] : j];
          const m = local ? 0 : linearWeights[k],
            w = weights[k++];
          r += source.rgb[0] * w + source.slope[0] * m;
          g += source.rgb[1] * w + source.slope[1] * m;
          b += source.rgb[2] * w + source.slope[2] * m;
        }
        if (local) {
          const open = local.open[i];
          r *= open;
          g *= open;
          b *= open;
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
    }
    this.solidFaces = scene.roundedBoxes.map((b, i) => {
      let faces = this.shells.get(b);
      if (!faces) {
        faces = shell(b);
        this.shells.set(b, faces);
      }
      const refs = solution.proxy[i].map((r) => modeSources(r, 0));
      const lamp = relevant(solution.boxes[i], solution.lampVolumes);
      const shaded = new Map();
      return faces.map((vertices) => {
        const p = vertices[0].p,
          n = unit(cross(sub(vertices[1].p, p), sub(vertices[2].p, p)));
        // Defer clipping and lighting until this part of the convex shell is visible.
        const face = {
          p,
          n,
          regions: null,
          prepare() {
            const poly = vertices.map((vertex) => {
              if (shaded.has(vertex)) return shaded.get(vertex);
              const { p, normal } = vertex;
              const value = [0, 0, 0];
              for (let k = 0; k < 3; k++) {
                const c = dot(b.axes[k], normal);
                if (Math.abs(c) < 1e-7) continue;
                const f = field(p, refs[2 * k + Number(c < 0)], rgb);
                for (let j = 0; j < 3; j++) value[j] += c * c * f[j];
              }
              const result = [...p, ...value, direct(p, normal, scene.lightPosition)];
              shaded.set(vertex, result);
              return result;
            });
            const [lit, dark] = poly.every((v) => v[6] === 0) ? [[], [poly]] : trim([poly], lamp, true);
            return (face.regions = [lit, dark]);
          },
        };
        return face;
      });
    });
    this.shadows = scene.rectangles.map((r) => solution.lampRegions.get(r));
  }
  drawList(eye) {
    const order = painter(this.shapes, eye);
    const volumes = order ? null : this.shapes.map((s) => [s, volume(s, eye)]);
    let data = this.vertices,
      length = 0;
    function fan(poly, vertex) {
      if (poly.length < 3) return;
      const end = length + 39 * (poly.length - 2);
      if (end > data.length) {
        const next = new Float32Array(Math.max(end, 1024, data.length * 2));
        next.set(data.subarray(0, length));
        data = next;
      }
      const a = vertex(poly[0]);
      let b = vertex(poly[1]);
      for (let i = 2; i < poly.length; i++) {
        const c = vertex(poly[i]);
        data.set(a, length);
        data.set(b, length + 13);
        data.set(c, length + 26);
        length += 39;
        b = c;
      }
    }
    const { scene, atlas } = this;
    const sheet = (r, i) => {
      const side = Number(dot(sub(eye, r.center), cross(r.axisU, r.axisV)) < 0),
        g = this.faces[i][side];
      const visible = volumes ? regions(r, volumes).filter((c) => c.lit) : [null];
      for (const v of visible)
        for (const s of this.shadows[i]) {
          const uv = v ? intersect(v.uv, s.uv) : s.uv;
          const poly = uv.map(([u, v]) => point(r, u, v));
          for (const cell of g.coords ? chartPolygons(poly, [g]) : [poly])
            fan(cell, (p) => {
              return [
                ...p,
                (0.5 + (atlas.width - 1) * chartCoordinate(g, 0, dot(g.du, p) + g.du[3])) / atlas.width,
                (g.displayRow + 0.5 + (atlas.rows - 1) * chartCoordinate(g, 1, dot(g.dv, p) + g.dv[3])) /
                  atlas.height,
                ...r.color,
                0,
                0,
                0,
                0,
                +s.lit,
              ];
            });
        }
    };
    const solid = (b, i) => {
      const blocks = volumes ? relevant(b, volumes) : null;
      for (const f of this.solidFaces[i]) {
        if (dot(sub(eye, f.p), f.n) <= 0) continue;
        const parts = f.regions || f.prepare();
        for (let j = 0; j < 2; j++) {
          const polys = parts[j],
            lit = 1 - j;
          for (const poly of blocks ? trim(polys, blocks, false)[0] : polys)
            fan(poly, (p) => [p[0], p[1], p[2], -1, 0, ...b.color, ...p.slice(3), lit]);
        }
      }
    };
    for (const i of order || this.shapes.keys()) {
      if (i < scene.rectangles.length) sheet(this.shapes[i], i);
      else solid(this.shapes[i], i - scene.rectangles.length);
    }
    this.vertices = data;
    return data.subarray(0, length);
  }
}
