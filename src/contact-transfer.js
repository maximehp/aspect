// The last diffuse transfer near a sheet that touches a receiver. Integrate
// the cosine-weighted solid angle of visible polygons; their radiance comes
// from the same equilibrium as the surrounding surface.
import { canShadow, cross, dot, point, regions, shadowBounds, sub, unit, volume } from './cpu-geometry.js';
import { projectedFootprint, partitionReceiver } from './visibility.js';
import { chartPoint } from './irradiance-layout.js';

const corners = [
  [-1, -1],
  [1, -1],
  [1, 1],
  [-1, 1],
];
const vertices = (r) => corners.map(([u, v]) => point(r, u, v));
function bounds(points) {
  return [0, 1, 2].map((k) => [Math.min(...points.map((p) => p[k])), Math.max(...points.map((p) => p[k]))]);
}
function overlaps(a, b) {
  return a.every(([lo, hi], k) => lo <= b[k][1] + 1e-7 && hi >= b[k][0] - 1e-7);
}
function contactFaces(n, receiver, sheets) {
  const u = unit(receiver.axisU),
    v = unit(receiver.axisV);
  if (Math.abs(dot(u, v)) > 1e-8) return [];
  const half = [Math.hypot(...receiver.axisU), Math.hypot(...receiver.axisV)];
  const contacts = [];
  for (const r of sheets) {
    if (r === receiver) continue;
    const ps = vertices(r),
      heights = ps.map((p) => dot(sub(p, receiver.center), n));
    const height = Math.max(...heights);
    if (height <= 1e-6 || Math.abs(Math.min(...heights)) > 1e-7) continue;
    const foot = ps.filter((_, i) => Math.abs(heights[i]) < 1e-7);
    if (foot.length !== 2 || Math.abs(dot(unit(cross(r.axisU, r.axisV)), n)) > 1e-8) continue;
    const mid = foot[0].map((x, k) => (x + foot[1][k]) / 2);
    if (![u, v].every((a, k) => Math.abs(dot(sub(mid, receiver.center), a)) < half[k] - 1e-7)) continue;
    contacts.push({ r, foot, height, top: bounds(ps.filter((_, i) => heights[i] > height - 1e-7)) });
  }
  return contacts;
}
function distanceToEdge(p, [a, b]) {
  const d = sub(b, a),
    delta = sub(p, a),
    t = Math.max(0, Math.min(1, dot(d, delta) / dot(d, d)));
  return Math.hypot(...delta.map((v, k) => v - t * d[k]));
}
function projectedArea(poly, p, n) {
  // Both the angle atan2(|a×b|, a·b) and the oriented unit edge
  // normal are invariant under positive scaling of either direction.
  // The closing edge uses the last nondegenerate direction, as do the interior edges.
  let last = poly.length - 1;
  const length2 = (q) => {
    const x = q[0] - p[0],
      y = q[1] - p[1],
      z = q[2] - p[2];
    return x * x + y * y + z * z;
  };
  while (last >= 0 && length2(poly[last]) <= 1e-24) last--;
  if (last < 2) return 0;
  let a = poly[last],
    ax = a[0] - p[0],
    ay = a[1] - p[1],
    az = a[2] - p[2],
    al = length2(a);
  let sum = 0,
    count = 0;
  for (let i = 0; i <= last; i++) {
    const b = poly[i],
      bx = b[0] - p[0],
      by = b[1] - p[1],
      bz = b[2] - p[2];
    const bl = bx * bx + by * by + bz * bz;
    if (bl <= 1e-24) continue;
    const x = ay * bz - az * by,
      y = az * bx - ax * bz,
      z = ax * by - ay * bx;
    const squared = x * x + y * y + z * z;
    if (squared > 1e-24 * al * bl) {
      const sine = Math.sqrt(squared);
      sum += ((x * n[0] + y * n[1] + z * n[2]) * Math.atan2(sine, ax * bx + ay * by + az * bz)) / sine;
    }
    ax = bx;
    ay = by;
    az = bz;
    al = bl;
    count++;
  }
  return count < 3 ? 0 : Math.abs(sum) / (2 * Math.PI);
}
export const azimuthCount = 8,
  sectorCount = 2 * azimuthCount,
  areaSteps = 6;
const halfSector = Math.PI / azimuthCount;
// Within a sector at azimuth a, the divider has tan(elevation) = cos(a).
// Integrate its cosine-weighted solid angle to normalize a constant field.
const lowerFraction = 1 - Math.atan(Math.tan(halfSector) / Math.SQRT2) / (halfSector * Math.SQRT2);

export function contactReceivers(sheets, groups) {
  const result = [];
  for (const receiver of sheets)
    for (const side of [0, 1]) {
      const n = unit(cross(receiver.axisU, receiver.axisV)).map((x) => (side ? -x : x));
      const contacts = contactFaces(n, receiver, sheets);
      if (!contacts.length) continue;
      const domain = bounds(
        contacts.flatMap((c) =>
          c.foot.flatMap((p) => [p.map((x) => x - 3 * c.height), p.map((x) => x + 3 * c.height)]),
        ),
      );
      const supports = new Set(contacts.map((c) => c.r));
      for (const r of sheets) {
        if (r === receiver || Math.abs(dot(unit(cross(r.axisU, r.axisV)), n)) < 1 - 1e-8) continue;
        const box = bounds(vertices(r));
        if (
          box.every(([lo, hi], k) => lo >= domain[k][0] && hi <= domain[k][1]) &&
          contacts.some((c) => overlaps(c.top, box))
        )
          supports.add(r);
      }
      const group = groups.get(receiver);
      const u = unit(receiver.axisU),
        v = cross(n, u);
      result.push({
        receiver,
        contacts,
        supports,
        sheets,
        domain,
        n,
        u,
        v,
        offset: group.offset,
        size: group.nu * group.nv,
        side,
      });
    }
  return result;
}
function clip(poly, p, axis) {
  const distance = (v) => (v[0] - p[0]) * axis[0] + (v[1] - p[1]) * axis[1] + (v[2] - p[2]) * axis[2];
  let positive = false,
    negative = false;
  for (const v of poly) {
    if (distance(v) >= 0) positive = true;
    else negative = true;
    if (positive && negative) break;
  }
  if (!positive) return [];
  if (!negative) return poly;
  const out = [];
  let a = poly.at(-1),
    da = distance(a);
  for (const b of poly) {
    const db = distance(b);
    if (da * db < 0) out.push(a.map((x, k) => x + ((b[k] - x) * da) / (da - db)));
    if (db >= 0) out.push(b);
    a = b;
    da = db;
  }
  return out;
}
export function compileContact(grid, feature, solution, width, rows, filter, stationary = [], backend) {
  const { receiver, contacts, supports, sheets, u, v } = feature;
  const planes = Array.from({ length: azimuthCount }, (_, i) => {
    const angle = (2 * Math.PI * (i - 0.5)) / azimuthCount;
    return u.map((x, k) => -x * Math.sin(angle) + v[k] * Math.cos(angle));
  });
  const elevationPlanes = Array.from({ length: azimuthCount }, (_, i) =>
    grid.n.map(
      (z, k) =>
        z -
        u[k] * Math.cos((2 * Math.PI * i) / azimuthCount) -
        v[k] * Math.sin((2 * Math.PI * i) / azimuthCount),
    ),
  );
  const farPlanes = planes.map((_, k) => planes[(k + 1) % azimuthCount].map((x) => -x));
  const lowerPlanes = elevationPlanes.map((axis) => axis.map((x) => -x));
  const contactModes = [];
  for (const r of supports) {
    const { offset, nu, nv } = solution.groups.get(r);
    for (let j = offset; j < offset + nu * nv; j++) contactModes.push({ j, m: solution.modes[j] });
  }
  const domain = feature.domain;
  const blockers = sheets.filter((r) => r !== receiver && overlaps(domain, bounds(vertices(r))));
  const moving = contactMoving(domain, solution.boxes);

  function influence(p) {
    const proximity = Math.min(...contacts.map((c) => distanceToEdge(p, c.foot) / c.height));
    const t = Math.max(0, Math.min(1, (3 - proximity) / 2));
    return t * t * (3 - 2 * t);
  }
  // The visible stationary portions and their angular integrals depend only on
  // the receiver site and the fixed sheets. A solid can only subtract from
  // those portions. Apply its shadow after the fixed blockers, in scene order.
  function integrate(m, side, p, silhouette, emit) {
    for (const polygon of silhouette) {
      if (!polygon.lit) continue;
      const ps = polygon.uv.map(([u, v]) => point(m, u, v));
      for (let k = 0; k < azimuthCount; k++) {
        let poly = clip(ps, p, planes[k]);
        if (poly.length < 3) continue;
        poly = clip(poly, p, farPlanes[k]);
        if (poly.length < 3) continue;
        for (let band = 0; band < 2; band++) {
          const part = clip(poly, p, band ? elevationPlanes[k] : lowerPlanes[k]);
          if (part.length < 3) continue;
          const w = projectedArea(part, p, grid.n);
          if (w <= 1e-10) continue;
          const centroid = part.reduce((s, q) => s + dot(sub(q, m.center), m.momentAxis), 0) / part.length;
          emit(side, w, w * centroid, k + band * azimuthCount);
        }
      }
    }
  }
  const batch = backend?.beginChart?.(feature, width, rows);
  const readback = backend && !backend.beginChart;
  function stencil(site, emit) {
    const p = site.p;
    if (!site.samples) {
      const samples = [],
        integrals = [];
      const shadows = blockers.map((s) => [s, volume(s, p)]);
      for (const { m, j } of contactModes) {
        const plane = dot(sub(p, m.center), m.normal);
        if (Math.abs(plane) < 1e-8) continue;
        const side = 2 * j + Number(plane < 0);
        const lit = regions(m, shadows).filter((poly) => poly.lit);
        if (!lit.length) continue;
        const start = integrals.length;
        integrate(m, side, p, lit, (_, w, moment, bin) => integrals.push(w, moment, bin));
        samples.push({ m, side, lit, start, end: integrals.length });
      }
      site.samples = samples;
      site.integrals = Float64Array.from(integrals);
    }
    const { samples, integrals } = site;
    if (!samples.length) return;
    const shadows = moving.map((shape) => ({ shape, bounds: shadowBounds(shape, p) }));
    for (const { m, side, lit, start, end } of samples) {
      const cuts = [];
      for (const shadow of shadows) {
        if (!canShadow(m, shadow.bounds)) continue;
        shadow.planes ??= volume(shadow.shape, p, shadow.bounds.basis);
        const polygon = projectedFootprint(m, shadow.planes);
        if (polygon) cuts.push(polygon);
      }
      if (cuts.length) {
        const visible = partitionReceiver(cuts, lit);
        if (visible.length !== lit.length || visible.some((c, i) => !c.lit || c.uv !== lit[i].uv)) {
          if (!batch && !readback) integrate(m, side, p, visible, emit);
          else
            for (const poly of visible) {
              if (
                poly.lit &&
                !(batch
                  ? batch.enqueue(m, side, site, poly)
                  : backend.enqueue(m, side, p, poly, feature, emit))
              )
                integrate(m, side, p, [poly], emit);
            }
          continue;
        }
      }
      for (let i = start; i < end; i += 3) emit(side, integrals[i], integrals[i + 1], integrals[i + 2]);
    }
  }
  // The amount of background visible in each angular bin is averaged over
  // each transport element. Compare this with the opening at the display site.
  const { offset, size } = feature,
    cellOpen = new Float64Array(size * sectorCount),
    slopeOpen = new Float64Array(cellOpen.length);
  const empty = Array.from(
    { length: sectorCount },
    (_, d) => (d < azimuthCount ? lowerFraction : 1 - lowerFraction) / azimuthCount,
  );
  // Site membership and quadrature coordinates are intrinsic to the receiver.
  // Only visibility and its integrals depend on the current solid transforms.
  if (!stationary.length) {
    for (let j = 0; j < size; j++) {
      const mode = solution.modes[offset + j];
      for (let y = 0; y < areaSteps; y++)
        for (let x = 0; x < areaSteps; x++) {
          const p = point(mode, (2 * x + 1) / areaSteps - 1, (2 * y + 1) / areaSteps - 1);
          if (influence(p)) stationary.push({ p, cell: j, pos: dot(sub(p, mode.center), mode.momentAxis) });
        }
    }
    for (let j = 0; j < rows; j++)
      for (let i = 0; i < width; i++) {
        const p = chartPoint(grid, i / (width - 1), j / (rows - 1)),
          fade = influence(p);
        if (fade) stationary.push({ p, texel: j * width + i, fade });
      }
  }
  for (let j = 0; j < size; j++) cellOpen.set(empty, j * sectorCount);
  let index = 0;
  for (; index < stationary.length && stationary[index].cell !== undefined; index++) {
    const site = stationary[index],
      base = site.cell * sectorCount;
    stencil(site, (_, w, moment, d) => {
      cellOpen[base + d] -= w / (areaSteps * areaSteps);
      slopeOpen[base + d] -= (3 * site.pos * w) / (areaSteps * areaSteps);
    });
  }
  function limitOpen() {
    for (let i = 0; i < cellOpen.length; i++) {
      cellOpen[i] = Math.max(0, cellOpen[i]);
      slopeOpen[i] = Math.max(-cellOpen[i], Math.min(cellOpen[i], slopeOpen[i]));
    }
  }
  if (!batch && !readback) limitOpen();
  const { terms, weights, linearWeights, indices } = filter;
  const sites = [],
    sides = [],
    localMean = [],
    localSlope = [],
    ends = [],
    gains = [],
    strength = [],
    openings = [];
  function finishSite(site, fade, open, local) {
    for (const [side, [w, m]] of local) {
      sides.push(side);
      localMean.push(w);
      localSlope.push(m);
    }
    sites.push(site);
    strength.push(fade);
    ends.push(sides.length);
    if (batch) {
      openings.push(...open);
      return;
    }
    for (let d = 0; d < sectorCount; d++) {
      let mass = 0;
      for (let t = site * terms; t < (site + 1) * terms; t++) {
        const k = indices[t] * sectorCount + d;
        mass += weights[t] * cellOpen[k] + linearWeights[t] * slopeOpen[k];
      }
      // A bin smaller than one cell quadrature contribution is unresolved.
      gains.push(Math.max(0, open[d]) / Math.max(1 / (sectorCount * areaSteps * areaSteps), mass));
    }
  }
  const deferred = [];
  for (; index < stationary.length; index++) {
    const record = stationary[index],
      site = record.texel,
      fade = record.fade;
    const open = empty.slice(),
      local = new Map();
    stencil(record, (side, w, m, d) => {
      open[d] -= w;
      const pair = local.get(side) || [0, 0];
      pair[0] += w;
      pair[1] += m;
      local.set(side, pair);
    });
    if (readback) deferred.push([site, fade, open, local]);
    else finishSite(site, fade, open, local);
  }
  if (readback) {
    backend.flush();
    limitOpen();
    for (const args of deferred) finishSite(...args);
  }
  return {
    feature,
    resident: batch?.finish(cellOpen, slopeOpen, sites, openings),
    stationary,
    domain,
    moving,
    sites: Uint32Array.from(sites),
    strength: Float32Array.from(strength),
    ends: Uint32Array.from(ends),
    sides: Uint32Array.from(sides),
    localMean: Float32Array.from(localMean),
    localSlope: Float32Array.from(localSlope),
    gains: Float32Array.from(gains),
  };
}
export function contactMoving(domain, boxes) {
  return boxes.filter((b) =>
    overlaps(
      domain,
      b.bounds.min.map((x, k) => [x, b.bounds.max[k]]),
    ),
  );
}

export function drawContact(contact, filter, solution, tex, base) {
  const { sites, strength, ends, sides, localMean, localSlope, gains, feature } = contact;
  const { weights, linearWeights, indices, terms } = filter;
  const { field, slope } = solution.directional.get(feature);
  let link = 0;
  const rgb = new Float64Array(3);
  for (let i = 0; i < sites.length; i++) {
    rgb.fill(0);
    for (; link < ends[i]; link++) {
      const j = sides[link] * 3;
      for (let c = 0; c < 3; c++)
        rgb[c] += localMean[link] * solution.outgoing[j + c] + localSlope[link] * solution.outSlope[j + c];
    }
    const site = sites[i];
    for (let t = site * terms; t < (site + 1) * terms; t++) {
      const cell = indices[t] * sectorCount;
      for (let d = 0; d < sectorCount; d++) {
        const gain = gains[i * sectorCount + d],
          w = gain * weights[t],
          m = gain * linearWeights[t];
        const j = (cell + d) * 3;
        for (let c = 0; c < 3; c++) rgb[c] += w * field[j + c] + m * slope[j + c];
      }
    }
    for (let c = 0; c < 3; c++) {
      const k = base + 4 * site + c;
      tex[k] += strength[i] * (rgb[c] - tex[k]);
    }
  }
}

// Recover directional moments of the distant light already carried by the
// sparse equilibrium operator. Source-centre direction is an approximation.
export function incidentDirections(
  contacts,
  modes,
  offsets,
  sourceIds,
  coefficients,
  sourceRGB,
  sourceSlope,
) {
  const directional = new Map();

  for (const feature of contacts) {
    const { offset: start, size, side, u, v, supports } = feature;
    const field = new Float32Array(size * sectorCount * 3),
      slope = new Float32Array(field.length);
    for (let y = 0; y < size; y++) {
      const i = 2 * (start + y) + side,
        center = modes[start + y].center;
      for (let k = offsets[i]; k < offsets[i + 1]; k++) {
        const p = 5 * k,
          j = sourceIds[p],
          source = modes[j >> 1];
        if (supports.has(source.rect)) continue;
        const d = sub(source.center, center);
        const az = (Math.atan2(dot(d, v), dot(d, u)) * azimuthCount) / (2 * Math.PI) + azimuthCount;
        const height = dot(d, feature.n),
          z2 = (height * height) / (dot(d, d) || 1);
        const upper = Math.max(0, Math.min(1, 2 * z2 - 0.5));
        const lo = Math.floor(az),
          f = az - lo;
        for (let band = 0; band < 2; band++) {
          const share = band ? upper : 1 - upper,
            first = y * sectorCount + band * azimuthCount;

          for (let c = 0; c < 3; c++) {
            const energy =
              coefficients[p + 1] * sourceRGB[3 * j + c] + coefficients[p + 2] * sourceSlope[3 * j + c];
            const gradient =
              3 * (coefficients[p + 3] * sourceRGB[3 * j + c] + coefficients[p + 4] * sourceSlope[3 * j + c]);
            slope[(first + (lo % azimuthCount)) * 3 + c] += share * (1 - f) * gradient;
            slope[(first + ((lo + 1) % azimuthCount)) * 3 + c] += share * f * gradient;
            field[(first + (lo % azimuthCount)) * 3 + c] += share * (1 - f) * energy;
            field[(first + ((lo + 1) % azimuthCount)) * 3 + c] += share * f * energy;
          }
        }
      }
    }
    for (let i = 0; i < slope.length; i++) slope[i] = Math.min(field[i], Math.max(-field[i], slope[i]));
    directional.set(feature, { field, slope });
  }
  return directional;
}
