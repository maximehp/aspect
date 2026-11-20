// Operations on radiance. Directions here point along the travel of light.
// An angular field integrates a scalar weight against an RGB radiance measure.
// A collimated emitter can thus contribute at a single direction without bins.
import { roundedBoxDistance } from './rounded-box.js';

const channels = [0, 1, 2];
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const at = (a, d, s) => a.map((v, i) => v + s * d[i]);

export function diffuseRadiance(reflectance, normal, incoming) {
  const irradiance = incoming((d) => Math.max(0, -dot(normal, d)));
  return reflectance.map((rho, i) => (rho * irradiance[i]) / Math.PI);
}

// Returns forward flux fractions. Radiance crossing the interface also changes
// by the squared index ratio because a refracted bundle changes solid angle.
export function dielectric(incident, outwardNormal, exteriorIndex, interiorIndex) {
  if (![exteriorIndex, interiorIndex].every((n) => Number.isFinite(n) && n > 0))
    throw RangeError('Indices of refraction must be finite and positive.');
  const outside = dot(incident, outwardNormal) <= 0;
  const normal = outside ? outwardNormal : outwardNormal.map((x) => -x);
  const ratio = outside ? exteriorIndex / interiorIndex : interiorIndex / exteriorIndex;
  const cosine = Math.min(1, Math.max(0, -dot(incident, normal)));
  const reflected = at(incident, normal, 2 * cosine);
  if (ratio === 1) return { reflected, transmitted: incident, R: 0, T: 1, radianceGain: 1 };
  const sin2 = ratio * ratio * (1 - cosine * cosine);
  if (sin2 >= 1) return { reflected, transmitted: null, R: 1, T: 0, radianceGain: 0 };
  const ct = Math.sqrt(1 - sin2);
  const rs = (ratio * cosine - ct) / (ratio * cosine + ct);
  const rp = (cosine - ratio * ct) / (cosine + ratio * ct);
  const R = (rs * rs + rp * rp) / 2;
  return {
    reflected,
    transmitted: at(
      incident.map((v) => ratio * v),
      normal,
      ratio * cosine - ct,
    ),
    R,
    T: 1 - R,
    radianceGain: (1 - R) / (ratio * ratio),
  };
}

export function phaseHG(g = 0) {
  if (!(Math.abs(g) < 1)) throw RangeError('HG anisotropy must lie strictly between -1 and 1.');
  if (g === 0) return () => 1 / (4 * Math.PI);
  return (incoming, outgoing) =>
    (1 - g * g) /
    (4 * Math.PI * (1 + g * g - 2 * g * Math.max(-1, Math.min(1, dot(incoming, outgoing)))) ** 1.5);
}

// RGB absorption and scattering coefficients per unit density, per metre.
// Optionally supply the integral of density along an authored ray segment.
export function medium({ density = 1, absorption, scattering, phase = phaseHG(), columnDensity }) {
  if ([absorption, scattering].some((s) => s?.length !== 3 || !s.every((x) => Number.isFinite(x) && x >= 0)))
    throw RangeError('Supply three finite nonnegative absorption and scattering coefficients.');
  if (typeof density !== 'function' && !(Number.isFinite(density) && density >= 0))
    throw RangeError('Density must be nonnegative.');
  return {
    density,
    absorption,
    scattering,
    extinction: absorption.map((v, i) => v + scattering[i]),
    phase,
    columnDensity,
  };
}

// Collect the radiance scattered into direction d per scattering event.
// incoming(weight) can contain lamp, surface and emerging volume radiance.
export function inScatter(material, d, incoming) {
  return incoming((w) => material.phase(w, d));
}

// Light enters at origin and exits at origin + distance * direction.
// The two coefficients describe an affine operation on any boundary radiance.
// Compose in propagation order, so emerging light is available to every consumer.
export function compose(first, second) {
  return {
    transmission: first.transmission.map((v, i) => v * second.transmission[i]),
    source: second.source.map((v, i) => v + second.transmission[i] * first.source[i]),
  };
}
export function emerge(segment, incident) {
  return incident.map((v, i) => segment.source[i] + segment.transmission[i] * v);
}

export function throughMedium(
  material,
  origin,
  direction,
  distance,
  scattered = [0, 0, 0],
  tolerance = 1e-6,
) {
  if (!(Number.isFinite(distance) && distance >= 0 && Number.isFinite(tolerance) && tolerance > 0))
    throw RangeError('A finite length and positive tolerance are required.');
  const { extinction, scattering, density, columnDensity } = material;
  const varyingSource = typeof scattered === 'function' && scattering.some((v) => v > 0);
  function segment(lo, hi) {
    const m = (lo + hi) / 2,
      p = at(origin, direction, m);
    const column = columnDensity
      ? columnDensity(at(origin, direction, lo), direction, hi - lo)
      : (hi - lo) * (typeof density === 'function' ? density(p) : density);
    if (!(Number.isFinite(column) && column >= 0)) throw RangeError('Invalid integrated density.');
    const S = varyingSource ? scattered(p, direction) : scattered;
    return {
      transmission: extinction.map((k) => Math.exp(-k * column)),
      source: channels.map((i) =>
        scattering[i] === 0
          ? 0
          : -Math.expm1(-extinction[i] * column) * (scattering[i] / extinction[i]) * S[i],
      ),
    };
  }
  if (distance === 0) return { transmission: [1, 1, 1], source: [0, 0, 0] };
  const whole = segment(0, distance);
  if (!varyingSource && (columnDensity || typeof density === 'number')) return whole;
  return refineSegment(segment, 0, distance, whole, tolerance);
}

function refineSegment(evaluate, begin, end, whole, tolerance) {
  function integrate(lo, hi, coarse, budget, depth) {
    const mid = (lo + hi) / 2,
      left = evaluate(lo, mid),
      right = evaluate(mid, hi),
      fine = compose(left, right);
    const error = Math.max(
      ...channels.map((i) =>
        Math.max(
          Math.abs(fine.transmission[i] - coarse.transmission[i]),
          Math.abs(fine.source[i] - coarse.source[i]),
        ),
      ),
    );
    if (error <= budget) return fine;
    if (depth === 0) throw RangeError('Medium segment refinement exhausted its error budget.');
    return compose(
      integrate(lo, mid, left, budget / 2, depth - 1),
      integrate(mid, hi, right, budget / 2, depth - 1),
    );
  }
  return integrate(begin, end, whole, tolerance, 22);
}

// Geometric support of a medium is independent of its material. This uses the
// same exact rounded solids as the room. Positive thickness separates a true
// crossing from a tangent. Either endpoint may already be inside.
function inside(box, p) {
  const q = p.map((v, i) => v - box.center[i]);
  return (
    box.axes.reduce(
      (sum, axis, i) => sum + Math.max(0, Math.abs(dot(q, axis)) - box.halfSize[i] + box.radius) ** 2,
      0,
    ) <=
    box.radius ** 2
  );
}
export function transportBetween(regions, origin, destination, incident, incoming, tolerance = 1e-6) {
  if (!(Number.isFinite(tolerance) && tolerance > 0)) throw RangeError('A positive tolerance is required.');
  const delta = destination.map((v, i) => v - origin[i]),
    length = Math.hypot(...delta);
  if (!length || !regions.length) return incident;
  const direction = delta.map((v) => v / length),
    reverse = direction.map((v) => -v);
  const spans = [];
  for (const { geometry, material } of regions) {
    const lo = inside(geometry, origin) ? 0 : roundedBoxDistance(geometry, origin, direction, 0, length);
    const hi = inside(geometry, destination)
      ? length
      : length - roundedBoxDistance(geometry, destination, reverse, 0, length);
    if (hi > lo) spans.push({ lo, hi, material });
  }
  const cuts = [...new Set(spans.flatMap((v) => [v.lo, v.hi]))].sort((a, b) => a - b);
  let light = incident;
  for (let k = 1; k < cuts.length; k++) {
    const lo = cuts[k - 1],
      hi = cuts[k],
      active = spans.filter((s) => s.lo < hi && s.hi > lo).map((s) => s.material);
    if (!active.length) continue;
    const budget = (tolerance * (hi - lo)) / length;
    // A constant incoming angular measure may still vary with outgoing
    // direction through the phase function. It stays constant on this ray.
    const sources = active.map((m) =>
      !incoming
        ? [0, 0, 0]
        : typeof incoming === 'function'
          ? (p) => inScatter(m, direction, (weight) => incoming(p, weight))
          : inScatter(m, direction, incoming.uniform),
    );
    let segment;
    if (active.length === 1)
      segment = throughMedium(active[0], at(origin, direction, lo), direction, hi - lo, sources[0], budget);
    else {
      function combine(a, b) {
        const p = at(origin, direction, (a + b) / 2),
          start = at(origin, direction, a),
          tau = [0, 0, 0],
          q = [0, 0, 0];
        active.forEach((m, j) => {
          const column = m.columnDensity
            ? m.columnDensity(start, direction, b - a)
            : (b - a) * (typeof m.density === 'function' ? m.density(p) : m.density);
          if (!(Number.isFinite(column) && column >= 0)) throw RangeError('Invalid integrated density.');
          const S = typeof sources[j] === 'function' ? sources[j](p) : sources[j];
          for (const c of channels) {
            tau[c] += column * m.extinction[c];
            q[c] += column * m.scattering[c] * S[c];
          }
        });
        return {
          transmission: tau.map((t) => Math.exp(-t)),
          source: tau.map((t, c) => (t ? (-Math.expm1(-t) * q[c]) / t : q[c])),
        };
      }
      segment = combine(lo, hi);
      if (typeof incoming === 'function' || active.some((m) => typeof m.density === 'function'))
        segment = refineSegment(combine, lo, hi, segment, budget);
    }
    light = emerge(segment, light);
  }
  return light;
}

// Bind point and collimated emitters to an incident angular measure. The latter
// uses irradiance on a plane perpendicular to its unit travel direction.
// Call again when the authored scene changes. All visibility uses current solids.
export function emitterField({
  rectangles = [],
  roundedBoxes = [],
  media = [],
  emitters,
  lightPosition,
  lampRGB,
}) {
  const sources = emitters ?? [{ position: lightPosition, intensity: lampRGB }];
  const sheets = rectangles.map((s) => {
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const n = cross(s.axisU, s.axisV),
      area2 = dot(n, n);
    return {
      center: s.center,
      n,
      u: cross(s.axisV, n).map((x) => x / area2),
      v: cross(n, s.axisU).map((x) => x / area2),
    };
  });
  return (p, weight) => {
    const light = [0, 0, 0];
    for (const source of sources) {
      let start = source.position,
        d,
        distance,
        flux;
      if (start) {
        const delta = p.map((v, i) => v - start[i]);
        distance = Math.hypot(...delta);
        if (!distance) throw RangeError('Point emitter and measurement position coincide.');
        d = delta.map((v) => v / distance);
        flux = source.intensity.map((v) => v / (distance * distance));
      } else {
        d = source.direction;
        flux = source.irradiance;
        // Start beyond every potentially intersecting support plane.
        let extent = 0;
        for (const s of rectangles)
          extent = Math.max(
            extent,
            dot(
              d,
              p.map((v, i) => v - s.center[i]),
            ) +
              Math.abs(dot(d, s.axisU)) +
              Math.abs(dot(d, s.axisV)),
          );
        for (const s of [...roundedBoxes, ...media.map((m) => m.geometry)])
          extent = Math.max(
            extent,
            dot(
              d,
              p.map((v, i) => v - s.center[i]),
            ) +
              s.radius +
              s.axes.reduce((sum, a, i) => sum + Math.abs(dot(d, a)) * (s.halfSize[i] - s.radius), 0),
          );
        distance = Math.max(0, extent) + 1;
        start = at(p, d, -distance);
      }
      const angular = weight(d);
      if (!angular) continue;
      const hidden =
        sheets.some((s) => {
          const den = dot(s.n, d);
          if (!den) return false;
          const t =
            dot(
              s.n,
              s.center.map((v, i) => v - start[i]),
            ) / den;
          if (t <= 0 || t >= distance - 1e-10) return false;
          const rel = start.map((v, i) => v + t * d[i] - s.center[i]);
          return Math.abs(dot(rel, s.u)) <= 1 && Math.abs(dot(rel, s.v)) <= 1;
        }) || roundedBoxes.some((s) => roundedBoxDistance(s, start, d, 0, distance - 1e-10) < distance);
      if (hidden) continue;
      const transmitted = transportBetween(media, start, p, flux);
      for (const c of channels) light[c] += angular * transmitted[c];
    }
    return light;
  };
}
