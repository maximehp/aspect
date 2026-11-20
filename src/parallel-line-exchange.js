// Transfer a spatial field between parallel planes. Sheets spanning their gap
// become opaque segments in projection. One projected line describes every
// source/receiver pair on that line, including the ones on either side of a wall.
import { lineMomentsVertex } from './line-moments.js';
import { AngularMeasure } from './angular-measure.js';
import { exact, subQ, addQ, mulQ } from './authored-scalar.js';

const vertex = `#version 300 es
precision highp float;
layout(location=0) in vec3 section; // source half-length, plane gap, quadrature weight
layout(location=1) in vec3 ends; // receiver ends relative to source midpoint, receiver length
layout(location=2) in vec3 uhat;
layout(location=3) in vec3 vhat;
out vec4 measure;
// The two normalized distance quadratics have constant coefficient one.
// Expand their rational product about the segment midpoint and contract
// its antiderivative directly with the first three polynomial moments.
void main() {
  float w=section.x,h=section.y,A=ends.x,B=ends.y;
  vec2 d=vec2(A,B),den=d*d+h*h;
  vec2 a=-2.*w*d/den, b=w*w/den;
  vec4 q=vec4(a.x+a.y,b.x+b.y+a.x*a.y,a.x*b.y+b.x*a.y,b.x*b.y);
  vec4 n=vec4(0.); float n4=0.;
  if(A*B>=0.){
    float k=(h*h/den.x)*(ends.z/den.y);
    n.xy=k*vec2(abs(A+B),-2.*w*sign(A+B));
  }else{
    vec2 c=d*d/den;
    n=vec4(c.x+c.y, dot(c,a.yx)+a.x+a.y,
      dot(c,b.yx)+2.*a.x*a.y+b.x+b.y,2.*(a.x*b.y+b.x*a.y));
    n4=2.*b.x*b.y;
  }
  vec4 prev=vec4(0.);vec3 moments=vec3(0.);
  for(int i=0;i<=20;i++){
    float f=(i<4?n[min(3,i)]:(i==4?n4:0.))-dot(q,prev);
    prev=vec4(f,prev.xyz);
    if((i&1)==0)moments.xz+=2.*f/vec2(i+1,i+3);
    else moments.y+=2.*f/float(i+2);
    if(float(i)>=uhat.z)break;
  }
  vec2 u=uhat.xy,v=vhat.xy;
  // Each of the four tensor hats becomes a quadratic on this line.
  vec2 U=vec2(1.-u.x,u.x), dU=vec2(-u.y,u.y);
  vec2 V=vec2(1.-v.x,v.x), dV=vec2(-v.y,v.y);
  measure=w*section.z*(U.xyxy*V.xxyy*moments.x+
    (U.xyxy*dV.xxyy+dU.xyxy*V.xxyy)*moments.y+dU.xyxy*dV.xxyy*moments.z);
  gl_Position=vec4(0,0,0,1);gl_PointSize=1.;
}`;

const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
const nodes = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526];
const weights = [0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539];
const endsOf = (c, v) => [-1, 1].map((s) => c.map((x, i) => x + s * v[i]));
function frame(rect, project, bounds) {
  const c = project(rect.center),
    a = project(rect.axisU),
    b = project(rect.axisV),
    det = cross(a, b);
  const du = [b[1] / (2 * det), -b[0] / (2 * det)],
    dv = [-a[1] / (2 * det), a[0] / (2 * det)];
  const corners = [bounds[0], bounds[1]].flatMap((u) =>
    [bounds[2], bounds[3]].map((v) => c.map((x, i) => x + (2 * u - 1) * a[i] + (2 * v - 1) * b[i])),
  );
  return {
    c,
    du,
    dv,
    bounds,
    corners,
    area: 4 * Math.abs(det) * (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]),
  };
}
function coordinateRange(f, direction, origin) {
  const p = origin.map((x, k) => x - f.c[k]);
  let lo = -Infinity,
    hi = Infinity;
  const coord = [f.du, f.dv].map((d) => [0.5 + dot(p, d), dot(direction, d)]);
  for (let j = 0; j < 2; j++) {
    const [c, d] = coord[j],
      l = f.bounds[2 * j],
      h = f.bounds[2 * j + 1];
    if (d === 0) {
      if (c < l || c > h) return null;
      continue;
    }
    const a = (l - c) / d,
      b = (h - c) / d;
    lo = Math.max(lo, Math.min(a, b));
    hi = Math.min(hi, Math.max(a, b));
  }
  return hi > lo ? { lo, hi, coord } : null;
}
function cell(knots, t) {
  let lo = 0,
    hi = knots.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t < knots[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}

// Both footprints on the same side of a projected wall cannot exchange a
// ray across it. Use authored binary coordinates for a doubtful contact.
function sideOfScreen(shape, center, edge, project) {
  const c = project(shape.center),
    u = project(shape.axisU),
    v = project(shape.axisV);
  const m = cross(
      edge,
      c.map((x, k) => x - center[k]),
    ),
    r = Math.abs(cross(edge, u)) + Math.abs(cross(edge, v));
  const guard =
    1e-12 *
    Math.hypot(...edge) *
    (1 + Math.hypot(...c) + Math.hypot(...center) + Math.hypot(...u) + Math.hypot(...v));
  if (m - r > guard) return 1;
  if (m + r < -guard) return 2;
  if (Math.abs(m) < r - guard) return 0;
  const [a, b] = edge.map(exact);
  const sign = (p) => subQ(mulQ(a, p[1]), mulQ(b, p[0]));
  const abs = (x) => (x.n < 0n ? { n: -x.n, d: x.d } : x);
  const mid = sign(c.map((x, k) => subQ(exact(x), exact(center[k]))));
  const radius = addQ(abs(sign(u.map(exact))), abs(sign(v.map(exact))));
  return Number(subQ(mid, radius).n >= 0n) | (Number(addQ(mid, radius).n <= 0n) << 1);
}
// Endpoints of the authored sheet are compared before rounding their sums.
// An almost-spanning screen with a real gap needs ordinary 3D visibility.
function screens(shapes, source, receiver, axis, project, corners) {
  const l = exact(Math.min(source.center[axis], receiver.center[axis]));
  const h = exact(Math.max(source.center[axis], receiver.center[axis]));
  const result = [];
  const span = [0, 1].map((k) => [
    Math.min(...corners.map((p) => p[k])),
    Math.max(...corners.map((p) => p[k])),
  ]);
  for (const s of shapes) {
    if (s === source || s === receiver) continue;
    if (
      span.some(([l, h], k) => {
        const j = (axis + k + 1) % 3,
          c = s.center[j],
          extent = s.axes ? Math.hypot(...s.halfSize) : Math.abs(s.axisU[j]) + Math.abs(s.axisV[j]);
        const pad = 1e-12 * (1 + Math.abs(c) + extent + Math.abs(l) + Math.abs(h));
        return c + extent < l - pad || c - extent > h + pad;
      })
    )
      continue;
    if (s.axes) {
      const low = Math.min(source.center[axis], receiver.center[axis]),
        high = Math.max(source.center[axis], receiver.center[axis]);
      const pad =
        1e-12 * (1 + Math.abs(low) + Math.abs(high) + Math.abs(s.center[axis]) + Math.hypot(...s.halfSize));
      if (s.bounds.min[axis] > high + pad || s.bounds.max[axis] < low - pad) continue;
      return null;
    }
    const origin = exact(s.center[axis]),
      a = Math.abs(s.axisU[axis]),
      b = Math.abs(s.axisV[axis]);
    const radius = addQ(exact(a), exact(b)),
      bottom = subQ(origin, radius),
      top = addQ(origin, radius);
    if (subQ(bottom, h).n >= 0n || subQ(top, l).n <= 0n) continue;
    const vertical = a ? s.axisU : s.axisV,
      flat = a ? s.axisV : s.axisU;
    if (flat[axis] !== 0 || project(vertical).some((x) => x !== 0)) return null;
    const center = project(s.center),
      edge = project(flat);
    if (sideOfScreen(source, center, edge, project) & sideOfScreen(receiver, center, edge, project)) continue;
    if (subQ(bottom, l).n > 0n || subQ(top, h).n < 0n) return null;
    result.push(endsOf(center, edge));
  }
  return result;
}

export function parallelLineDomain(geometry, source, receiver, bounds = [0, 1, 0, 1]) {
  if (source?.type !== 0 || receiver?.type !== 0) return null;
  if (
    bounds.length !== 4 ||
    !bounds.every((x) => Number.isFinite(x) && x >= 0 && x <= 1) ||
    bounds[0] >= bounds[1] ||
    bounds[2] >= bounds[3]
  )
    throw RangeError('Expected a finite receiver rectangle');
  const S = geometry.shapes[source.object],
    R = geometry.shapes[receiver.object];
  if (!S.axisU || !R.axisU) return null;
  const axis = [0, 1, 2].find(
    (i) => S.axisU[i] === 0 && S.axisV[i] === 0 && R.axisU[i] === 0 && R.axisV[i] === 0,
  );
  if (axis === undefined) return null;
  const gap = R.center[axis] - S.center[axis];
  if (!(gap * source.n[axis] > 0 && gap * receiver.n[axis] < 0)) return null;
  const project = (v) => [v[(axis + 1) % 3], v[(axis + 2) % 3]],
    h = Math.abs(gap);
  const src = frame(S, project, [0, 1, 0, 1]),
    dst = frame(R, project, bounds);
  const edges = screens(geometry.shapes, S, R, axis, project, src.corners.concat(dst.corners));
  if (!edges || !Number.isFinite(src.area) || !Number.isFinite(dst.area) || !(src.area > 0 && dst.area > 0))
    return null;
  return { src, dst, edges, h };
}

export function parallelLineExchange(
  gl,
  geometry,
  source,
  receiver,
  bounds = [0, 1, 0, 1],
  { moments = false } = {},
) {
  const domain = parallelLineDomain(geometry, source, receiver, bounds);
  if (!domain) return null;
  const { src, dst, edges, h } = domain;
  const tests = moments ? 4 : 1;
  const gpu = new AngularMeasure(gl, {
      apertureVertex: moments ? lineMomentsVertex : vertex,
      components: 4 * tests,
      attributeSize: moments ? 4 : 3,
    }),
    capacity = 16384;
  gpu.reserve(capacity);
  const keys = new Int32Array(capacity),
    stride = source.cols + 1,
    offsets = [0, 1, stride, stride + 1];
  // On the pencil of lines with direction e, topology changes at projected
  // endpoints or intersections. Endpoint order in offset swaps precisely
  // when two of these points project to the same offset.
  const outline = (f) => [0, 1, 3, 2].map((j, k) => [f.corners[j], f.corners[[1, 3, 2, 0][k]]]);
  const segments = [...edges, ...outline(src), ...outline(dst)];
  const points = segments.flat();
  for (let i = 0; i < segments.length; i++)
    for (let j = 0; j < i; j++) {
      const [a, b] = segments[i],
        [c, d] = segments[j];
      const ab = b.map((x, k) => x - a[k]),
        cd = d.map((x, k) => x - c[k]),
        ac = c.map((x, k) => x - a[k]);
      const det = cross(ab, cd);
      if (!det) continue;
      const t = cross(ac, cd) / det,
        v = cross(ac, ab) / det;
      if (t > 0 && t < 1 && v > 0 && v < 1) points.push(a.map((x, k) => x + t * ab[k]));
    }
  const features = [...new Map(points.map((p) => [p.join(','), p])).values()];
  const angleSet = new Set([0, Math.PI]);
  for (let i = 0; i < features.length; i++)
    for (let j = 0; j < i; j++) {
      const p = features[i],
        q = features[j];
      let phi = Math.atan2(p[1] - q[1], p[0] - q[0]);
      if (phi < 0) phi += Math.PI;
      angleSet.add(phi);
    }
  const events = [...angleSet].sort((a, b) => a - b);
  return {
    dispose: () => gpu.dispose(),
    solve(tolerance, { maxAngles = 128, maxOffsetPanels = 8 } = {}) {
      if (!(Number.isFinite(tolerance) && tolerance > 0))
        throw RangeError('Expected a positive absolute coefficient tolerance');
      if (![maxAngles, maxOffsetPanels].every(Number.isSafeInteger) || maxAngles < 16 || maxOffsetPanels < 2)
        throw RangeError('Expected at least one refinement in each coordinate');
      let a = 8,
        p = 1,
        coarse = this.integrate(a, p);
      const difference = (x, y) => x.coefficients.reduce((s, v, i) => s + Math.abs(v - y.coefficients[i]), 0);
      for (;;) {
        const angular = this.integrate(2 * a, p),
          fine = this.integrate(2 * a, 2 * p);
        const angleError = difference(angular, coarse),
          offsetError = difference(fine, angular);
        const converged = angleError + offsetError <= tolerance;
        if (converged || 4 * a > maxAngles || 4 * p > maxOffsetPanels)
          return { ...fine, angles: 2 * a, offsetPanels: 2 * p, angleError, offsetError, converged };
        coarse = fine;
        a *= 2;
        p *= 2;
      }
    },
    integrate(angles, offsetPanels = 1) {
      if (![angles, offsetPanels].every((n) => Number.isSafeInteger(n) && n > 0))
        throw RangeError('Expected positive panel counts');
      const coefficients = new Float64Array(source.ids.length * tests);
      let used = 0,
        jobs = 0,
        peakDeviceBytes = 0,
        lines = 0,
        blocks = 0;
      function flush() {
        if (!used) return;
        gpu.integrate(used);
        peakDeviceBytes = Math.max(peakDeviceBytes, gpu.inputBytes + gpu.outputBytes);
        for (let i = 0; i < used; i++)
          for (let k = 0; k < 4; k++)
            for (let t = 0; t < tests; t++)
              coefficients[tests * (keys[i] + offsets[k]) + t] += gpu.values[tests * i * 4 + 4 * t + k];
        jobs += used;
        used = 0;
      }
      // A source coordinate crosses a knot at most once on the whole line.
      function line(e, normal, p, stops, weight) {
        const origin = normal.map((x) => x * p),
          s = coordinateRange(src, e, origin),
          t = coordinateRange(dst, e, origin);
        if (!s || !t) return;
        lines++;
        const cuts = [s.lo, s.hi],
          walls = [];
        for (const [pl, ph, along, slope] of stops)
          if (p > pl && p < ph) walls.push(along + (p - pl) * slope);
        walls.sort((a, b) => a - b);
        for (const cut of [t.lo, t.hi, ...walls]) if (cut > s.lo && cut < s.hi) cuts.push(cut);
        for (let d = 0; d < 2; d++) {
          const [base, slope] = s.coord[d];
          if (slope === 0) continue;
          for (const knot of source.knots[d].slice(1, -1)) {
            const cut = (knot - base) / slope;
            if (cut > s.lo && cut < s.hi) cuts.push(cut);
          }
        }
        cuts.sort((a, b) => a - b);
        let j = 0;
        for (let i = 1; i < cuts.length; i++) {
          const a = cuts[i - 1],
            b = cuts[i],
            mid = (a + b) / 2;
          if (!(mid > a && mid < b)) continue;
          while (j < walls.length && walls[j] <= mid) j++;
          const lo = Math.max(t.lo, j ? walls[j - 1] : -Infinity),
            hi = Math.min(t.hi, j < walls.length ? walls[j] : Infinity);
          if (!(hi > lo)) {
            blocks++;
            continue;
          }
          const ij = s.coord.map(([v, d], k) => cell(source.knots[k], v + d * mid));
          function emit(a, b) {
            const m = (a + b) / 2,
              w = (b - a) / 2;
            const ratio = w / Math.hypot(h, Math.min(Math.abs(lo - m), Math.abs(hi - m)));
            if (ratio > 0.2) {
              emit(a, m);
              emit(m, b);
              return;
            }
            const uv = s.coord.map(([v, d], k) => {
              const knots = source.knots[k],
                i = ij[k],
                range = knots[i + 1] - knots[i];
              return [(v + d * m - knots[i]) / range, (d * w) / range, 0];
            });
            uv[0][2] = ratio < 0.016 ? 8 : ratio < 0.05 ? 12 : ratio < 0.12 ? 16 : 20;
            keys[used] = ij[0] + stride * ij[1];
            gpu.packed.set([w, h, weight, lo - m, hi - m, hi - lo, ...uv[0], ...uv[1]], 12 * used++);
            if (used === capacity) flush();
          }
          if (!moments) {
            emit(a, b);
            continue;
          }
          // The convolution interval in s changes owner at these differences.
          const breaks = [lo - b, lo - a, hi - b, hi - a].sort((x, y) => x - y);
          if (breaks[0] < 0 && breaks[3] > 0) {
            breaks.push(0);
            breaks.sort((x, y) => x - y);
          }
          const sourceUV = s.coord.map(([v, d], k) => {
            const knots = source.knots[k],
              i = ij[k],
              range = knots[i + 1] - knots[i];
            return [(v + d * mid - knots[i]) / range, d / range];
          });
          function pair(start, end) {
            const center = (start + end) / 2,
              half = (end - start) / 2;
            if (!(half > 0)) return;
            if (half > 0.2 * Math.hypot(center, h)) {
              pair(start, center);
              pair(center, end);
              return;
            }
            const intervals = [start, end].flatMap((z) => [
              Math.max(a, lo - z) - mid,
              Math.min(b, hi - z) - mid,
            ]);
            const destUV = t.coord.map(([v, d], k) => {
              const base = bounds[2 * k],
                range = bounds[2 * k + 1] - base;
              return [(v + d * (mid + center) - base) / range, d / range];
            });
            keys[used] = ij[0] + stride * ij[1];
            gpu.packed.set(
              [
                center,
                half,
                h,
                2 * weight,
                ...intervals,
                sourceUV[0][0],
                sourceUV[1][0],
                sourceUV[0][1],
                sourceUV[1][1],
                destUV[0][0],
                destUV[1][0],
                destUV[0][1],
                destUV[1][1],
              ],
              16 * used++,
            );
            if (used === capacity) flush();
          }
          for (let j = 1; j < breaks.length; j++) pair(breaks[j - 1], breaks[j]);
        }
      }
      function direction(theta, angleWeight) {
        const e = [Math.cos(theta), Math.sin(theta)],
          n = [-e[1], e[0]];
        const projections = dst.corners.map((x) => dot(n, x)),
          lo = Math.min(...projections),
          hi = Math.max(...projections);
        const stops = edges.map((edge) => {
          let [a, b] = edge,
            pl = dot(a, n),
            ph = dot(b, n);
          if (ph < pl) {
            [a, b] = [b, a];
            [pl, ph] = [ph, pl];
          }
          return [pl, ph, dot(a, e), ph > pl ? (dot(b, e) - dot(a, e)) / (ph - pl) : 0];
        });
        const cuts = [lo, hi, ...features.map((x) => dot(n, x)).filter((x) => x > lo && x < hi)].sort(
          (a, b) => a - b,
        );
        for (let j = 1; j < cuts.length; j++) {
          const span = (cuts[j] - cuts[j - 1]) / offsetPanels;
          if (!span) continue;
          for (let k = 0; k < offsetPanels; k++)
            for (let q = 0; q < nodes.length; q++) {
              const p = cuts[j - 1] + span * (k + (1 + nodes[q]) / 2);
              line(e, n, p, stops, (weights[q] * span * angleWeight) / (4 * Math.PI * dst.area));
            }
        }
      }
      for (let j = 1; j < events.length; j++) {
        const steps = Math.ceil(((events[j] - events[j - 1]) * angles) / Math.PI);
        if (!steps) continue;
        const width = (events[j] - events[j - 1]) / steps;
        for (let i = 0; i < steps; i++) {
          const theta = events[j - 1] + width * (i + 0.5);
          if (theta > events[j - 1] && theta < events[j]) direction(theta, width);
        }
      }
      flush();
      return {
        coefficients,
        orientationEvents: events.length,
        lines,
        jobs,
        blockedIntervals: blocks,
        angularPayloadBytes:
          gpu.packed.byteLength +
          gpu.values.byteLength +
          peakDeviceBytes +
          keys.byteLength +
          coefficients.byteLength,
        apply(field, test) {
          if (test !== undefined && (!Number.isInteger(test) || test < 0 || test >= tests))
            throw RangeError('Unknown receiver weight');
          const rgb = [0, 0, 0];
          source.ids.forEach((id, i) => {
            if (id < 0) return;
            let w = 0;
            for (let t = test ?? 0; t < (test === undefined ? tests : test + 1); t++)
              w += coefficients[i * tests + t];
            for (let c = 0; c < 3; c++) rgb[c] += w * field[3 * id + c];
          });
          return rgb;
        },
      };
    },
  };
}
