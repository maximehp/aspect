// Integrate all four receiver hats across each lit trapezoid in its own UV
// coordinates. At a fixed v the two limits are affine. A straight generator
// integrates analytically, including its two linear endpoint functions.
import { linearGenerator, gauss8 } from './rectangle-aperture.js';

export const weightedApertureVertex = `#version 300 es
precision highp float;
layout(location=0) in vec3 displacement;
layout(location=1) in vec3 transverse;
layout(location=2) in vec3 lowerEdge;
layout(location=3) in vec3 upperEdge;
uniform vec3 extent; // |U|, component of V along U, perpendicular component of V
out vec4 measure;
${gauss8}
${linearGenerator}
vec4 integrateSlice(float t) {
  vec3 section = mix(lowerEdge, upperEdge, t); // left u, right u, v
  float a = section.x, b = section.y, v = section.z, halfWidth = .5*(b-a)*extent.x;
  float mid = .5*(a+b);
  vec2 xy = displacement.xy + vec2(mid*extent.x + v*extent.y, v*extent.z);
  vec2 w = lengthWeights(xy, halfWidth, displacement.z, transverse.x,
                         transverse.z + mid*extent.x*transverse.x + v*transverse.y);
  // Map the integrals of the two *local* lengthwise hats onto the
  // full receiver hats. This preserves gradients across clipped boundaries.
  float right = .5*dot(w, vec2(1.+a,1.+b));
  vec2 along = vec2(.5*dot(w, vec2(1.-a,1.-b)),right);
  return vec4((1.-v)*along,(1.+v)*along);
}
void main() {
  vec4 value=vec4(0.);
  for(int j=0;j<8;j++) value+=weights8[j]*integrateSlice(.5*(1.+nodes8[j]));
  measure=value*((upperEdge.z-lowerEdge.z)*extent.z/(4.*3.14159265359));
  gl_Position=vec4(0,0,0,1); gl_PointSize=1.;
}`;

// The intersections of straight edges with a horizontal line vary affinely
// between consecutive vertices. For each nonempty band emit both endpoints.
export function apertureBands(cell, emit) {
  const levels = [...new Set(cell.map((p) => p[1]))].sort((a, b) => a - b);
  for (let j = 1; j < levels.length; j++) {
    const lo = levels[j - 1],
      hi = levels[j];
    let al = Infinity,
      ar = -Infinity,
      bl = Infinity,
      br = -Infinity;
    for (let i = 0; i < cell.length; i++) {
      const a = cell[i],
        b = cell[(i + 1) % cell.length];
      if (Math.min(a[1], b[1]) > lo || Math.max(a[1], b[1]) < hi) continue;
      const slope = (b[0] - a[0]) / (b[1] - a[1]);
      const at = (v) => (v === a[1] ? a[0] : v === b[1] ? b[0] : a[0] + slope * (v - a[1]));
      const x = at(lo),
        y = at(hi);
      al = Math.min(al, x);
      ar = Math.max(ar, x);
      bl = Math.min(bl, y);
      br = Math.max(br, y);
    }
    if (ar > al || br > bl) emit([al, ar, lo], [bl, br, hi]);
  }
}

// For a fixed fraction along the segment, position is affine in v. Its squared
// distance D is quadratic. On the Bernstein ellipse of radius four, each
// complex root of D is at least |midpoint|/|half-step| from zero. This bounds
// 1/D² on the ellipse. Its degree-11 Chebyshev truncation times the nonnegative
// quartic numerator integrates exactly with eight nodes. Bound the remainder
// for both the integral and the rule against their common numerator mass.
export function boundedBand(lower, upper, mid, extent, tolerance, emit, other = null, depth = 0) {
  const [w, shear, height] = extent;
  const middle = lower.map((x, k) => (x + upper[k]) * 0.5);
  const distance = (origin) => {
    const a = origin[0] + middle[2] * shear;
    return Math.hypot(
      Math.max(0, a + middle[0] * w, -a - middle[1] * w),
      origin[1] + middle[2] * height,
      origin[2],
    );
  };
  const dv = (upper[2] - lower[2]) * 0.5;
  const dx = Math.max(
    Math.abs(0.5 * (upper[0] - lower[0]) * w + dv * shear),
    Math.abs(0.5 * (upper[1] - lower[1]) * w + dv * shear),
  );
  const step = Math.hypot(dx, dv * height);
  const nearest = typeof other === 'number' ? other : other ? distance(other) : Infinity;
  const ratio = step / Math.min(distance(mid), nearest);
  const separation = 1 - ((4 + 1 / 4) * ratio) / 2,
    power = other ? 8 : 4;
  if (separation > 0 && 4 * (1 + ratio) ** power <= tolerance * 3 * 4 ** 11 * separation ** power) {
    emit(lower, upper);
    return true;
  }
  if (depth >= 12) return false;
  return (
    boundedBand(lower, middle, mid, extent, tolerance, emit, other, depth + 1) &&
    boundedBand(middle, upper, mid, extent, tolerance, emit, other, depth + 1)
  );
}

// Products of two diffuse kernels on the same plane. Both squared distances
// are quadratic in the straight coordinate. Integrate the quadratic numerator
// against the series for their reciprocal product. No difference of nearly
// coincident poles is needed when the two eyes approach the same position.
// Cauchy tail for the integrated product through degree N. On |z|=B,
// each linear cosine is bounded by (1+B) times its midpoint value, and
// their mean product on [-1,1] is at least 2/3 of their midpoint product.
// Convert the permitted half-step at the midpoint to one at the left end.
export function generatorSteps(tolerance) {
  if (!(tolerance > 0 && Number.isFinite(tolerance)))
    throw RangeError('Expected a positive finite tolerance');
  return [8, 16, 24].map((order) => {
    let radius = 4,
      logA;
    do {
      logA =
        (Math.log(tolerance) -
          Math.log(1.5) -
          2 * Math.log1p(radius) +
          (order + 1) * Math.log(radius) +
          Math.log1p(-1 / radius)) /
        8;
      if (logA > Math.LN2) break;
      radius *= 2;
    } while (Number.isFinite(radius));
    const inv = Math.exp(-logA),
      eta = (1 - inv) / (radius + inv);
    return eta / (1 + eta);
  });
}
export const composedApertureVertex = `#version 300 es
precision highp float;
layout(location=0) in vec3 displacement;
layout(location=1) in vec3 transverse;
layout(location=2) in vec3 lowerEdge;
layout(location=3) in vec3 upperEdge;
uniform vec3 extent, nextDisplacement, nextTilt, halfSteps;
out float measure;
${gauss8}
float generator(vec3 band) {
  float v=band.z;
  vec2 base=vec2(displacement.x,nextDisplacement.x)+v*extent.y;
  vec2 y=vec2(displacement.y,nextDisplacement.y)+v*extent.z;
  vec2 z=vec2(displacement.z,nextDisplacement.z), g2=y*y+z*z;
  vec2 slope=vec2(transverse.x,nextTilt.x), atZero=vec2(transverse.z,nextTilt.z)+v*vec2(transverse.y,nextTilt.y);
  float lo=band.x*extent.x, hi=band.y*extent.x, value=0.;
  for(int piece=0;piece<256 && lo<hi;piece++) {
    vec2 offset=base+lo;
    vec2 radius=sqrt(offset*offset+g2);
    float nearest=min(radius.x,radius.y);
    float end=min(hi,lo+2.*halfSteps.z*nearest), h=.5*(end-lo), mid=lo+h;
    int order=h<=halfSteps.x*nearest?8:h<=halfSteps.y*nearest?16:24;
    if(end<=lo) return -1.;
    vec2 x=base+mid, D=x*x+g2, u=2.*x*h/D, t=h*h/D;
    vec4 q=vec4(u.x+u.y,t.x+t.y+u.x*u.y,u.x*t.y+u.y*t.x,t.x*t.y);
    vec2 c=atZero+slope*mid, b=slope*h;
    vec3 n=vec3(c.x*c.y,c.x*b.y+c.y*b.x,b.x*b.y);
    vec4 prev=vec4(1,0,0,0);
    float total=n.x;
    for (int i=1; i<=order; i++) {
      float f=-dot(q*(vec4(1,2,3,4)+float(i)),prev)/float(i);
      if ((i&1)==0) total+=(n.x*f+n.y*prev.x+n.z*prev.y)/float(i+1);
      prev=vec4(f,prev.xyz);
    }
    value+=2.*h*total/(D.x*D.x*D.y*D.y);
    lo=end;
  }
  return lo<hi ? -1. : value*z.x*z.y;
}
void main() {
  // A clipped curve supplies one exact open interval at a Gauss ordinate.
  // Its transverse weight is packed in upperEdge.x; regular bands have |z|<=1.
  if(upperEdge.z < -2.) {
    measure=generator(lowerEdge)*upperEdge.x*extent.z/(3.14159265359*3.14159265359);
    gl_Position=vec4(0,0,0,1);gl_PointSize=1.;return;
  }
  float value=0.;
  for(int i=0;i<8;i++){
    float w=generator(mix(lowerEdge,upperEdge,.5*(1.+nodes8[i])));
    if(w<0.) {value=-1.;break;}
    value+=weights8[i]*w;
  }
  measure=value*((upperEdge.z-lowerEdge.z)*extent.z/(2.*3.14159265359*3.14159265359));
  gl_Position=vec4(0,0,0,1);gl_PointSize=1.;
}`;
