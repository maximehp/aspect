import { inverseTangentCoefficients } from './inverse-tangent.js';
// Angular integrals of the four bilinear source functions on a clear rectangle.
// Both source coordinates integrate to endpoint functions of receiver position.
export const rectangleAperture = `
float ti2(float z) {
  float q = z * z / (1. + z * z), s = 0.;
  ${Array.from(inverseTangentCoefficients(20))
    .reverse()
    .map((x) => 's=' + x.toFixed(10) + '+q*s;')
    .join('\n  ')}
  return z * s / (1. + z * z);
}
float potential(float x, float y, float h) {
  vec2 t = abs(vec2(x, y)) / h;
  vec2 v = log(t + sqrt(1. + t * t));
  return sign(x) * sign(y) *
         (1.57079632679 * min(v.x, v.y) - ti2(exp(-abs(v.x - v.y))) + ti2(exp(-v.x - v.y)));
}
vec2 edgeIntegrals(vec2 endpoints, vec2 interval, float h) {
  vec2 a = sqrt(h * h + endpoints * endpoints);
  return atan(a * (interval.y - interval.x), a * a + interval.x * interval.y) / a;
}
// Integrate the quadratic expansion of 1/D² against each bilinear hat.
// For a hat with sign e, its normalised moments are 1/(k+1) for even k
// and e/(k+2) for odd k. Products of X and Y moments separate.
void axisMoments(vec4 e, float a, float b, out vec4 m, out vec4 um, out vec4 m2, out vec4 um2) {
  m = (a * e + b) / 3.;
  um = a / 3. + b * e / 5.;
  m2 = a * a / 3. + (2. * a * b * e + b * b) / 5.;
  um2 = (a * a * e + 2. * a * b) / 5. + b * b * e / 7.;
}
vec4 quadraticField(vec2 mid, vec2 halfSize, float D, vec3 light, float scale) {
  vec4 u = vec4(-1, 1, -1, 1), v = vec4(-1, -1, 1, 1), A, UA, AA, UAA, B, VB, BB, VBB;
  vec2 alpha = 2. * mid * halfSize / D, beta = halfSize * halfSize / D;
  axisMoments(u, alpha.x, beta.x, A, UA, AA, UAA);
  axisMoments(v, alpha.y, beta.y, B, VB, BB, VBB);
  vec4 pu = 1. - 2. * A + 3. * AA, pv = 1. - 2. * B + 3. * BB;
  return scale * (light.z * (pu + pv - 1. + 6. * A * B) +
                  light.x * (u * pv / 3. - 2. * UA + 3. * UAA + 6. * UA * B) +
                  light.y * (v * pu / 3. - 2. * VB + 3. * VBB + 6. * A * VB));
}
// Positive Gauss rules for the single transverse coordinate.
const float nodes4[4] = float[4](-.8611363116, -.3399810436, .3399810436, .8611363116);
const float weights4[4] = float[4](.3478548451, .6521451549, .6521451549, .3478548451);
const float nodes8[8] = float[8](-.9602898565, -.7966664774, -.5255324099, -.1834346425, .1834346425,
                                 .5255324099, .7966664774, .9602898565);
const float weights8[8] = float[8](.1012285363, .2223810345, .3137066459, .3626837834, .3626837834,
                                   .3137066459, .2223810345, .1012285363);
// A linear hat along a straight generator is proportional to the sine of
// the angular distance from the opposite endpoint. Integrating the product
// with the receiver cosine requires only two positive sine-product moments.
vec2 lengthWeights(vec2 mid, float w, float h, float slope, float middleCosine) {
  float g = length(vec2(h, mid.y));
  vec2 X = mid.x + vec2(-w, w), lens = sqrt(g * g + X * X);
  float den = lens.x * lens.y, sine = 2. * w * g / den, cosine = (g * g + X.x * X.y) / den;
  float crossTerm, squareTerm;
  if (cosine > 0. && sine < .16) {
    float z = sine * sine,
          ratio = 1. + z * (1. / 6. + z * (3. / 40. + z * (5. / 112. + z * 35. / 1152.)));
    float a2 = z * ratio * ratio, scale = sine * ratio * ratio * ratio;
    crossTerm = scale * (1. / 6. + a2 * (-1. / 60. + a2 * (1. / 1680. - a2 / 90720.)));
    squareTerm = scale * (1. / 3. + a2 * (-1. / 15. + a2 * (2. / 315. - a2 / 2835.)));
  } else {
    float a = atan(sine, cosine), scale = .5 / (sine * sine);
    crossTerm = (sine - a * cosine) * scale;
    squareTerm = (a - sine * cosine) * scale;
  }
  vec2 endpoints = (middleCosine + slope * vec2(-w, w)) / lens;
  return h / (g * lens) * (squareTerm * endpoints + crossTerm * endpoints.yx);
}
vec4 lineRule(vec2 mid, vec2 extent, float h, vec3 light, int rule) {
  bool swap = rule >= 5;
  if (swap) {
    mid = mid.yx;
    extent = extent.yx;
    light.xy = light.yx;
    rule -= 3;
  }
  int n = rule == 2 ? 2 : rule == 3 ? 4 : 8;
  float slope = light.x / extent.x;
  vec4 sum = vec4(0.);
  for (int j = 0; j < n; j++) {
    float t = n == 2 ? (j == 0 ? -.5773502692 : .5773502692) : n == 4 ? nodes4[j] : nodes8[j];
    float weight = n == 2 ? 1. : n == 4 ? weights4[j] : weights8[j];
    vec2 k =
        lengthWeights(vec2(mid.x, mid.y + t * extent.y), extent.x, h, slope, light.z + t * light.y);
    sum += weight * vec4((1. - t) * k, (1. + t) * k);
  }
  return (swap ? sum.xzyw : sum) * (extent.y / (2. * 3.14159265359));
}
vec4 bilinear(vec3 center, vec3 u, vec3 v, vec3 normal, float w, float t, int rule) {
  vec3 n = cross(u, v);
  float z = dot(center, n), h = abs(z), h2 = h * h;
  vec3 tilt = vec3(dot(normal, u), dot(normal, v), dot(normal, n) * z);
  vec2 mid = vec2(dot(center, u), dot(center, v)), extent = vec2(w, t);
  vec3 light = vec3(tilt.xy * extent, dot(center, normal));
  float D = dot(mid, mid) + h2;
  if (rule == 1)
    return quadraticField(mid, extent, D, light, w * t * h / (3.14159265359 * D * D));
  if (rule >= 2)
    return lineRule(mid, extent, h, light, rule);
  vec2 X = dot(center, u) + vec2(-w, w), Y = dot(center, v) + vec2(-t, t);
  vec2 A = edgeIntegrals(X, Y, h), B = edgeIntegrals(Y, X, h);
  float ax = X.y * A.y - X.x * A.x, by = Y.y * B.y - Y.x * B.x;
  float m00 = (ax + by) / (2. * h2), m10 = .5 * (A.x - A.y), m01 = .5 * (B.x - B.y);
  vec2 X2 = X * X, Y2 = Y * Y;
  float Dlo = h2 + X2.x + Y2.x, Dhi = h2 + X2.y + Y2.y,
        Dcross = (h2 + X2.x + Y2.y) * (h2 + X2.y + Y2.x);
  float d = (X.y - X.x) * (X.y + X.x) * (Y.x - Y.y) * (Y.x + Y.y) / Dcross;
  float m11 = -.25 * (abs(d) < .01 ? d * (1. - d * (.5 - d / 3.)) : log(Dlo * Dhi / Dcross));
  float total = tilt.z * m00 + tilt.x * m10 + tilt.y * m01;
  float mx = tilt.z * m10, my = tilt.z * m01, mxy = tilt.z * m11;
  if (tilt.x != 0. || tilt.y != 0.) {
    float J = potential(X.x, Y.x, h) - potential(X.x, Y.y, h) - potential(X.y, Y.x, h) +
              potential(X.y, Y.y, h);
    mx += .5 * tilt.x * (J - ax) + tilt.y * m11;
    my += .5 * tilt.y * (J - by) + tilt.x * m11;
    mxy += .5 * (tilt.x * ((h2 + Y2.y) * B.y - (h2 + Y2.x) * B.x) +
                 tilt.y * ((h2 + X2.y) * A.y - (h2 + X2.x) * A.x));
  }
  float s = (mx - X.x * total) / (2. * w), r = (my - Y.x * total) / (2. * t);
  float sr = (mxy - X.x * my - Y.x * mx + X.x * Y.x * total) / (4. * w * t);
  return vec4(total - s - r + sr, s - sr, r - sr, sr) * (h / 3.14159265359);
}`;
