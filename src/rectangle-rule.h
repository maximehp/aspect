// All four hats share one distance interval and a positive affine numerator.
// Select a complete area expression (1), a bounded transverse rule (2–7),
// exact endpoint primitives (0), or the original adaptive generator (-1).
static int rectangleRule(const Chart *c, Vec d, Vec normal) {
  double x = absd(dp(d, c->ax)), y = absd(dp(d, c->v)), h = absd(dp(d, c->n)), w = c->width,
         t = c->height;
  if (h < 1e-5)
    return -1;
  double a = high(0, x - w), b = high(0, y - t), lower = h * h + a * a + b * b,
         upper = h * h + (x + w) * (x + w) + (y + t) * (y + t);
  double D = dp(d, d), maxN = dp(d, normal) + w * absd(dp(c->ax, normal)) + t * absd(dp(c->v, normal));
  double gain = w * t * h * maxN / PI, scale = gain / (D * D), tol = 1e-10 + 2e-5 * scale;
  double z = (upper - D) / D;
  if (z < .8 && scale * z * z * z * (4 - 3 * z) <= tol * (1 - z) * (1 - z))
    return 1;
  // For orthogonal rectangles with appreciable angular extent, the elementary
  // endpoint moments retain useful significance. Grazing and thin supports
  // can lose most digits in their alternating sum.
  if (D < 100 * low(w * w, t * t) && h * h > .0625 * (a * a + b * b))
    return 0;
  // Only the transverse coordinate needs numerical integration. At every
  // point of the long axis, D is bounded below by lower. Its relative spread
  // is largest at the nearest point along that axis.
  double vx = (x + w) * (x + w) - a * a, vy = (y + t) * (y + t) - b * b;
  int flip = vx < vy;
  upper = lower + (flip ? vx : vy);
  double s = ROOT(lower * upper), q = (upper - lower) / (upper + lower + 2 * s), one = 1 - q;
  double bound = 4 * gain / (s * s * one), shift = .5 * (upper + lower) / s + q / one;
  int base = flip ? 5 : 2;
  if (bound * q * (shift + 1) <= tol)
    return base;
  double q3 = q * q * q;
  if (bound * q3 * (shift + 3) <= tol)
    return base + 1;
  if (bound * q3 * q3 * q * (shift + 7) <= tol)
    return base + 2;
  return -1;
}
