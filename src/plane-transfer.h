// For a planar source x=c+A u+B v, contract its six spatial moments with the
// quadratic Taylor expansion of the geometric kernel. The coefficients remain
// functions of y over a receiving rectangle. Visibility and a remainder bound
// must hold over that entire rectangle before it can use this expression.
typedef struct {
  Vec c, a, b, n;
  int object, pad;
} Domain;
typedef struct {
  Vec c, a, b;
  const Shape *s;
} Obstacle;
typedef struct {
  double v[6];
} Moments;
__attribute__((import_module("math"), import_name("asinh"))) double asinhValue(double);

static double radiusOn(Vec a, Vec b, Vec n) { return absd(dp(a, n)) + absd(dp(b, n)); }
static double obstacleRadius(const Obstacle *o, Vec n) {
  const Shape *s = o->s;
  if (!s->r)
    return radiusOn(o->a, o->b, n);
  return (s->h.x - s->r) * absd(dp(n, s->ax[0])) + (s->h.y - s->r) * absd(dp(n, s->ax[1])) +
         (s->h.z - s->r) * absd(dp(n, s->ax[2])) + s->r * length(n);
}
// A separating projection excludes every open segment from the source to any
// y in the receiver domain. Failure to separate leaves the pair unresolved.
static int separate(Vec axis, const Chart *s, Vec A, Vec B, const Domain *g, const Obstacle *o) {
  double size = dp(axis, axis);
  if (size < 1e-22)
    return 0;
  double a = dp(minus(s->c, o->c), axis), b = dp(minus(g->c, o->c), axis);
  double ar = radiusOn(A, B, axis), br = radiusOn(g->a, g->b, axis), r = obstacleRadius(o, axis);
  return low(a - ar, b - br) > r + 1e-12 || high(a + ar, b + br) < -r - 1e-12;
}
static int separated(const Chart *s, Vec A, Vec B, const Domain *g, const Obstacle *o) {
  // Open-segment endpoints on the receiving or emitting plane do not occlude.
  if (dp(minus(o->c, s->c), s->n) + obstacleRadius(o, s->n) <= 1e-12)
    return 1;
  if (dp(minus(o->c, g->c), g->n) + obstacleRadius(o, g->n) <= 1e-12)
    return 1;
  const Vec axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int i = 0; i < 3; i++)
    if (separate(axes[i], s, A, B, g, o))
      return 1;
  Vec d = minus(g->c, s->c);
  for (int i = 0; i < 3; i++)
    if (separate(cross3(d, axes[i]), s, A, B, g, o))
      return 1;
  return 0;
}
// If a sheet separates the source plane domain from the receiver domain,
// their connecting rays cross it inside the convex hull of the 16 corner
// crossings. The same test proves a clear opening or a complete shadow.
static int sheetCrossings(const Chart *s, Vec A, Vec B, const Domain *g, const Shape *o) {
  Vec n = o->ax[2];
  double h = dp(minus(s->c, o->c), n), k = dp(minus(g->c, o->c), n);
  double a = radiusOn(A, B, n), b = radiusOn(g->a, g->b, n);
  if ((h > a && k > b) || (h < -a && k < -b))
    return 1;
  if (!((h > a + 1e-9 && k < -b - 1e-9) || (h < -a - 1e-9 && k > b + 1e-9)))
    return 0;
  Vec p[4], q[4];
  double ph[4], qh[4];
  for (int i = 0; i < 4; i++) {
    int u = i & 1 ? 1 : -1, v = i & 2 ? 1 : -1;
    p[i] = plus(minus(s->c, o->c), plus(scale3(A, u), scale3(B, v)));
    q[i] = plus(minus(g->c, o->c), plus(scale3(g->a, u), scale3(g->b, v)));
    ph[i] = dp(p[i], n);
    qh[i] = dp(q[i], n);
  }
  unsigned common = 15, any = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      Vec x = scale3(minus(scale3(q[j], ph[i]), scale3(p[i], qh[j])), 1 / (ph[i] - qh[j]));
      double u = dp(x, o->ax[0]), v = dp(x, o->ax[1]);
      unsigned outside = (u > 1 + 1e-10) | ((u < -1 - 1e-10) << 1) | ((v > 1 + 1e-10) << 2) |
                         ((v < -1 - 1e-10) << 3);
      common &= outside;
      any |= (absd(u) > 1 - 1e-10 || absd(v) > 1 - 1e-10);
    }
  return common ? 1 : any ? 0 : 2;
}
static int qualify(const Chart *s, const Domain *g, const Obstacle *obs, int ns, Vec lamp,
                   double tol) {
  if (s->object == g->object)
    return 2;
  if (s->type)
    return (arcFacesAway(s, minus(g->c, s->c), g->a, g->b, 1) ||
            belowHorizon(s, minus(g->c, s->c), g->n))
               ? CURVED_DOMAIN_DARK
               : 0;
  Vec D = minus(g->c, s->c), A = scale3(s->ax, s->width), B = scale3(s->v, s->height);
  double h = -dp(g->n, D), dh = radiusOn(A, B, g->n), k = dp(s->n, D),
         dk = radiusOn(g->a, g->b, s->n);
  if (h + dh <= 0 || k + dk <= 0)
    return 2;
  if (h < dh || k < dk)
    return 3;
  // Smallest distance from the source center to the receiver parallelogram
  // is at least its distance to each supporting slab.
  double d = length(D), r = high(length(plus(g->a, g->b)), length(minus(g->a, g->b)));
  double dmin = high(d - r, absd(dp(g->n, D)));
  if (dmin <= 1e-10)
    return 4;
  const double dd = dmin * dmin;
  double z1 =
      2 * (absd(dp(D, A)) + absd(dp(D, B)) + radiusOn(g->a, g->b, A) + radiusOn(g->a, g->b, B)) /
      dd;
  double z2 = (dp(A, A) + dp(B, B) + 2 * absd(dp(A, B))) / dd, z = z1 + z2;
  if (z >= 1)
    return 4;
  double tail = z * z * z * (4 - 3 * z) / ((1 - z) * (1 - z)) + 6 * z1 * z2 + 3 * z2 * z2;
  double error = (k + dk) * ((h + dh) * tail + dh * (2 * z2 + 3 * z1 * z1)) / (PI * dd * dd);
  double lampDist =
      high(0, length(minus(lamp, s->c)) - high(length(plus(A, B)), length(minus(A, B))));
  double maxLamp = dp(s->n, minus(lamp, s->c)) > 0 ? 1 / high(1e-20, lampDist * lampDist) : 0;
  if (error * 4 * length(cross3(A, B)) * high(1, maxLamp) > tol * .25)
    return 4;
  for (int j = 0; j < ns; j++)
    if (j != s->object && j != g->object && !separated(s, A, B, g, obs + j)) {
      int test = obs[j].s->r ? 0 : sheetCrossings(s, A, B, g, obs[j].s);
      if (test == 2)
        return 2;
      if (!test)
        return 5;
    }
  return 1;
}
// Moments of direct lamp irradiance include the union of every caster shadow.
// Integrals of 1,x,x² over (x²+d²)^(3/2) have elementary primitives.
static Moments lampSlice(Pair *w, double t) {
  w->stats->momentSlices++;
  Moments M = {0};
  const Chart *s = w->c;
  Vec n, p = cp(s, 0, t, &n), l = minus(w->lamp, p);
  double cosine = dp(l, n), d2 = dp(cross3(l, s->ax), cross3(l, s->ax)), q = dp(l, s->ax);
  if (cosine <= 0 || d2 <= 0)
    return M;
  int cap = w->count[1] + 1;
  Interval dark[cap], open[cap + 1];
  int nDark = shadows(w, 1, p, s->ax, -s->width, s->width, dark);
  int count = openings(dark, nDark, -s->width, s->width, open);
  for (int i = 0; i < count; i++) {
    double a = open[i].lo - q, b = open[i].hi - q, A = ROOT(a * a + d2), B = ROOT(b * b + d2);
    double j0 = (b / B - a / A) / d2, j1 = (b - a) * (b + a) / (A * B * (A + B));
    if (a * b > 0)
      j0 = (b - a) * (b + a) / (A * B * (b * A + a * B));
    double j2 = asinhValue(b / ROOT(d2)) - asinhValue(a / ROOT(d2)) - d2 * j0;
    double u = (j1 + q * j0) / s->width, v = t / s->height;
    double m[] = {j0,    u,         v * j0, (j2 + 2 * q * j1 + q * q * j0) / (s->width * s->width),
                  u * v, v * v * j0};
    for (int k = 0; k < 6; k++)
      M.v[k] += cosine * m[k];
  }
  return M;
}
static Moments lampPanel(Pair *w, double a, double b, double tol, int depth) {
  double m = .5 * (a + b), h = .5 * (b - a), z = h * .7745966692414834;
  Moments c = lampSlice(w, m), l = lampSlice(w, m - z), r = lampSlice(w, m + z), M;
  double err = 0;
  for (int k = 0; k < 6; k++) {
    M.v[k] = h * (8 * c.v[k] + 5 * (l.v[k] + r.v[k])) / 9;
    err = high(err, absd(M.v[k] - 2 * h * c.v[k]));
  }
  if (err <= tol)
    return M;
  if (!depth) {
    w->stats->limited++;
    return M;
  }
  l = lampPanel(w, a, m, tol * .5, depth - 1);
  r = lampPanel(w, m, b, tol * .5, depth - 1);
  for (int k = 0; k < 6; k++)
    M.v[k] = l.v[k] + r.v[k];
  return M;
}
// A sheet edge has one limiting plane for a whole receiver parallelogram.
// The signed slope (1 - sign*u_r)/depth_r is linear-fractional with strictly
// positive denominator, so its minimum is attained at a receiver corner.
typedef struct {
  Vec n[5], center;
  double margin[5];
  int valid;
} Umbra;
static Umbra sharedShadow(const Domain *g, const Shape *b) {
  Umbra result = {0};
  if (b->r)
    return result;
  Vec d = minus(b->c, g->c), z = b->ax[2];
  if (dp(d, z) < 0)
    z = scale3(z, -1);
  if (dp(d, z) - radiusOn(g->a, g->b, z) <= 1e-8)
    return result;
  result.center = b->c;
  result.n[0] = z;
  result.valid = 1;
  for (int axis = 0; axis < 2; axis++)
    for (int k = 0; k < 2; k++) {
      Vec u = scale3(b->ax[axis], k ? 1 : -1);
      double slope = FAR;
      for (int i = 0; i < 4; i++) {
        Vec delta = plus(d, plus(scale3(g->a, i & 1 ? 1 : -1), scale3(g->b, i & 2 ? 1 : -1)));
        slope = low(slope, (1 + dp(u, delta)) / dp(z, delta));
      }
      result.n[1 + 2 * axis + k] = minus(scale3(z, slope), u);
    }
  for (int k = 0; k < 5; k++)
    result.margin[k] = 1 + length(result.n[k]);
  return result;
}
static int shadowClass(const Umbra *shadow, const Obstacle *o) {
  Vec d = minus(o->c, shadow->center);
  int inside = 1;
  for (int i = 0; i < 5; i++) {
    Vec n = shadow->n[i];
    double mid = (i != 0) + dp(d, n), ext = obstacleRadius(o, n);
    double eps = 1e-8 * (shadow->margin[i] + absd(mid) + ext);
    if (mid + ext < -eps)
      return -1;
    inside &= mid - ext > eps;
  }
  return inside;
}
static int coversChart(const Umbra *shadow, const Chart *c) {
  Vec d = minus(c->c, shadow->center);
  for (int i = 0; i < 5; i++) {
    Vec n = shadow->n[i];
    double mid = (i != 0) + dp(d, n);
    double ext = c->type == 2 ? c->r * length(n)
                              : c->width * absd(dp(c->ax, n)) +
                                    (c->type ? c->r * length(n) : c->height * absd(dp(c->v, n)));
    if (mid - ext <= 1e-8 * (shadow->margin[i] + absd(mid) + ext))
      return 0;
  }
  return 1;
}
void buildRegions(const Shape *s, int ns, const Chart *tiles, int nt, const Domain *groups, int ng,
                  const Vec *lamp, double tol, unsigned char *modes, Moments *moments,
                  Counts *stats, int *lampIds) {
  Cone lamps[ns];
  Obstacle obs[ns];
  for (int j = 0; j < ns; j++) {
    lamps[j] = shadow(s + j, j, *lamp);
    obs[j] = (Obstacle){.c = s[j].c, .s = s + j};
    if (!s[j].r) {
      Vec u = cross3(s[j].ax[1], s[j].ax[2]), v = cross3(s[j].ax[2], s[j].ax[0]);
      obs[j].a = scale3(u, 1 / dp(s[j].ax[0], u));
      obs[j].b = scale3(v, 1 / dp(s[j].ax[1], v));
    }
  }
  unsigned char needed[nt];
  for (int k = 0; k < nt; k++)
    needed[k] = 0;
  for (int g = 0; g < ng; g++) {
    const Domain *domain = groups + g;
    Umbra shadows[ns];
    int nearby[ns], last = -1, n = 0;
    for (int j = 0; j < ns; j++)
      shadows[j] = sharedShadow(domain, s + j);
    for (int k = 0; k < nt; k++) {
      const Chart *c = tiles + k;
      int obj = c->object, mode = 2;
      if (obj != last) {
        last = obj;
        n = 0;
        if (obj == domain->object)
          n = -1;
        else
          for (int j = 0; j < ns; j++)
            if (j != obj && j != domain->object && shadows[j].valid) {
              int type = shadowClass(shadows + j, obs + obj);
              if (type > 0) {
                n = -1;
                break;
              }
              if (!type)
                nearby[n++] = j;
            }
      }
      if (n >= 0) {
        mode = qualify(c, domain, obs, ns, *lamp, tol);
        if (mode != 1 && mode != 2 && mode != CURVED_DOMAIN_DARK)
          for (int i = 0; i < n; i++) {
            if (coversChart(shadows + nearby[i], c)) {
              mode = 2;
              break;
            }
          }
      }
      modes[g * nt + k] = mode;
      needed[k] |= mode == 1;
    }
  }
  for (int k = 0; k < nt; k++) {
    const Chart *c = tiles + k;
    Moments M = {0};
    {
      int hidden = 0;
      int *ids = lampIds + k * (ns + 1);
      Pair w = {.s = s,
                .c = c,
                .lamp = *lamp,
                .cone = {0, lamps},
                .blockers = {0, ids + 1},
                .stats = stats};
      w.count[1] = candidates(c, s, -1, lamps, ns, ids + 1, &hidden);
      if (c->type && arcFacesAway(c, minus(*lamp, c->c), (Vec){0}, (Vec){0}, 0))
        hidden = 1;
      if (!c->type && dp(c->n, minus(*lamp, c->c)) <= 0)
        hidden = 1;
      ids[0] = hidden ? -1 : w.count[1];
      if (needed[k] && !hidden) {
        stats->momentTiles++;
        int cap = 32 + 32 * w.count[1];
        double cuts[cap + 1];
        int n = planarEvents(&w, cuts, cap, -c->height, c->height);
        if (n == cap)
          stats->limited++;
        cuts[n++] = c->height;
        double a = -c->height;
        for (int j = 0; j < n; j++) {
          double b = cuts[j];
          Moments v = lampPanel(&w, a, b, 2e-6 * (b - a) / (2 * c->height), 20);
          for (int i = 0; i < 6; i++)
            M.v[i] += v.v[i];
          a = b;
        }
      }
    }
    moments[k] = M;
  }
}
static void evaluatePlane(const Chart *s, Receiver y, const Moments *m, float *w) {
  Vec D = minus(y.p, s->c), A = scale3(s->ax, s->width), B = scale3(s->v, s->height);
  double q = 1 / dp(D, D), h = -dp(y.n, D), a = dp(D, A) * q, b = dp(D, B) * q, ha = dp(y.n, A),
         hb = dp(y.n, B);
  double f = dp(s->n, D) * q * q / PI;
  double P[] = {h,
                ha + 4 * h * a,
                hb + 4 * h * b,
                h * (12 * a * a - 2 * q * dp(A, A)) + 4 * ha * a,
                h * (24 * a * b - 4 * q * dp(A, B)) + 4 * (ha * b + hb * a),
                h * (12 * b * b - 2 * q * dp(B, B)) + 4 * hb * b};
  for (int j = 0; j < 4; j++) {
    int u = j & 1 ? 1 : -1, v = j & 2 ? 1 : -1;
    w[j] = f * s->width * s->height *
           (P[0] + (u * P[1] + v * P[2] + P[3] + P[5]) / 3 + u * v * P[4] / 9);
  }
  double lamp = 0;
  for (int k = 0; k < 6; k++)
    lamp += P[k] * m->v[k];
  w[4] = f * high(0, lamp);
}
