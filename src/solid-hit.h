// Geometry is evaluated in double precision on the CPU. The output stream
// describes horizontal pixel runs, each with one known surface and lamp state.
typedef struct {
  double x, y, z;
} Vec;
typedef struct {
  Vec c, ax[3], h;
  double r;
  Vec extent;
} Shape;
typedef struct {
  Vec n;
  double b;
} Plane;
typedef struct {
  int first, n;
} Range;
typedef struct {
  int x, end, y, id, lit;
} Run;
static Vec plus(Vec a, Vec b) { return (Vec){a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec minus(Vec a, Vec b) { return (Vec){a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec scale3(Vec a, double x) { return (Vec){a.x * x, a.y * x, a.z * x}; }
static double dp(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static double low(double x, double y) { return x < y ? x : y; }
static double high(double x, double y) { return x > y ? x : y; }
static Vec local(const Shape *s, Vec v) {
  return (Vec){dp(v, s->ax[0]), dp(v, s->ax[1]), dp(v, s->ax[2])};
}
static double absd(double x) { return __builtin_fabs(x); }
// Error-free residual of a binary64 product of normal room-scale operands.
static double productTail(double a, double b, double product) {
  const double splitter = 134217729.;
  double x = splitter * a, y = splitter * b;
  double ah = x - (x - a), al = a - ah, bh = y - (y - b), bl = b - bh;
  return ((ah * bh - product + ah * bl) + al * bh) + al * bl;
}
static double quadraticDisc(double a, double b, double c) {
  double x = b * b, y = a * c, d = x - y;
  if (absd(d) > 4e-16 * (absd(x) + absd(y)))
    return d;
  return d + (productTail(b, b, x) - productTail(a, c, y));
}
#define FAR 1.e25
__attribute__((import_module("geometry"), import_name("hit"))) double
exactBoxHit(int, double, double, double, double, double, double, double, int);
static const Shape *predicateBase;
static int predicateCount;
static const Vec *predicateEndpoint;
static int shapeIndex(const Shape *s) {
  unsigned long p = (unsigned long)s, b = (unsigned long)predicateBase;
  return p >= b && p < b + sizeof(Shape) * predicateCount && (p - b) % sizeof(Shape) == 0
             ? (p - b) / sizeof(Shape)
             : -1;
}
static double hit(const Shape *s, Vec o, Vec d, double limit) {
  Vec rel = minus(o, s->c);
  if (!s->r) {
    double den = dp(d, s->ax[2]);
    if (!den)
      return FAR;
    double t = -dp(rel, s->ax[2]) / den;
    if (t <= 1.e-8 || t >= limit)
      return FAR;
    Vec p = plus(rel, scale3(d, t));
    if (absd(dp(p, s->ax[0])) > 1 || absd(dp(p, s->ax[1])) > 1)
      return FAR;
    return t;
  }
  Vec p = local(s, rel), v = local(s, d);
  double O[] = {p.x, p.y, p.z}, D[] = {v.x, v.y, v.z}, H[] = {s->h.x, s->h.y, s->h.z}, from = -FAR,
         to = FAR;
  for (int k = 0; k < 3; k++) {
    if (!D[k]) {
      if (absd(O[k]) > H[k])
        return FAR;
      continue;
    }
    double a = (-H[k] - O[k]) / D[k], b = (H[k] - O[k]) / D[k];
    from = high(from, low(a, b));
    to = low(to, high(a, b));
  }
  if (to <= 1.e-8 || from >= limit || from > to)
    return FAR;
  double end = low(to, limit), r = s->r, events[8];
  int n = 0;
  double dist = 0;
  for (int k = 0; k < 3; k++) {
    H[k] -= r;
    double q = high(0, absd(O[k] + from * D[k]) - H[k]);
    dist += q * q;
  }
  if (from > 1.e-8 && dist <= r * r * (1 + 1e-12))
    return from;
  from = high(from, 1.e-8);
  for (int k = 0; k < 3; k++)
    if (D[k])
      for (int side = -1; side <= 1; side += 2) {
        double t = (side * H[k] - O[k]) / D[k];
        if (t > from && t < end) {
          int j = n++;
          while (j && events[j - 1] > t) {
            events[j] = events[j - 1];
            j--;
          }
          events[j] = t;
        }
      }
  events[n++] = end;
  for (int i = 0; i < n; i++) {
    double until = events[i], m = .5 * (from + until), A = 0, B = 0, C = -r * r;
    for (int k = 0; k < 3; k++) {
      double projection = O[k] + m * D[k];
      if (absd(projection) <= H[k])
        continue;
      double q = O[k] + from * D[k] - (projection > 0 ? H[k] : -H[k]);
      A += D[k] * D[k];
      B += q * D[k];
      C += q * q;
    }
    double disc = quadraticDisc(A, B, C);
    // A curved silhouette can be much closer than one ulp of the two
    // squared terms. Include local-frame cancellation in the filter.
    if (A > 0 && absd(disc) <= 3e-14 * A * (r * r + dp(p, p) + dp(v, v) * from * from)) {
      int id = shapeIndex(s);
      if (id >= 0) {
        Vec target = predicateEndpoint ? *predicateEndpoint : d;
        return exactBoxHit(id, o.x, o.y, o.z, target.x, target.y, target.z, limit,
                           predicateEndpoint != 0);
      }
    }
    if (A > 0 && disc >= 0) {
      double root = __builtin_sqrt(disc), q = -B - (B >= 0 ? root : -root), a = q / A,
             b = q ? C / q : a;
      double t = from + low(a, b);
      if (t < from - 1e-12 || t <= 1e-8)
        t = from + high(a, b);
      if (t >= from - 1e-12 && t > 1e-8 && t <= until + 1e-12 && t < limit)
        return t;
    }
    from = until;
  }
  return FAR;
}

static double segmentHit(const Shape *s, Vec eye, Vec point, double limit) {
  predicateEndpoint = &point;
  double t = hit(s, eye, minus(point, eye), limit);
  predicateEndpoint = 0;
  return t;
}
