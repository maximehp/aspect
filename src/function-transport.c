// Receiver collocation of the same continuous-source integral used on every bounce.
#include "solid-hit.h"
#define PI 3.14159265358979323846
#define ROOT(x) __builtin_sqrt(x)
__attribute__((import_module("math"), import_name("sin"))) double sine(double);
__attribute__((import_module("math"), import_name("cos"))) double cosine(double);
__attribute__((import_module("math"), import_name("atan2"))) double angle(double, double);
static Vec cross3(Vec a, Vec b) {
  return (Vec){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static Vec normalized(Vec x) { return scale3(x, 1 / ROOT(dp(x, x))); }
static double length(Vec x) { return ROOT(dp(x, x)); }
// A chart is a rectangle, quarter cylinder, or spherical octant. ax is the
// straight generator or the polar axis. u,v are transverse directions.
// The solver and shader interpolate identical positive nodal functions.
typedef struct {
  Vec c, ax, u, v, n;
  double width, height, r;
  int type, object, face, base, nu, nv;
} Chart;
typedef struct {
  Vec p, n;
  int object, pad;
} Receiver;
typedef struct {
  Plane p[12];
  int n, obj;
  unsigned tangentMask;
} Cone;
typedef struct {
  double lo, hi;
  int id;
} Interval;
typedef struct {
  long long strips, endpoints, exact, numeric, boundary, splits, limited, gpu, axialHits, regional,
      dark, facing, expansion, obstruction, individual, momentTiles, momentSlices, refined,
      sliceHits, sliceMisses;
} Counts;
typedef struct {
  double w[5];
} Integral; // four nodal functions and direct lamp
static Integral sum(Integral a, Integral b) {
  for (int j = 0; j < 5; j++)
    a.w[j] += b.w[j];
  return a;
}
static Integral times(Integral a, double k) {
  for (int j = 0; j < 5; j++)
    a.w[j] *= k;
  return a;
}
static Vec cp(const Chart *p, double x, double t, Vec *normal) {
  if (p->type == 0) {
    *normal = p->n;
    return plus(p->c, plus(scale3(p->ax, x), scale3(p->v, t)));
  }
  Vec circle = plus(scale3(p->u, cosine(t)), scale3(p->v, sine(t)));
  if (p->type == 1) {
    *normal = circle;
    return plus(p->c, plus(scale3(p->ax, x), scale3(circle, p->r)));
  }
  *normal = plus(scale3(circle, cosine(x)), scale3(p->ax, sine(x)));
  return plus(p->c, scale3(*normal, p->r));
}
static void addPlane(Cone *c, Vec n, Vec p, Vec inside) {
  double size = length(n);
  if (size < 1e-18)
    return;
  if (dp(n, inside) < 0)
    size = -size;
  n = scale3(n, 1 / size);
  c->p[c->n++] = (Plane){n, dp(n, p)};
}
static Cone shadow(const Shape *s, int id, Vec eye) {
  Cone out = {.obj = id};
  Vec d = minus(s->c, eye);
  if (!s->r) {
    Vec normal = s->ax[2];
    if (absd(dp(d, normal)) < 1e-9)
      return out;
    addPlane(&out, normal, s->c, d);
    for (int k = 0; k < 2; k++)
      for (int sign = -1; sign <= 1; sign += 2) {
        // Homogeneous clipping of a finite sheet in its authored dual coordinates.
        double h = dp(normal, d);
        Vec side = minus(scale3(normal, 1 + sign * dp(d, s->ax[k])), scale3(s->ax[k], sign * h));
        addPlane(&out, side, eye, d);
      }
    return out;
  }
  double h[3] = {s->h.x, s->h.y, s->h.z};
  // A tangent contact has n dot (eye - center) = radius. Bound that
  // quantity on each feature's orthant of unit normals for this entire eye.
  Vec O = local(s, minus(eye, s->c));
  unsigned bit = 1;
  for (int feature = -1; feature < 3; feature++)
    for (int signs = 0; signs < 8; signs++) {
      if (feature >= 0 && (signs & (1 << feature)))
        continue;
      double positive = 0, min = FAR;
      for (int d = 0; d < 3; d++)
        if (d != feature) {
          double q = (signs & (1 << d) ? 1 : -1) * ((double *)&O)[d] - (h[d] - s->r);
          positive += high(0, q) * high(0, q);
          min = low(min, q);
        }
      double eps = 1e-9 * (1 + length(O));
      if (positive >= (high(0, s->r - eps) * high(0, s->r - eps)) && min <= s->r + eps)
        out.tangentMask |= bit;
      bit <<= 1;
    }
  int front[3];
  for (int k = 0; k < 3; k++) {
    double a = -dp(d, s->ax[k]);
    front[k] = a > h[k] ? 1 : a < -h[k] ? -1 : 0;
    if (front[k]) {
      Vec n = scale3(s->ax[k], -front[k]);
      addPlane(&out, n, plus(s->c, scale3(s->ax[k], front[k] * h[k])), n);
    }
  }
  for (int k = 0; k < 3; k++) {
    int i = (k + 1) % 3, j = (k + 2) % 3;
    for (int a = -1; a <= 1; a += 2)
      for (int b = -1; b <= 1; b += 2)
        if ((a == front[i]) != (b == front[j])) {
          Vec edge = plus(d, plus(scale3(s->ax[i], a * h[i]), scale3(s->ax[j], b * h[j])));
          addPlane(&out, cross3(edge, s->ax[k]), eye, d);
        }
  }
  return out;
}
static int clipPositive(double a, double b, double *lo, double *hi) {
  if (b > 0)
    *lo = high(*lo, -a / b);
  else if (b < 0)
    *hi = low(*hi, -a / b);
  else if (a < 0)
    return 0;
  return *lo < *hi;
}
// Within one pose the lamp-lit subset of a source generator is independent
// of the receiver. Cache complete interval unions with the exact binary
// transverse parameter as key. A collision only causes recomputation.
typedef struct {
  double t, ends[6];
  int source, count;
} LampSlice;
#define SLICE_SLOTS 16384
typedef struct {
  int source, eye, object, feature, count;
  double roots[2];
} Tangents;
#define TANGENT_SLOTS 2048
typedef struct {
  const Shape *s;
  const Chart *c;
  Receiver y;
  Vec lamp;
  const Cone *cone[2];
  const int *blockers[2];
  int count[2];
  Counts *stats;
  double tolerance, warpCenter, warpDepth;
  int lampDark, source;
  LampSlice *cache;
  Tangents *tangents;
  int receiver;
  double lift[5];
} Pair;
// Each catalog entry refers to an immutable cone and carries just the
// source-specific endpoint flag and axial dimension (zero means none).
static const Cone *blockerCone(const Pair *w, int eye, int j) {
  return w->cone[eye] + (w->blockers[eye][j] >> 3);
}
static int blocked(const Shape *s, Vec eye, Vec p) {
  return segmentHit(s, eye, p, 1. - 1e-10) < 1.;
}
static int inSolid(const Shape *s, Vec p) {
  Vec q = local(s, minus(p, s->c));
  double a = high(0, absd(q.x) - s->h.x + s->r), b = high(0, absd(q.y) - s->h.y + s->r),
         c = high(0, absd(q.z) - s->h.z + s->r);
  return a * a + b * b + c * c < s->r * s->r;
}
// A convex solid casts one interval on a line. Topology can change only at a
// tangent ray or when the endpoint crosses the solid. On a rounded box the
// smooth features are spheres, circular cylinders and planes. Collect all
// candidate events from their supporting quadrics and test the open cells.
static int rootsWithDisc(double A, double B, double C, double disc, double a, double b,
                         double *events, int n) {
  if (disc < 0)
    return n;
  if (A == 0) {
    if (B) {
      double x = -C / (2 * B);
      if (x > a && x < b)
        events[n++] = x;
    }
    return n;
  }
  double q = -B - __builtin_copysign(ROOT(disc), B);
  double x = q / A, y = q ? C / q : x;
  if (x > a && x < b)
    events[n++] = x;
  if (y > a && y < b)
    events[n++] = y;
  return n;
}
static int roots(double A, double B, double C, double a, double b, double *events, int n) {
  return rootsWithDisc(A, B, C, quadraticDisc(A, B, C), a, b, events, n);
}
__attribute__((import_module("geometry"), import_name("refine"))) void
refineConic(int, int, int, double, double, double, double, double, double, int, double *);
static int featureRoots(double A, double B, double C, double sizeA, double sizeB, double sizeC,
                        double a, double b, double *events, int count, int object, int axis,
                        int signs, Vec p, Vec u, int endpoint, Counts *stats) {
  double disc = quadraticDisc(A, B, C);
  // Reconstruct near-cancelling conics from the authored radius and the exact
  // binary values of this pose. The scales precede coefficient subtraction.
  if (object >= 0 && absd(disc) <= 3e-14 * (sizeB * sizeB + sizeA * sizeC)) {
    double f[4];
    refineConic(object, axis, signs, p.x, p.y, p.z, u.x, u.y, u.z, endpoint, f);
    stats->refined++;
    return rootsWithDisc(f[0], f[1], f[2], f[3], a, b, events, count);
  }
  return rootsWithDisc(A, B, C, disc, a, b, events, count);
}
static int trimRounded(const Shape *s, Vec eye, Vec p, Vec ax, double *lo, double *hi,
                       Counts *stats, int object, const Cone *cone, int descriptor, Pair *pair,
                       int eyeId) {
  const double a = *lo, b = *hi;
  const int endpoint = descriptor & 4, axial = (descriptor & 3) - 1;
  Vec O = local(s, minus(eye, s->c)), D = local(s, minus(p, eye)), U = local(s, ax),
      fan = cross3(D, U);
  double h[] = {s->h.x - s->r, s->h.y - s->r, s->h.z - s->r}, r = s->r, size = length(fan);
  if (absd(dp(O, fan)) >
      absd(fan.x) * h[0] + absd(fan.y) * h[1] + absd(fan.z) * h[2] + r * size + 1e-10)
    return 0;
  int left = blocked(s, eye, plus(p, scale3(ax, a))),
      right = blocked(s, eye, plus(p, scale3(ax, b)));
  if (left && right)
    return 1;
  double start = left ? a : FAR, end = right ? b : -FAR;
  Vec P = plus(O, D);
  if (endpoint)
    for (int k = 0; k < 3; k++)
      if (((double *)&U)[k])
        for (int sign = -1; sign <= 1; sign += 2) {
          double x = (sign * (h[k] + r) - ((double *)&P)[k]) / ((double *)&U)[k];
          if (x <= a || x >= b)
            continue;
          Vec v = plus(P, scale3(U, x));
          int j = (k + 1) % 3, i = (k + 2) % 3;
          if (absd(((double *)&v)[j]) <= h[j] + 1e-10 && absd(((double *)&v)[i]) <= h[i] + 1e-10) {
            start = low(start, x);
            end = high(end, x);
          }
        }
  unsigned bit = 1;
  for (int feature = -1; feature < 3; feature++)
    for (int bits = 0; bits < 8; bits++) {
      if (feature >= 0 && bits & (1 << feature))
        continue;
      const int mayTangent = !!(cone->tangentMask & bit);
      bit <<= 1;
      if (!mayTangent && !endpoint)
        continue;
      Vec center = {0};
      for (int j = 0; j < 3; j++)
        if (j != feature)
          ((double *)&center)[j] = (bits & (1 << j)) ? h[j] : -h[j];
      Vec o = minus(O, center), v = D, u = U;
      double spread = feature < 0 ? 0 : absd(((double *)&fan)[feature]) * h[feature];
      if (absd(dp(o, fan)) > r * size + spread + 1e-10)
        continue;
      Tangents *entry = 0;
      int eyeKey = eyeId ? 0 : pair->receiver + 1, code = 8 * feature + bits;
      if (mayTangent && pair->tangents && feature >= 0 && axial == feature) {
        unsigned key = (pair->source * 31u + eyeKey) * 65599u + cone->obj * 31u + code;
        entry = pair->tangents + (key & (TANGENT_SLOTS - 1));
      }
      double events[4];
      int tangent = 0;
      int cached = entry && entry->source == pair->source + 1 && entry->eye == eyeKey &&
                   entry->object == cone->obj && entry->feature == code;
      if (cached) {
        for (int j = 0; j < entry->count; j++)
          if (entry->roots[j] > a && entry->roots[j] < b)
            events[tangent++] = entry->roots[j];
        stats->axialHits++;
        if (!tangent && !endpoint)
          continue;
      }
      if (feature >= 0)
        ((double *)&o)[feature] = ((double *)&v)[feature] = ((double *)&u)[feature] = 0;
      double uu = dp(u, u);
      if (uu == 0)
        continue;
      if (mayTangent && !cached) {
        double S = dp(o, o) - r * r, ou = dp(o, u), ov = dp(o, v), uv = dp(u, v), vv = dp(v, v);
        double lo = entry ? -pair->c->width : a, hi = entry ? pair->c->width : b;
        tangent = featureRoots(ou * ou - S * uu, ou * ov - S * uv, ov * ov - S * vv,
                               ou * ou + absd(S * uu), absd(ou * ov) + absd(S * uv),
                               ov * ov + absd(S * vv), lo, hi, events, 0, object, feature, bits, p,
                               ax, 0, stats);
        if (entry) {
          entry->source = pair->source + 1;
          entry->eye = eyeKey;
          entry->object = cone->obj;
          entry->feature = code;
          entry->count = tangent;
          int n = tangent;
          tangent = 0;
          for (int j = 0; j < n; j++) {
            double x = events[j];
            entry->roots[j] = x;
            if (x > a && x < b)
              events[tangent++] = x;
          }
        }
      }
      Vec ep = plus(o, v);
      double eu = dp(ep, u), ee = dp(ep, ep);
      int n = endpoint ? featureRoots(uu, eu, ee - r * r, uu, length(ep) * length(u), ee + r * r, a,
                                      b, events, tangent, object, feature, bits, p, ax, 1, stats)
                       : tangent;
      for (int i = 0; i < n; i++) {
        double x = events[i], t = 1;
        if (i < tangent) {
          Vec dir = plus(v, scale3(u, x));
          double q = dp(dir, dir);
          if (q <= 1e-24)
            continue;
          t = -dp(o, dir) / q;
          if (t <= 1e-10 || t > 1 + 1e-10)
            continue;
        }
        Vec contact = plus(O, scale3(plus(D, scale3(U, x)), t));
        int ok = 1;
        for (int j = 0; j < 3; j++) {
          double z = ((double *)&contact)[j];
          if (j == feature) {
            if (absd(z) > h[j] + 1e-8)
              ok = 0;
          } else if ((bits & (1 << j) ? z : -z) < h[j] - 1e-8)
            ok = 0;
        }
        if (ok) {
          start = low(start, x);
          end = high(end, x);
        }
      }
      stats->boundary++;
    }
  *lo = start;
  *hi = end;
  return end > start;
}
static int shadows(Pair *w, int eyeId, Vec p, Vec ax, double lo, double hi, Interval *out) {
  int used = 0;
  Vec eye = eyeId ? w->lamp : w->y.p;
  for (int j = 0; j < w->count[eyeId]; j++) {
    const Cone *c = blockerCone(w, eyeId, j);
    double a = lo, b = hi;
    for (int k = 0; k < c->n; k++) {
      Plane h = c->p[k];
      double v = dp(h.n, p) - h.b;
      if (!clipPositive(v - (k == 0 && !w->s[c->obj].r ? 1e-9 : -1e-11), dp(h.n, ax), &a, &b))
        goto clear;
    }
    if (w->s[c->obj].r && !trimRounded(w->s + c->obj, eye, p, ax, &a, &b, w->stats,
                                       eyeId ? c->obj : -1, c, w->blockers[eyeId][j], w, eyeId))
      goto clear;
    out[used++] = (Interval){a, b, c->obj};
  clear:;
  }
  return used;
}
static int openings(Interval *blocked, int n, double a, double b, Interval *open) {
  for (int i = 1; i < n; i++) {
    Interval x = blocked[i];
    int j = i;
    while (j && blocked[j - 1].lo > x.lo) {
      blocked[j] = blocked[j - 1];
      j--;
    }
    blocked[j] = x;
  }
  int count = 0, id = -1;
  for (int i = 0; i < n; i++) {
    if (blocked[i].lo > a)
      open[count++] = (Interval){a, blocked[i].lo, id};
    if (blocked[i].hi > a) {
      a = blocked[i].hi;
      id = blocked[i].id;
    }
  }
  if (a < b)
    open[count++] = (Interval){a, b, id};
  return count;
}
static int litGenerator(Pair *w, double t, Vec p, Interval *lit) {
  const Chart *c = w->c;
  if (!w->count[1]) {
    lit[0] = (Interval){-c->width, c->width, 0};
    return 1;
  }
  union {
    double d;
    unsigned long long u;
  } bits = {t};
  unsigned long long h = bits.u ^ ((unsigned long long)(w->source + 1) * 0x9e3779b97f4a7c15ULL);
  h ^= h >> 30;
  h *= 0xbf58476d1ce4e5b9ULL;
  h ^= h >> 32;
  LampSlice *entry = w->cache + (h & (SLICE_SLOTS - 1));
  if (entry->source == w->source + 1 && entry->t == t) {
    for (int i = 0; i < entry->count; i++)
      lit[i] = (Interval){entry->ends[2 * i], entry->ends[2 * i + 1], 0};
    w->stats->sliceHits++;
    return entry->count;
  }
  Interval blocked[w->count[1] + 1];
  int n = shadows(w, 1, p, c->ax, -c->width, c->width, blocked);
  n = openings(blocked, n, -c->width, c->width, lit);
  w->stats->sliceMisses++;
  if (n <= 3) {
    entry->source = w->source + 1;
    entry->t = t;
    entry->count = n;
    for (int i = 0; i < n; i++) {
      entry->ends[2 * i] = lit[i].lo;
      entry->ends[2 * i + 1] = lit[i].hi;
    }
  }
  return n;
}
// Ql and Qy are squared distances on the unit-speed straight generator.
// Expand the analytic primitive locally about the interval midpoint. D f'=N f
// for f=Ql^(-3/2) Qy^(-2). Cauchy's estimate on |z|=2 in scaled coordinates
// bounds the integrated tail. No lengthwise source-light evaluations occur.
static double directPrimitive(double l, double q, double L, double Y, double c, double slope,
                              double a, double b, double tol, Counts *stats) {
  double mid = .5 * (a + b), h = .5 * (b - a), x = mid - l, y = mid - q, A = x * x + L,
         B = y * y + Y;
  double dl = ROOT(A), dy = ROOT(B);
  if (3 * h >= low(dl, dy)) {
    return directPrimitive(l, q, L, Y, c, slope, a, mid, tol * .5, stats) +
           directPrimitive(l, q, L, Y, c, slope, mid, b, tol * .5, stats);
  }
  double C = c + slope * mid, k = slope * h, ml = dl - 2 * h, my = dy - 2 * h;
  double bound = 4 * h * (absd(C) + 2 * absd(k)) / (ml * ml * ml * my * my * my * my);
  double u = 2 * x * h / A, v = h * h / A, s = 2 * y * h / B, t = h * h / B;
  double den[5] = {1, u + s, v + t + u * s, u * t + v * s, v * t};
  double num[4] = {-1.5 * u - 2 * s, -3 * v - 4 * t - 3.5 * u * s, -5.5 * u * t - 5 * v * s,
                   -7 * v * t};
  // z^2 coefficient of -1.5 Ql'Qy -2 Qy'Ql is -5.5 u*t -5 v*s.
  double prev[4] = {0}, f = 1, total = C;
  int n = 0;
  while (bound > tol || n < 4) {
    double v0 = 0;
    for (int j = 0; j < 4 && j <= n; j++)
      v0 += (num[j] - den[j + 1] * (n - j)) * (j ? prev[(n - j) & 3] : f);
    prev[n & 3] = f;
    double next = v0 / (n + 1);
    n++;
    if (!(n & 1))
      total += (C * next + k * f) / (n + 1);
    f = next;
    bound *= .5;
  }
  stats->exact++;
  return 2 * h * total / (A * dl * B * B);
}
static void moments(double a, double b, double d, double *m0, double *m1, double *m2) {
  double A = a * a + d * d, B = b * b + d * d, turn = angle(d * (b - a), d * d + a * b);
  *m0 = ((b - a) * (d * d - a * b) / (A * B) + turn / d) / (2 * d * d);
  *m1 = .5 * (b - a) * (b + a) / (A * B);
  *m2 = turn / d - d * d * (*m0);
}
static Integral strip(Pair *w, double t) {
  Integral out = {0};
  const Chart *c = w->c;
  Vec n, p = cp(c, 0, t, &n), dy = minus(w->y.p, p), dl = minus(w->lamp, p);
  double sourceCos = dp(n, dy), lampCos = dp(n, dl), slope = dp(w->y.n, c->ax),
         constant = -dp(w->y.n, dy);
  double a = -c->width, b = c->width;
  if (sourceCos <= 1e-11 || !clipPositive(constant, slope, &a, &b))
    return out;
  int capacity = w->count[0] + w->count[1] + 1;
  Interval dark[capacity], open[capacity + 1];
  int nVis = shadows(w, 0, p, c->ax, a, b, dark);
  double y = dp(dy, c->ax), l = dp(dl, c->ax), D = length(cross3(dy, c->ax)),
         L = length(cross3(dl, c->ax));
  if (D < 1e-14)
    return out;
  double v = c->type ? t / (PI / 2) : .5 + .5 * t / c->height, jac = c->type ? c->r : 1,
         scale = jac * sourceCos / PI;
  int count = openings(dark, nVis, a, b, open);
  w->stats->strips++;
  for (int j = 0; j < count; j++) {
    double lo = open[j].lo, hi = open[j].hi, m0, m1, m2;
    moments(lo - y, hi - y, D, &m0, &m1, &m2);
    double k = constant + slope * y, total = scale * (k * m0 + slope * m1);
    double u =
        .5 * total + scale * (k * m1 + slope * m2 + y * (k * m0 + slope * m1)) / (2 * c->width);
    u = high(0, low(total, u));
    total = high(0, total);
    for (int i = 0; i < 5; i++)
      out.w[i] -= w->lift[i] * total;
    out.w[0] += (1 - v) * (total - u);
    out.w[1] += (1 - v) * u;
    out.w[2] += v * (total - u);
    out.w[3] += v * u;
  }
  if (w->lampDark || !count || lampCos <= 1e-11 || L < 1e-14)
    return out;
  Interval lit[w->count[1] + 2];
  int nLit = litGenerator(w, t, p, lit);
  for (int j = 0, i = 0; j < count && i < nLit;) {
    double lo = high(open[j].lo, lit[i].lo), hi = low(open[j].hi, lit[i].hi);
    if (hi > lo) {
      double factor = scale * lampCos;
      out.w[4] += factor * directPrimitive(l, y, L * L, D * D, constant, slope, lo, hi,
                                           1e-9 * (1 + absd(t)) / (1 + factor), w->stats);
      w->stats->endpoints += 2;
    }
    if (open[j].hi <= lit[i].hi)
      j++;
    else
      i++;
  }
  return out;
}
// A meridian is circular on a spherical octant. For a whole meridian interval,
// its endpoints stay within radius delta of the midpoint. Every segment from
// an eye to the arc stays within the same delta of the central segment.
static int arcState(Pair *w, int eyeId, Vec p, double delta) {
  Vec eye = eyeId ? w->lamp : w->y.p;
  int maybe = 0;
  for (int j = 0; j < w->count[eyeId]; j++) {
    const Cone *c = blockerCone(w, eyeId, j);
    int full = 1;
    for (int k = 0; k < c->n; k++) {
      double d = dp(c->p[k].n, p) - c->p[k].b;
      if (d < -delta - 1e-10)
        goto outside;
      if (d <= delta + 1e-10)
        full = 0;
    }
    const Shape *s = w->s + c->obj;
    if (!s->r) {
      if (full)
        return -1;
    } else {
      Shape outer = *s;
      outer.r += delta;
      outer.h = plus(outer.h, (Vec){delta, delta, delta});
      if (!blocked(&outer, eye, p) && !inSolid(&outer, eye) && !inSolid(&outer, p))
        goto outside;
      if (s->r > delta) {
        Shape inner = *s;
        inner.r -= delta;
        inner.h = minus(inner.h, (Vec){delta, delta, delta});
        if (blocked(&inner, eye, p))
          return -1;
      }
    }
    if (delta == 0) {
      if (blocked(s, eye, p))
        return -1;
    } else
      maybe = 1;
  outside:;
  }
  return maybe ? 0 : 1;
}
static Integral sampleArc(Pair *w, double t, double x, int visible, int lit) {
  Integral out = {0};
  Vec n, p = cp(w->c, x, t, &n), d = minus(w->y.p, p);
  double z = dp(d, d), c = dp(n, d), s = -dp(w->y.n, d);
  if (c <= 0 || s <= 0 || z < 1e-22 || (!visible && arcState(w, 0, p, 0) < 0))
    return out;
  double weight = c * s / (PI * z * z) * w->c->r * w->c->r * cosine(x), u = x / (PI / 2),
         v = t / (PI / 2);
  out.w[0] = weight * (1 - u) * (1 - v);
  out.w[1] = weight * u * (1 - v);
  out.w[2] = weight * (1 - u) * v;
  out.w[3] = weight * u * v;
  d = minus(w->lamp, p);
  z = dp(d, d);
  c = dp(n, d);
  if (c > 0 && lit >= 0 && (lit || arcState(w, 1, p, 0) > 0))
    out.w[4] = weight * c / (z * ROOT(z));
  return out;
}
static double diff(Integral a, Integral b) {
  double err = 0;
  for (int k = 0; k < 5; k++)
    err = high(err, absd(a.w[k] - b.w[k]));
  return err;
}
// Bound the weight of an unresolved visibility interval by projected area
// divided by its smallest possible squared distance. This also covers a
// narrow sliver that all embedded quadrature points miss.
typedef struct {
  Integral value;
  double error;
} Panel;
static Panel arcPanel(Pair *w, double t, double a, double b) {
  double x = .5 * (a + b), h = .5 * (b - a);
  Vec n, p = cp(w->c, x, t, &n);
  double delta = w->c->r * h;
  int vis = arcState(w, 0, p, delta);
  if (vis < 0)
    return (Panel){0};
  int lit = w->lampDark ? -1 : arcState(w, 1, p, delta);
  Integral mid = sampleArc(w, t, x, vis, lit);
  double offset = h * .5773502691896258;
  Integral result =
      times(sum(sampleArc(w, t, x - offset, vis, lit), sampleArc(w, t, x + offset, vis, lit)), h);
  double error = diff(result, times(mid, 2 * h));
  if (vis == 0 || lit == 0) {
    Vec e = minus(w->y.p, p), l = minus(w->lamp, p);
    double distance = high(1e-12, length(e) - delta), lampDistance = high(1e-12, length(l) - delta);
    // The entire spherical arc has |n-n_mid| <= h and |p-p_mid| <= delta.
    double source = low(1, high(0, dp(n, e) + h * length(e) + delta) / distance);
    double receiver = low(1, high(0, -dp(w->y.n, e) + delta) / distance);
    double weight = 2 * h * w->c->r * w->c->r * source * receiver / (PI * distance * distance);
    double lamp = low(1, high(0, dp(n, l) + h * length(l) + delta) / lampDistance) /
                  (lampDistance * lampDistance);
    error = high(error, weight * high(vis == 0 ? 1 : 0, lit >= 0 ? lamp : 0));
  }
  w->stats->numeric += 3;
  return (Panel){result, error};
}
static Integral arcIntegral(Pair *w, double t) {
  typedef struct {
    double a, b;
    Panel p;
  } Leaf;
  // Spend one absolute error budget over the complete clipped meridian.
  // A shadow edge may consume most of the budget at only one angle.
  Leaf heap[128];
  int n = 1;
  heap[0] = (Leaf){0, PI / 2, arcPanel(w, t, 0, PI / 2)};
  Integral value = heap[0].p.value;
  double error = heap[0].p.error;
  while (error > w->tolerance * .25) {
    if (n == 128) {
      w->stats->limited++;
      break;
    }
    Leaf old = heap[0];
    double m = .5 * (old.a + old.b);
    Leaf left = {old.a, m, arcPanel(w, t, old.a, m)}, right = {m, old.b, arcPanel(w, t, m, old.b)};
    error += left.p.error + right.p.error - old.p.error;
    value = sum(value, sum(times(old.p.value, -1), sum(left.p.value, right.p.value)));
    int i = 0;
    while (2 * i + 1 < n) {
      int j = 2 * i + 1;
      if (j + 1 < n && heap[j + 1].p.error > heap[j].p.error)
        j++;
      if (left.p.error >= heap[j].p.error)
        break;
      heap[i] = heap[j];
      i = j;
    }
    heap[i] = left;
    i = n++;
    while (i && heap[(i - 1) / 2].p.error < right.p.error) {
      heap[i] = heap[(i - 1) / 2];
      i = (i - 1) / 2;
    }
    heap[i] = right;
  }
  return value;
}
static Integral slice(Pair *w, double t) {
  if (w->warpDepth > 0) {
    double cs = cosine(t);
    return times(strip(w, w->warpCenter + w->warpDepth * sine(t) / cs), w->warpDepth / (cs * cs));
  }
  return w->c->type == 2 ? arcIntegral(w, t) : strip(w, t);
}
static Panel panel(Pair *w, double a, double b) {
  double m = .5 * (a + b), h = .5 * (b - a), z = h * .7745966692414834;
  Integral mid = slice(w, m);
  Integral value =
      times(sum(times(mid, 8. / 9), times(sum(slice(w, m - z), slice(w, m + z)), 5. / 9)), h);
  return (Panel){value, diff(value, times(mid, 2 * h))};
}
static Integral refine(Pair *w, double a, double b, Panel p, double tol, int depth) {
  if (p.error <= tol + .002 * diff(p.value, (Integral){0}))
    return p.value;
  if (!depth) {
    w->stats->limited++;
    return p.value;
  }
  w->stats->splits++;
  double m = .5 * (a + b);
  Panel l = panel(w, a, m), r = panel(w, m, b);
  double error = l.error + r.error;
  if (!error)
    return sum(l.value, r.value);
  return sum(refine(w, a, m, l, tol * l.error / error, depth - 1),
             refine(w, m, b, r, tol * r.error / error, depth - 1));
}
// Translation along this chart coordinate does not enter a cylindrical tangent equation.
static int invariantAxis(const Chart *c, const Shape *s) {
  if (c->type)
    return -1;
  for (int d = 0; d < 3; d++)
    if (absd(((const double *)&c->v)[d]) == 1 && ((const double *)&c->v)[(d + 1) % 3] == 0 &&
        ((const double *)&c->v)[(d + 2) % 3] == 0) {
      for (int k = 0; k < 3; k++)
        if (absd(((const double *)&s->ax[k])[d]) == 1 &&
            ((const double *)&s->ax[(k + 1) % 3])[d] == 0 &&
            ((const double *)&s->ax[(k + 2) % 3])[d] == 0)
          return k;
    }
  return -1;
}
// An endpoint on this source cannot cross the body if a local supporting
// slab separates the entire chart. Silhouette tangencies still apply.
static int endpointPossible(const Chart *c, const Shape *s) {
  Vec d = minus(c->c, s->c);
  for (int k = 0; k < 3; k++) {
    Vec axis = s->ax[k];
    double e = c->type == 2 ? c->r
                            : c->width * absd(dp(c->ax, axis)) +
                                  (c->type ? c->r : c->height * absd(dp(c->v, axis)));
    if (absd(dp(d, axis)) > e + ((const double *)&s->h)[k] + 1e-9 * (1 + e + length(d)))
      return 0;
  }
  return 1;
}
static int candidates(const Chart *c, const Shape *shapes, int skip, const Cone *all, int n,
                      int *out, int *hidden) {
  int count = 0;
  for (int j = 0; j < n; j++) {
    const Cone *k = all + j;
    if (j == c->object || j == skip || (!k->n && !shapes[j].r))
      continue;
    int full = k->n > 0;
    for (int f = 0; f < k->n; f++) {
      const Plane *p = k->p + f;
      double d = dp(c->c, p->n) - p->b,
             extent = c->type == 2 ? c->r
                                   : c->width * absd(dp(c->ax, p->n)) +
                                         (c->type ? c->r : c->height * absd(dp(c->v, p->n)));
      if (d + extent < 1e-10)
        goto clear;
      if (d - extent < 1e-9)
        full = 0;
    }
    if (full && !shapes[j].r) {
      *hidden = 1;
      return 0;
    }
    int flags = 0;
    if (shapes[j].r)
      flags = 4 * endpointPossible(c, shapes + j) + invariantAxis(c, shapes + j) + 1;
    out[count++] = 8 * j + flags;
  clear:;
  }
  return count;
}
static void insert(double *cuts, int *n, double x, double a, double b) {
  if (x <= a + 1e-10 || x >= b - 1e-10)
    return;
  int j = *n;
  while (j && cuts[j - 1] > x)
    j--;
  if ((j && absd(cuts[j - 1] - x) < 1e-9) || (j < *n && absd(cuts[j] - x) < 1e-9))
    return;
  for (int k = (*n)++; k > j; k--)
    cuts[k] = cuts[k - 1];
  cuts[j] = x;
}
typedef struct {
  double x, t;
} UV;
typedef struct {
  UV a, b;
} Edge;
// Clip the analytic generator bounds to each planar shadow cone. Vertices are
// birth/death events. Intersections of projected edges exchange endpoint owners.
static int footprint(const Chart *c, const Cone *k, UV *poly) {
  UV tmp[24];
  int n = 4;
  poly[0] = (UV){-c->width, -c->height};
  poly[1] = (UV){c->width, -c->height};
  poly[2] = (UV){c->width, c->height};
  poly[3] = (UV){-c->width, c->height};
  for (int j = 0; j < k->n && n; j++) {
    Plane h = k->p[j];
    double origin = dp(h.n, c->c) - h.b, A = dp(h.n, c->ax), B = dp(h.n, c->v);
    int used = 0;
    for (int z = 0; z < n; z++) {
      UV p = poly[z], q = poly[(z + 1) % n];
      double a = origin + A * p.x + B * p.t, b = origin + A * q.x + B * q.t;
      if (a >= 0)
        tmp[used++] = p;
      if ((a < 0 && b > 0) || (a > 0 && b < 0)) {
        double u = a / (a - b);
        tmp[used++] = (UV){p.x + u * (q.x - p.x), p.t + u * (q.t - p.t)};
      }
    }
    n = used;
    for (int z = 0; z < n; z++)
      poly[z] = tmp[z];
  }
  return n;
}
static int planarEvents(Pair *w, double *cuts, int capacity, double a, double b) {
  const Chart *c = w->c;
  int nc = 0, ne = 0;
  Edge edges[16 * (w->count[0] + w->count[1]) + 1];
  insert(cuts, &nc, w->warpCenter, a, b);
  for (int e = 0; e < 2; e++)
    for (int o = 0; o < w->count[e]; o++) {
      UV poly[24];
      int n = footprint(c, blockerCone(w, e, o), poly);
      for (int i = 0; i < n; i++) {
        UV p = poly[i], q = poly[(i + 1) % n];
        if (nc < capacity)
          insert(cuts, &nc, p.t, a, b);
        if (absd(q.t - p.t) > 1e-12)
          edges[ne++] = (Edge){p, q};
      }
    }
  for (int i = 0; i < ne; i++)
    for (int j = 0; j < i; j++) {
      Edge u = edges[i], v = edges[j];
      double lo = high(low(u.a.t, u.b.t), low(v.a.t, v.b.t)),
             hi = low(high(u.a.t, u.b.t), high(v.a.t, v.b.t));
      if (hi <= lo)
        continue;
      double p = (u.b.x - u.a.x) / (u.b.t - u.a.t), q = (v.b.x - v.a.x) / (v.b.t - v.a.t);
      if (p == q)
        continue;
      double t = (v.a.x - q * v.a.t - u.a.x + p * u.a.t) / (p - q);
      if (t > lo && t < hi && nc < capacity)
        insert(cuts, &nc, t, a, b);
    }
  return nc;
}
typedef struct {
  double k, u, v, x;
} Constraint;
static void angularCuts(Constraint r, double *cuts, int *nc, int cap) {
  // tan(theta/2) maps the quarter circle to [0,1].
  double t[2];
  int n = roots(r.k - r.u, r.v, r.k + r.u, 0, 1, t, 0);
  for (int j = 0; j < n && *nc < cap; j++)
    insert(cuts, nc, 2 * angle(t[j], 1), 0, PI / 2);
}
static int cylinderEvents(Pair *w, double *cuts, int cap) {
  const Chart *c = w->c;
  int nc = 0;
  for (int e = 0; e < 2; e++) {
    Vec eye = e ? w->lamp : w->y.p, rel = minus(eye, c->c);
    angularCuts((Constraint){-c->r, dp(c->u, rel), dp(c->v, rel), 0}, cuts, &nc, cap);
    for (int j = 0; j < w->count[e]; j++) {
      const Cone *co = blockerCone(w, e, j);
      Constraint line[14] = {{c->width, 0, 0, 1}, {c->width, 0, 0, -1}};
      for (int i = 0; i < co->n; i++) {
        Plane p = co->p[i];
        line[i + 2] = (Constraint){dp(c->c, p.n) - p.b, c->r * dp(c->u, p.n), c->r * dp(c->v, p.n),
                                   dp(c->ax, p.n)};
      }
      for (int i = 2; i < co->n + 2; i++) {
        Constraint p = line[i];
        if (absd(p.x) < 1e-12) {
          angularCuts(p, cuts, &nc, cap);
          continue;
        }
        for (int k = 0; k < i; k++) {
          Constraint q = line[k];
          Constraint event = {p.k * q.x - q.k * p.x, p.u * q.x - q.u * p.x, p.v * q.x - q.v * p.x,
                              0};
          double t[2];
          int count = roots(event.k - event.u, event.v, event.k + event.u, 0, 1, t, 0);
          for (int z = 0; z < count && nc < cap; z++) {
            double h = t[z], cs = (1 - h * h) / (1 + h * h), sn = 2 * h / (1 + h * h),
                   x = -(p.k + p.u * cs + p.v * sn) / p.x;
            int visible = 1;
            for (int l = 0; l < co->n + 2; l++) {
              Constraint f = line[l];
              if (f.k + f.u * cs + f.v * sn + f.x * x < -1e-9) {
                visible = 0;
                break;
              }
            }
            if (visible)
              insert(cuts, &nc, 2 * angle(h, 1), 0, PI / 2);
          }
        }
      }
    }
  }
  return nc;
}

#include "plane-transfer.h"
// The lamp cones and both interval caches belong to a single fresh pose.
typedef struct {
  LampSlice slices[SLICE_SLOTS];
  Tangents tangents[TANGENT_SLOTS];
  Cone lamps[];
} TransportWorkspace;

unsigned transportBytes(int ns) { return sizeof(TransportWorkspace) + ns * sizeof(Cone); }

void prepareTransport(TransportWorkspace *workspace, const Shape *s, int ns, const Vec *lamp) {
  for (int j = 0; j < SLICE_SLOTS; j++)
    workspace->slices[j].source = 0;
  for (int j = 0; j < TANGENT_SLOTS; j++)
    workspace->tangents[j].source = 0;
  for (int j = 0; j < ns; j++)
    workspace->lamps[j] = shadow(s + j, j, *lamp);
}

// Emit only integrated tiles, in authored order. out and tileIds hold at most
// one receiver's contributions and can immediately be reused for the next.
int transportRow(TransportWorkspace *workspace, const Shape *s, int ns, const Chart *tiles, int nt,
                 const Receiver *receivers, int i, const Vec *lamp, double tolerance, float *output,
                 int *tileIds, Counts *stats, const unsigned char *modes, const Moments *moments,
                 const int *lampIds) {
  Cone eyes[ns];
  int localEye[ns], localLamp[ns];
  Receiver y = receivers[i];
  int count = 0;
  for (int j = 0; j < ns; j++)
    eyes[j] = shadow(s + j, j, y.p);
  for (int k = 0; k < nt; k++) {
    const Chart *c = tiles + k;
    float *weights = output + 5 * count;
    int gpu = 0;
    int mode = modes && y.pad ? modes[(y.pad - 1) * nt + k] : 0;
    int twoSided = mode & 128;
    if (twoSided) {
      if(lampIds && lampIds[k*(ns+1)]<0)continue;
      mode &= 127;
    }
    if (mode == 1) {
      evaluatePlane(c, y, moments + k, weights);
      if (twoSided) for(int j=0;j<4;j++)weights[j]=0;
      tileIds[count++] = k;
      stats->regional++;
      continue;
    }
    if (mode == 2) {
      stats->dark++;
      continue;
    }
    if (mode == 3)
      stats->facing++;
    else if (mode == 4)
      stats->expansion++;
    else if (mode == 5)
      stats->obstruction++;
    else
      stats->individual++;
    if (c->object == y.object)
      continue; // a convex solid cannot see itself
    Vec d = minus(c->c, y.p);
    double ext = c->type == 2 ? c->r
                              : c->width * absd(dp(c->ax, y.n)) +
                                    (c->type ? c->r : c->height * absd(dp(c->v, y.n)));
    if (dp(d, y.n) + ext <= 1e-10)
      continue;
    if (c->type == 0 && dp(d, c->n) >= -1e-10)
      continue;
    int hidden = 0;
    Pair w = {.s = s,
              .c = c,
              .source = k,
              .receiver = i,
              .tangents = workspace->tangents,
              .cache = workspace->slices,
              .y = y,
              .lamp = *lamp,
              .cone = {eyes, workspace->lamps},
              .blockers = {localEye, localLamp},
              .stats = stats,
              .tolerance = tolerance};
    w.count[0] = candidates(c, s, y.object, eyes, ns, localEye, &hidden);
    if (hidden)
      continue;
    if (lampIds) {
      const int *ids = lampIds + k * (ns + 1);
      w.lampDark = ids[0] < 0;
      w.count[1] = w.lampDark ? 0 : ids[0];
      w.blockers[1] = ids + 1;
    } else {
      w.count[1] = candidates(c, s, -1, workspace->lamps, ns, localLamp, &hidden);
      w.lampDark = hidden || (!c->type && dp(minus(*lamp, c->c), c->n) <= 0);
      if (w.lampDark)
        w.count[1] = 0;
    }
    double a = c->type ? 0 : -c->height, b = c->type ? PI / 2 : c->height;
    // Certified clear planar tiles use the GPU's projected angular measure
    // as the constant coefficient of their partition of unity.
    if (!twoSided && !c->type && !w.count[0] && dp(d, y.n) - ext > 1e-8 &&
        absd(dp(d, c->n)) < .8 * high(c->width, c->height)) {
      gpu = 1;
      stats->gpu++;
      w.lift[0] = 1;
    }
    int capacity = 32 + 32 * (w.count[0] + w.count[1]);
    double cuts[capacity + 1];
    int nc = 0;
    if (!c->type) {
      Vec dy = minus(y.p, c->c), lateral = cross3(c->ax, c->v);
      double h2 = dp(lateral, lateral);
      w.warpCenter = dp(cross3(c->ax, dy), lateral) / h2;
      double depth = absd(dp(dy, c->n)) / ROOT(h2), over = high(0, absd(dp(dy, c->ax)) - c->width);
      depth = ROOT(depth * depth + over * over);
      if (depth > 1e-12 && depth < .35 * c->height)
        w.warpDepth = depth;
      nc = planarEvents(&w, cuts, capacity, a, b);
    }
    if (c->type == 1)
      nc = cylinderEvents(&w, cuts, capacity);
    if (nc == capacity)
      stats->limited++;
    cuts[nc++] = b;
    Integral value = {0};
    for (int j = 0; j < nc; j++) {
      double hi = cuts[j];
      double L = a, H = hi;
      if (w.warpDepth) {
        L = angle(a - w.warpCenter, w.warpDepth);
        H = angle(hi - w.warpCenter, w.warpDepth);
      }
      value = sum(value, refine(&w, L, H, panel(&w, L, H),
                                tolerance * (hi - a) / (c->type ? PI / 2 : 2 * c->height), 22));
      a = hi;
    }
    for (int j = 0; j < 5; j++)
      weights[j] = (j < 4 && twoSided) || (j == 4 && hidden) ? 0 : value.w[j];
    tileIds[count++] = gpu ? ~k : k;
  }
  return count;
}
void *memcpy(void *dst, const void *src, unsigned long n) {
  volatile unsigned char *a = dst;
  const unsigned char *b = src;
  for (unsigned long i = 0; i < n; i++)
    a[i] = b[i];
  return dst;
}
void *memset(void *dst, int v, unsigned long n) {
  volatile unsigned char *a = dst;
  for (unsigned long i = 0; i < n; i++)
    a[i] = v;
  return dst;
}
