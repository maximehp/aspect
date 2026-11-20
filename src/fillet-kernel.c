#include "solid-hit.h"
static Vec surfNormal(const Shape *s, Vec p) {
  if (!s->r)
    return s->ax[2];
  Vec u = local(s, minus(p, s->c));
  double v[] = {u.x, u.y, u.z}, h[] = {s->h.x, s->h.y, s->h.z};
  Vec n = {0};
  for (int k = 0; k < 3; k++)
    n = plus(n, scale3(s->ax[k], __builtin_copysign(high(0, absd(v[k]) - h[k] + s->r), v[k])));
  double len = __builtin_sqrt(dp(n, n));
  return len ? scale3(n, 1 / len) : n;
}
static int nearest(const Shape *s, int n, Vec o, Vec d, int skip, double limit, double *t) {
  int id = -1;
  for (int i = 0; i < n; i++)
    if (i != skip) {
      double z = hit(s + i, o, d, limit);
      if (z < limit) {
        limit = z;
        id = i;
      }
    }
  *t = limit;
  return id;
}
// Intersect a beam of segments with the outside support planes of a shadow.
// A single separating plane rules out all endpoints in the convex hull.
static int outside(const Plane *planes, Range r, Vec a, Vec b) {
  for (int i = 0; i < r.n; i++) {
    Plane h = planes[r.first + i];
    if (dp(h.n, a) < h.b - 1e-10 && dp(h.n, b) < h.b - 1e-10)
      return 1;
  }
  return 0;
}
typedef struct {
  const Shape *s;
  const Plane *cam, *lamp;
  const Range *cr, *lr;
  const int *order, *groups;
  int n, w, h, cap, used;
  Vec eye, light, base, dx, dy;
  Run *out;
  int *stats;
  const int *blockers;
  const Range *blocks;
  const double *nearDepth;
} View;
static void emit(View *v, int x, int end, int y, int id, int lit) {
  if (id < 0)
    return;
  if (v->used && v->used <= v->cap) {
    Run *p = v->out + v->used - 1;
    if (p->y == y && p->end == x && p->id == id && p->lit == lit) {
      p->end = end;
      return;
    }
  }
  if (v->used < v->cap)
    v->out[v->used] = (Run){x, end, y, id, lit};
  v->used++;
}
static int lampHit(View *v, int k, Vec p) {
  v->stats[3]++;
  return segmentHit(v->s + k, v->light,p, 1 - 1e-9) < 1;
}
static int shadowAt(View *v, int id, Vec p, const int *ids, int n) {
  for (int i = 0; i < n; i++)
    if (ids[i] != id && lampHit(v, ids[i], p))
      return 0;
  return 1;
}
// On a planar receiver each screen run maps projectively to a straight line.
// Clip that line once against each blocker's outer cone. A rounded blocker's
// true shadow is convex, so after finding a hit, two binary searches recover
// the complete covered interval at pixel-center precision.
typedef struct {
  int l, r;
} Interval;
static int coveredAt(View *v, int id, int block, Vec row, int x) {
  Vec d = plus(row, scale3(v->dx, x));
  const Shape *s = v->s + id;
  double t = dp(minus(s->c, v->eye), s->ax[2]) / dp(d, s->ax[2]);
  Vec p = plus(v->eye, scale3(d, t));
  return lampHit(v, block, p);
}
static int findCovered(View *v, int id, int k, Vec row, int l, int r) {
  // Used only inside a conservative tangent envelope, including tiny grazes.
  if (l >= r)
    return -1;
  int m = (l + r) / 2;
  if (coveredAt(v, id, k, row, m))
    return m;
  int p = findCovered(v, id, k, row, l, m);
  return p >= 0 ? p : findCovered(v, id, k, row, m + 1, r);
}
static void planarRun(View *v, int id, int x, int end, int y, Vec row, const int *ids, int n) {
  Vec d = plus(row, scale3(v->dx, x)), z = plus(row, scale3(v->dx, end - 1));
  const Shape *s = v->s + id;
  double h = dp(minus(s->c, v->eye), s->ax[2]), a = h / dp(d, s->ax[2]), b = h / dp(z, s->ax[2]);
  Vec p = plus(v->eye, scale3(d, a)), q = plus(v->eye, scale3(z, b));
  Interval spans[n ? n : 1];
  int count = 0;
  for (int j = 0; j < n; j++) {
    int k = ids[j];
    if (k == id)
      continue;
    Range cone = v->lr[k];
    double l = 0, r = 1;
    for (int i = 0; i < cone.n; i++) {
      Plane h = v->lamp[cone.first + i];
      double offset = h.b + (i == 0 && !v->s[k].r ? 1e-9 : -1e-11);
      double u = dp(p, h.n) - offset, w = dp(q, h.n) - offset;
      if (u < 0 && w < 0) {
        r = -1;
        break;
      }
      if (u < 0)
        l = high(l, u / (u - w));
      else if (w < 0)
        r = low(r, u / (u - w));
      if (l > r)
        break;
    }
    if (l > r)
      continue;
    // Invert perspective interpolation of the two camera-forward depths.
    int L = x + (int)__builtin_ceil((end - 1 - x) * l * b / (a * (1 - l) + l * b) - 1e-9);
    int R = x + 1 + (int)__builtin_floor((end - 1 - x) * r * b / (a * (1 - r) + r * b) + 1e-9);
    if (R <= L)
      continue;
    if (v->s[k].r || !cone.n) {
      int m = findCovered(v, id, k, row, L, R);
      if (m < 0)
        continue;
      int lo = L, hi = m;
      while (lo < hi) {
        int z = lo + (hi - lo) / 2;
        if (coveredAt(v, id, k, row, z))
          hi = z;
        else
          lo = z + 1;
      }
      L = lo;
      lo = m;
      hi = R - 1;
      while (lo < hi) {
        int z = lo + (hi - lo + 1) / 2;
        if (coveredAt(v, id, k, row, z))
          lo = z;
        else
          hi = z - 1;
      }
      R = lo + 1;
    }
    if (L == x && R == end) {
      emit(v, x, end, y, id, 0);
      v->stats[end - x == 1] += end - x;
      return;
    }
    int t = count++;
    while (t && spans[t - 1].l > L) {
      spans[t] = spans[t - 1];
      t--;
    }
    spans[t] = (Interval){L, R};
  }
  int start = x;
  for (int i = 0; i < count; i++) {
    Interval s = spans[i];
    if (s.r <= start)
      continue;
    if (s.l > start)
      emit(v, start, s.l, y, id, 1);
    int last = s.r;
    while (i + 1 < count && spans[i + 1].l <= last) {
      i++;
      if (spans[i].r > last)
        last = spans[i].r;
    }
    emit(v, s.l > start ? s.l : start, last, y, id, 0);
    start = last;
  }
  if (start < end)
    emit(v, start, end, y, id, 1);
  v->stats[end - x == 1] += end - x;
}
static void shadeSpan(View *v, int id, int x, int end, int y, Vec d0, const int *ids, int n) {
  Vec d = plus(d0, scale3(v->dx, x)), z = plus(d0, scale3(v->dx, end - 1));
  const Shape *s = v->s + id;
  if (!s->r) {
    planarRun(v, id, x, end, y, d0, ids, n);
    return;
  }
  if (x + 1 == end) {
    double t = hit(s, v->eye, d, FAR);
    Vec p = plus(v->eye, scale3(d, t));
    emit(v, x, end, y, id, shadowAt(v, id, p, ids, n));
    v->stats[1]++;
    return;
  }
  if (!n) {
    emit(v, x, end, y, id, 1);
    v->stats[0] += end - x;
    return;
  }
  int candidates[n], count = 0;
  // A chord between two first hits is inside the convex solid. Its tangent
  // plane at the middle hit bounds the same surface from the eye side. The
  // four endpoints enclose the entire curved row section in the ray plane.
  Vec p = plus(v->eye, scale3(d, hit(s, v->eye, d, FAR))),
      q = plus(v->eye, scale3(z, hit(s, v->eye, z, FAR)));
  Vec mid = scale3(plus(d, z), .5), contact = plus(v->eye, scale3(mid, hit(s, v->eye, mid, FAR)));
  Vec normal = surfNormal(s, contact);
  double H = dp(normal, minus(contact, v->eye)), ld = dp(normal, d), rd = dp(normal, z);
  int bounded = H < 0 && ld < 0 && rd < 0;
  Vec nearP = bounded ? plus(v->eye, scale3(d, H / ld)) : v->eye,
      nearQ = bounded ? plus(v->eye, scale3(z, H / rd)) : v->eye;
  for (int i = 0; i < n; i++) {
    int k = ids[i];
    if (k == id)
      continue;
    Range r = v->lr[k];
    int clear = 0;
    for (int j = 0; j < r.n; j++) {
      Plane h = v->lamp[r.first + j];
      double b = h.b - 1e-10;
      if (dp(h.n, p) < b && dp(h.n, q) < b && dp(h.n, nearP) < b && dp(h.n, nearQ) < b) {
        clear = 1;
        break;
      }
    }
    if (clear)
      continue;
    if (bounded && lampHit(v, k, p) && lampHit(v, k, q) && lampHit(v, k, nearP) &&
        lampHit(v, k, nearQ)) {
      emit(v, x, end, y, id, 0);
      v->stats[0] += end - x;
      return;
    }
    candidates[count++] = k;
  }
  if (!count) {
    emit(v, x, end, y, id, 1);
    v->stats[0] += end - x;
    return;
  }
  int m = (x + end) / 2;
  shadeSpan(v, id, x, m, y, d0, candidates, count);
  shadeSpan(v, id, m, end, y, d0, candidates, count);
}
typedef struct {
  int id, group, first, last;
} Window;
static void primary(View *v, int x, int end, int y, Vec row, const Window *win, int n,
                    const int *all) {
  Vec d = plus(row, scale3(v->dx, x)), z = plus(row, scale3(v->dx, end - 1));
  int best = -1, group = -1, uncertain = 0;
  double near = FAR;
  Window remain[n ? n : 1];
  int nr = 0;
  for (int i = 0; i < n; i++) {
    Window w = win[i];
    if (w.last < x || w.first >= end)
      continue;
    if (best >= 0 && group != w.group)
      break;
    remain[nr++] = w;
    if (x + 1 == end) {
      double t = hit(v->s + w.id, v->eye, d, near);
      if (t < near) {
        near = t;
        best = w.id;
        group = w.group;
      }
      continue;
    }
    // A convex silhouette covers the full row interval if it covers both ends.
    if (best >= 0 || w.first > x || w.last < end - 1) {
      uncertain = 1;
      continue;
    }
    if (hit(v->s + w.id, v->eye, d, FAR) < FAR && hit(v->s + w.id, v->eye, z, FAR) < FAR) {
      best = w.id;
      group = w.group;
    } else
      uncertain = 1;
  }
  if (uncertain && x + 1 < end && best >= 0) {
    double far = high(hit(v->s + best, v->eye, d, FAR), hit(v->s + best, v->eye, z, FAR));
    int separated = 1;
    for (int j = 0; j < nr; j++)
      if (remain[j].id != best &&
          (remain[j].group != group || v->nearDepth[remain[j].id] <= far + 1e-9)) {
        separated = 0;
        break;
      }
    if (separated)
      uncertain = 0;
  }
  if (x + 1 == end || !uncertain) {
    if (best >= 0) {
      v->stats[2] += end - x;
      Range r = v->blocks[best];
      if (r.n < 0) {
        emit(v, x, end, y, best, 0);
        v->stats[end - x == 1] += end - x;
      } else
        shadeSpan(v, best, x, end, y, row, v->blockers + r.first, r.n);
    }
    return;
  }
  int m = (x + end) / 2;
  primary(v, x, m, y, row, remain, nr, all);
  primary(v, m, end, y, row, remain, nr, all);
}
int renderRuns(const Shape *s, int n, const Plane *cam, const Range *cr, const Plane *lamp,
               const Range *lr, const int *order, const int *groups, const Vec *frame,
               const int *blockers, const Range *blocks, const double *nearDepth, int w, int h,
               Run *out, int cap, int *stats) {
  predicateBase=s;predicateCount=n;
  View v = {s,        cam, lamp,  cr,       lr,       order,    groups,   n,
            w,        h,   cap,   0,        frame[0], frame[1], frame[2], frame[3],
            frame[4], out, stats, blockers, blocks,   nearDepth};
  int ids[n ? n : 1];
  Window win[n ? n : 1];
  for (int i = 0; i < n; i++)
    ids[i] = i;
  for (int y = 0; y < h; y++) {
    Vec row = plus(v.base, scale3(v.dy, y));
    int count = 0;
    for (int i = 0; i < n; i++) {
      int id = order[i];
      if (nearDepth[n + id] <= 1e-8)
        continue;
      Range r = cr[id];
      double l = 0, b = w - 1;
      // Every lateral support plane passes through the eye. Plane zero is
      // the near cap. Exact finite-distance hits decide that condition.
      for (int j = 1; j < r.n; j++) {
        Vec N = cam[r.first + j].n;
        double u = dp(N, row), k = dp(N, v.dx);
        if (k > 0)
          l = high(l, (-1e-11 - u) / k);
        else if (k < 0)
          b = low(b, (-1e-11 - u) / k);
        else if (u < -1e-11) {
          l = 1;
          b = 0;
        }
        if (l > b)
          break;
      }
      if (l <= b)
        win[count++] = (Window){id, groups[i], (int)__builtin_ceil(l), (int)__builtin_floor(b)};
    }
    int edges[2 * count + 2], ne = 1;
    edges[0] = 0;
    for (int j = 0; j < count; j++) {
      int v[2] = {win[j].first, win[j].last + 1};
      for (int k = 0; k < 2; k++) {
        int x = v[k], i = ne;
        if (x <= 0 || x >= w)
          continue;
        while (i && edges[i - 1] > x)
          i--;
        if (i && edges[i - 1] == x)
          continue;
        for (int t = ne; t > i; t--)
          edges[t] = edges[t - 1];
        edges[i] = x;
        ne++;
      }
    }
    edges[ne++] = w;
    for (int k = 1; k < ne; k++)
      primary(&v, edges[k - 1], edges[k], y, row, win, count, ids);
  }
  return v.used;
}

// Quadrature sites are on the actual plane, quarter-cylinder or spherical
// octant. Rays leaving one site account for both directions of reciprocal
// exchange. A ray from a first obstruction also removes its stationary link.
typedef struct {
  Vec p, n;
  double area;
  int object, mode;
} Site;
typedef struct {
  Vec center, axis;
  double lo, hi, bottom, top;
} Mode;
typedef struct {
  int first, count;
} Group;
typedef struct {
  int a, b;
  float area, u, v, uv;
} Coupling;
static int modeAt(const Shape *s, int id, Vec p, Vec n, const Group *g, const Mode *m, int *face) {
  const Shape *b = s + id;
  Vec localP = local(b, minus(p, b->c));
  double u = localP.x, v = localP.y;
  int f = 0;
  if (b->r) {
    Vec normal = local(b, n);
    double co[] = {normal.x, normal.y, normal.z}, pos[] = {localP.x, localP.y, localP.z},
           h[] = {b->h.x, b->h.y, b->h.z};
    int k = absd(co[1]) > absd(co[0]) ? 1 : 0;
    if (absd(co[2]) > absd(co[k]))
      k = 2;
    int sign = co[k] < 0 ? -1 : 1;
    f = 2 * k + (sign < 0);
    u = sign * pos[(k + 1) % 3] / h[(k + 1) % 3];
    v = pos[(k + 2) % 3] / h[(k + 2) % 3];
  }
  *face = f;
  Group z = g[id * 6 + f];
  int best = z.first;
  for (int j = z.first; j < z.first + z.count; j++) {
    best = j;
    if (u <= m[j].hi + 1e-10 && u >= m[j].lo - 1e-10 && v >= m[j].bottom - 1e-10 &&
        v <= m[j].top + 1e-10)
      break;
  }
  return best;
}
static double term(const Mode *m, int j, Vec p) {
  return high(-1, low(1, dp(minus(p, m[j].center), m[j].axis)));
}
static void addTerm(Coupling *out, int *slots, int cap, int *used, int a, int b, double area,
                    double u, double v) {
  if (a > b) {
    int t = a;
    a = b;
    b = t;
    double x = u;
    u = v;
    v = x;
  }
  unsigned hash = (unsigned)a * 2654435761u + (unsigned)b * 2246822519u;
  unsigned j = (hash ^ (hash >> 16)) & (2 * cap - 1);
  while (slots[j]) {
    Coupling *c = out + slots[j] - 1;
    if (c->a == a && c->b == b)
      break;
    j = (j + 1) & (2 * cap - 1);
  }
  if (!slots[j]) {
    if (*used >= cap) {
      *used = cap + 1;
      return;
    }
    out[*used] = (Coupling){a, b};
    slots[j] = ++*used;
  }
  Coupling *c = out + slots[j] - 1;
  c->area += area;
  c->u += area * u;
  c->v += area * v;
  c->uv += area * u * v;
}
static int nearestList(const Shape *s, const int *ids, int n, Vec o, Vec d, int fixed, double *t) {
  int best = -1;
  double limit = FAR;
  for (int i = 0; i < n; i++) {
    int k = ids[i];
    if (k >= fixed)
      continue;
    double z = hit(s + k, o, d, limit);
    if (z < limit) {
      limit = z;
      best = k;
    }
  }
  *t = limit;
  return best;
}
static double extent(Vec axis, Vec half) {
  return absd(axis.x) * half.x + absd(axis.y) * half.y + absd(axis.z) * half.z;
}
int surfaceExchange(const Shape *s, int n, int fixed, const Site *sites, int ns, const Group *g,
                    const Mode *m, const Vec *directions, int nd, Coupling *out, int *slots,
                    int cap, int first, int stride) {
  for (int i = 0; i < 2 * cap; i++)
    slots[i] = 0;
  int used = 0;
  for (int j = first; j < nd; j += stride) {
    Vec w = directions[j];
    // These three axes are perpendicular to the beam. A separating axis
    // rejects the other object for every integration site on the emitter.
    Vec lateral[3] = {{0, w.z, -w.y}, {-w.z, 0, w.x}, {w.y, -w.x, 0}};
    int front[n ? n : 1], rear[n ? n : 1], nf = 0, nr = 0, last = -1;
    for (int k = 0; k < ns; k++) {
      Site site = sites[k];
      double u = term(m, site.mode, site.p);
      if (site.object != last) {
        last = site.object;
        nf = nr = 0;
        for (int t = 0; t < n; t++)
          if (t != last) {
            Vec delta = minus(s[t].c, s[last].c), h = plus(s[t].extent, s[last].extent);
            int clear = 0;
            for (int c = 0; c < 3; c++)
              if (absd(dp(delta, lateral[c])) > extent(lateral[c], h) + 1e-10) {
                clear = 1;
                break;
              }
            if (clear)
              continue;
            double z = dp(delta, w), r = extent(w, h) + 1e-10;
            if (z + r >= 0)
              front[nf++] = t;
            if (z - r <= 0)
              rear[nr++] = t;
          }
      }
      double cosine = dp(site.n, w);
      if (absd(cosine) < 1e-10)
        continue;
      Vec d = cosine < 0 ? scale3(w, -1) : w;
      double t;
      int id = nearestList(s, cosine < 0 ? rear : front, cosine < 0 ? nr : nf, site.p, d, n, &t);
      if (id < 0)
        continue;
      double area = site.area * absd(cosine) * 2 / nd;
      Vec q = plus(site.p, scale3(d, t)), normal = surfNormal(s + id, q);
      int side = dp(normal, d) > 0, f;
      if (id >= fixed && side)
        continue;
      int b = modeAt(s, id, q, normal, g, m, &f), B = 2 * b + side;
      addTerm(out, slots, cap, &used, 2 * site.mode, B, area * (id >= fixed ? .5 : 1), u,
              term(m, b, q));
      if (used > cap)
        return -1;
      if (cosine >= 0 || id >= fixed)
        continue;
      double t2;
      Vec back = scale3(d, -1);
      int other = nearestList(s, front, nf, site.p, back, fixed, &t2);
      if (other < 0 || other == id)
        continue;
      Vec q2 = plus(site.p, scale3(back, t2));
      int a = modeAt(s, other, q2, s[other].ax[2], g, m, &f);
      addTerm(out, slots, cap, &used, B, 2 * a + (dp(s[other].ax[2], back) > 0), -area,
              term(m, b, q), term(m, a, q2));
      if (used > cap)
        return -1;
    }
  }

  return used;
}
void directSites(const Shape *s, int n, const Site *sites, int ns, Vec *light, const Mode *m,
                 double *out) {
  for (int i = 0; i < ns; i++) {
    Site v = sites[i];
    Vec d = minus(*light, v.p);
    double z = dp(d, d), c = dp(v.n, d);
    if (c <= 0 || !z)
      continue;
    double t;
    if (nearest(s, n, v.p, d, v.object, 1 - 1e-9, &t) >= 0)
      continue;
    double f = v.area * c / (z * __builtin_sqrt(z));
    out[2 * v.mode] += f;
    out[2 * v.mode + 1] += f * term(m, v.mode, v.p);
  }
}
static void litSample(Vec p, Vec light, Vec origin, Vec axis, Vec normal, double area,
                      double *out) {
  Vec d = minus(light, p);
  double z = dp(d, d), f = area * absd(dp(d, normal)) / (z * __builtin_sqrt(z));
  out[0] += f;
  out[1] += f * dp(minus(p, origin), axis);
}
static void recoverTriangle(const Shape *s, const Plane *planes, const Range *ranges,
                            const int *ids, int n, Vec a, Vec b, Vec c, Vec light, Vec origin,
                            Vec axis, Vec normal, double area, int depth, double *out) {
  int active[n ? n : 1], count = 0;
  for (int j = 0; j < n; j++) {
    int id = ids[j];
    Range r = ranges[id];
    int clear = 0;
    for (int i = 0; i < r.n; i++) {
      Plane h = planes[r.first + i];
      double z = h.b - 1.e-10;
      if (dp(h.n, a) < z && dp(h.n, b) < z && dp(h.n, c) < z) {
        clear = 1;
        break;
      }
    }
    if (clear)
      continue;
    if (hit(s + id, light, minus(a, light), 1 - 1e-9) < 1 &&
        hit(s + id, light, minus(b, light), 1 - 1e-9) < 1 &&
        hit(s + id, light, minus(c, light), 1 - 1e-9) < 1)
      return;
    active[count++] = id;
  }
  if (!count) {
    Vec sum = scale3(plus(plus(a, b), c), 1. / 6);
    litSample(plus(sum, scale3(a, .5)), light, origin, axis, normal, area / 3, out);
    litSample(plus(sum, scale3(b, .5)), light, origin, axis, normal, area / 3, out);
    litSample(plus(sum, scale3(c, .5)), light, origin, axis, normal, area / 3, out);
    return;
  }
  if (depth > 0) {
    double ab = dp(minus(a, b), minus(a, b)), bc = dp(minus(b, c), minus(b, c)),
           ca = dp(minus(c, a), minus(c, a));
    if (bc >= ab && bc >= ca) {
      Vec t = a;
      a = b;
      b = c;
      c = t;
    } else if (ca >= ab) {
      Vec t = b;
      b = a;
      a = c;
      c = t;
    }
    Vec mid = scale3(plus(a, b), .5);
    recoverTriangle(s, planes, ranges, active, count, a, mid, c, light, origin, axis, normal,
                    area * .5, depth - 1, out);
    recoverTriangle(s, planes, ranges, active, count, mid, b, c, light, origin, axis, normal,
                    area * .5, depth - 1, out);
    return;
  }
  Vec p = scale3(plus(plus(a, b), c), 1. / 3);
  for (int i = 0; i < count; i++)
    if (hit(s + active[i], light, minus(p, light), 1 - 1e-9) < 1)
      return;
  litSample(p, light, origin, axis, normal, area, out);
}
void recoverFlux(const Shape *s, int n, const Plane *planes, const Range *ranges, const Vec *data,
                 int count, double *out) {
  out[0] = out[1] = 0;
  Vec a = data[4];
  int ids[n ? n : 1];
  for (int i = 0; i < n; i++)
    ids[i] = i;
  for (int j = 1; j + 1 < count; j++) {
    Vec b = data[4 + j], c = data[5 + j], u = minus(b, a), v = minus(c, a);
    Vec w = {u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
    double area = .5 * __builtin_sqrt(dp(w, w));
    if (area > 1e-15)
      recoverTriangle(s, planes, ranges, ids, n, a, b, c, data[0], data[1], data[2], data[3], area,
                      8, out);
  }
}
