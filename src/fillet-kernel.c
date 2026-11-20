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
  return occludesSegment(v->s + k, v->light, p);
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
