// Freestanding CPU WebAssembly. JS lays out the polygons and provides scratch
// space proportional to this scene. No fixed limits on active surfaces.
typedef struct {
  float x, y, z;
} V;
typedef struct {
  int start, count, mode, owner, solid;
  float area;
  V n, center, color, axis;
} Poly;
typedef struct {
  float lo, hi, a, b, c, cosine, mx, my, mc;
} Project;
typedef struct {
  int polygon;
  float low, high;
} Segment;
typedef struct {
  float x;
  int encoded;
} Edge;
static float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static float min(float a, float b) { return a < b ? a : b; }
static float max(float a, float b) { return a > b ? a : b; }
static void sort(Edge *a, int lo, int hi) {
  while (lo < hi) {
    int i = lo, j = hi;
    float pivot = a[(lo + hi) / 2].x;
    while (i <= j) {
      while (a[i].x < pivot)
        i++;
      while (a[j].x > pivot)
        j--;
      if (i <= j) {
        Edge e = a[i];
        a[i++] = a[j];
        a[j--] = e;
      }
    }
    if (j - lo < hi - i) {
      sort(a, lo, j);
      lo = i;
    } else {
      sort(a, i, hi);
      hi = j;
    }
  }
}
static void exchange(Poly *p, Project *q, int a, int b, float area, int n, float *K, float *mom,
                     int fixed, float x, float y) {
  Poly *A = p + a, *B = p + b;
  if (A->owner == B->owner)
    return;
  int sa = q[a].cosine < 0, sb = q[b].cosine > 0;
  if ((A->solid && sa) || (B->solid && sb))
    return;
  int i = 2 * A->mode + sa, j = 2 * B->mode + sb;
  K[i * n + j] += area / A->area;
  K[j * n + i] += area / B->area;
  // First moment of the same projected footprint along the short face axis.
  if (A->solid)
    mom[(a - fixed) * n + j] +=
        area / A->area * max(-1, min(1, q[a].mx * x + q[a].my * y + q[a].mc));
  if (B->solid)
    mom[(b - fixed) * n + i] +=
        area / B->area * max(-1, min(1, q[b].mx * x + q[b].my * y + q[b].mc));
}
__attribute__((visibility("default"))) void sweep(Poly *p, int count, V *verts, int nv, V *frame,
                                                  int dirs, int bins, int n, float *K,
                                                  int correction, char *scratch, float *mom,
                                                  int fixed) {
  float *xs = (float *)scratch, *ys = xs + nv;
  Project *q = (Project *)(ys + nv);
  Segment *segments = (Segment *)(q + count);
  Edge *edges = (Edge *)(segments + count);
  int *hits = (int *)(edges + 2 * count), *active = hits + count;
  Edge *starts = (Edge *)(active + count);
  if (!count)
    return;
  for (int d = 0; d < dirs; d++) {
    V u = frame[3 * d], v = frame[3 * d + 1], w = frame[3 * d + 2];
    for (int i = 0; i < nv; i++) {
      xs[i] = dot(verts[i], u);
      ys[i] = dot(verts[i], v);
    }
    float bottom = 1e20f, top = -1e20f, staticLo = 1e20f, staticHi = -1e20f, moveLo = 1e20f,
          moveHi = -1e20f;
    for (int i = 0; i < count; i++) {
      Poly f = p[i];
      float lo = 1e20f, hi = -1e20f;
      for (int j = f.start; j < f.start + f.count; j++) {
        lo = min(lo, ys[j]);
        hi = max(hi, ys[j]);
      }
      float g = dot(f.n, w);
      q[i] = (Project){lo, hi, -dot(f.n, u) / g, -dot(f.n, v) / g, dot(f.n, f.center) / g, g};
      if (f.solid) {
        float dw = dot(f.axis, w);
        q[i].mx = dot(f.axis, u) + dw * q[i].a;
        q[i].my = dot(f.axis, v) + dw * q[i].b;
        q[i].mc = dw * q[i].c - dot(f.axis, f.center);
      }
      if (f.solid) {
        moveLo = min(moveLo, lo);
        moveHi = max(moveHi, hi);
      } else {
        staticLo = min(staticLo, lo);
        staticHi = max(staticHi, hi);
      }
      bottom = min(bottom, lo);
      top = max(top, hi);
    }
    float step = staticHi > staticLo ? (staticHi - staticLo) / bins : 0.1f;
    if (staticLo > staticHi)
      staticLo = 0;
    if (correction) {
      bottom = moveLo;
      top = moveHi;
    }
    if (top <= bottom)
      continue;
    int first = (int)__builtin_floorf((bottom - staticLo) / step),
        end = (int)__builtin_ceilf((top - staticLo) / step);
    for (int i = 0; i < count; i++)
      starts[i] = (Edge){q[i].lo, i};
    sort(starts, 0, count - 1);
    int begin = 0, na = 0;
    for (int row = first; row < end; row++) {
      float y0 = staticLo + row * step, y1 = y0 + step;
      int ne = 0, ns = 0;
      while (begin < count && starts[begin].x < y1)
        active[na++] = starts[begin++].encoded;
      int kept = 0;
      for (int k = 0; k < na; k++) {
        int i = active[k];
        Project a = q[i];
        if (a.hi <= y0 || __builtin_fabsf(a.cosine) < 1e-7f)
          continue;
        active[kept++] = i;
        float lo = max(a.lo, y0), hi = min(a.hi, y1), y = (lo + hi) * .5f, L = 1e20f, R = -1e20f;
        int start = p[i].start, end = start + p[i].count;
        for (int j = start, k = end - 1; j < end; k = j++) {
          if ((ys[k] > y) != (ys[j] > y)) {
            float x = xs[k] + (xs[j] - xs[k]) * ((y - ys[k]) / (ys[j] - ys[k]));
            L = min(L, x);
            R = max(R, x);
          }
        }
        if (L >= R)
          continue;
        segments[ns] = (Segment){i, lo, hi};
        edges[ne++] = (Edge){L, ns + 1};
        edges[ne++] = (Edge){R, -ns - 1};
        ns++;
      }
      na = kept;
      if (!ne)
        continue;
      if (correction) {
        float left = 1e20f, right = -1e20f;
        for (int i = 0; i < ne; i++) {
          int id = edges[i].encoded;
          int j = id > 0 ? id - 1 : -id - 1;
          if (p[segments[j].polygon].solid) {
            left = min(left, edges[i].x);
            right = max(right, edges[i].x);
          }
        }
        if (left >= right)
          continue;
        // Skip fixed segments outside the projected footprint of every edited body.
        int out = 0;
        for (int i = 0; i < ne; i += 2) {
          if (edges[i + 1].x <= left || edges[i].x >= right)
            continue;
          Edge a = edges[i], b = edges[i + 1];
          a.x = max(a.x, left);
          b.x = min(b.x, right);
          edges[out++] = a;
          edges[out++] = b;
        }
        ne = out;
      }
      sort(edges, 0, ne - 1);
      int nh = 0, moving = 0;
      float prev = edges[0].x;
      for (int e = 0; e < ne; e++) {
        float x = edges[e].x;
        if (x > prev && nh > 1 && (!correction || moving)) {
          float mid = (prev + x) * .5f;
          for (float low = y0; low < y1;) {
            float high = y1;
            for (int i = 0; i < nh; i++) {
              Segment t = segments[hits[i]];
              if (t.low > low)
                high = min(high, t.low);
              if (t.high > low)
                high = min(high, t.high);
            }
            float y = (low + high) * .5f;
            for (int i = 1; i < nh; i++) {
              int s = hits[i], j = i;
              float z = q[segments[s].polygon].a * mid + q[segments[s].polygon].b * y +
                        q[segments[s].polygon].c;
              while (j > 0) {
                int t = hits[j - 1];
                if (q[segments[t].polygon].a * mid + q[segments[t].polygon].b * y +
                        q[segments[t].polygon].c <=
                    z)
                  break;
                hits[j] = t;
                j--;
              }
              hits[j] = s;
            }
            int previous = -1, stationary = -1;
            for (int i = 0; i < nh; i++) {
              Segment b = segments[hits[i]];
              if (b.low > low || b.high < high)
                continue;
              int id = b.polygon;
              float area = (x - prev) * (high - low) * (2.f / dirs);
              if (previous >= 0 && (!correction || p[id].solid || p[previous].solid))
                exchange(p, q, previous, id, area, n, K, mom, fixed, mid, y);
              if (correction && !p[id].solid) {
                if (stationary >= 0 && stationary != previous)
                  exchange(p, q, stationary, id, -area, n, K, mom, fixed, mid, y);
                stationary = id;
              }
              previous = id;
            }
            low = high;
          }
        }
        int s = edges[e].encoded;
        if (s > 0) {
          hits[nh++] = s - 1;
          moving += p[segments[s - 1].polygon].solid;
        } else {
          moving -= p[segments[-s - 1].polygon].solid;
          int t = 0;
          while (hits[t] != -s - 1)
            t++;
          hits[t] = hits[--nh];
        }
        prev = x;
      }
    }
  }
}

// Compact positive entries by row in place. The fixed-point update stops when
// every incident RGB coefficient changes by at most tolerance in lamp units.
int equilibrium(Poly *p, int n, float *K, float *direct, double *incoming, double *emitted,
                int *index, int *counts, double *bounds, float *mom, int fixed, float *slopes) {
  double maxRow = 0, residual = 0;
  for (int i = 0; i < n; i++) {
    float *row = K + i * n;
    double sum = 0;
    int count = 0;
    for (int j = 0; j < n; j++)
      if (row[j] > 0) {
        sum += row[j];
        index[count++] = j;
      } else
        row[j] = 0;
    if (sum > maxRow)
      maxRow = sum;
    double norm = sum > 1 ? 1 / sum : 1;
    if (!(i % 2) && i / 2 >= fixed && norm < 1) {
      float *m = mom + (i / 2 - fixed) * n;
      for (int j = 0; j < n; j++)
        m[j] *= norm;
    }
    if (count <= n / 2) {
      int *columns = (int *)(row + n - count);
      for (int j = 0; j < count; j++)
        row[j] = row[index[j]] * norm;
      for (int j = 0; j < count; j++)
        columns[j] = index[j];
      counts[i] = count;
    } else {
      for (int j = 0; j < n; j++)
        row[j] *= norm;
      counts[i] = -1;
    }
    for (int c = 0; c < 3; c++) {
      incoming[3 * i + c] = 0;
      emitted[3 * i + c] = direct[3 * i + c];
    }
  }
  for (int it = 1; it <= 512; it++) {
    residual = 0;
    for (int i = 0; i < n; i++) {
      double r = 0, g = 0, b = 0;
      float *weights = K + i * n;
      int count = counts[i];
      if (count >= 0) {
        int *columns = (int *)(weights + n - count);
        for (int k = 0; k < count; k++) {
          double w = weights[k];
          double *a = emitted + 3 * columns[k];
          r += w * a[0];
          g += w * a[1];
          b += w * a[2];
        }
      } else {
        for (int k = 0; k < n; k++) {
          double w = weights[k];
          double *a = emitted + 3 * k;
          r += w * a[0];
          g += w * a[1];
          b += w * a[2];
        }
      }
      double *a = incoming + 3 * i;
      double dr = __builtin_fabs(a[0] - r), dg = __builtin_fabs(a[1] - g),
             db = __builtin_fabs(a[2] - b);
      if (dr > residual)
        residual = dr;
      if (dg > residual)
        residual = dg;
      if (db > residual)
        residual = db;
      a[0] = r;
      a[1] = g;
      a[2] = b;
      V rho = p[i / 2].color;
      double *out = emitted + 3 * i, *h = incoming + 3 * i;
      float *d = direct + 3 * i;
      out[0] = d[0] + rho.x * h[0];
      out[1] = d[1] + rho.y * h[1];
      out[2] = d[2] + rho.z * h[2];
    }
    if (residual <= 1e-7) {
      for (int i = fixed; i < n / 2; i++)
        for (int c = 0; c < 3; c++) {
          double sum = 0;
          float *m = mom + (i - fixed) * n;
          for (int j = 0; j < n; j++)
            sum += m[j] * emitted[j * 3 + c];
          float limit = incoming[6 * i + c];
          slopes[3 * i + c] = max(-limit, min(limit, 3 * sum));
        }
      bounds[0] = residual;
      bounds[1] = maxRow;
      return it;
    }
  }
  return 0;
}
