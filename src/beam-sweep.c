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
  float lo, hi, a, b, c, cosine, mx, my, mc, left, right;
} Project;
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
// One record per observed unordered pair of surface sides. Store measures in
// projected square metres. Dividing by the receiving area gives each direction.
typedef struct {
  int a, b;
  float area, firstA, firstB, mixed;
} Link;
typedef struct {
  int used, capacity, failed;
  Link *links;
  int *slots;
} Operator;
static unsigned hashPair(int a, int b) {
  unsigned h = (unsigned)a * 0x9e3779b1u + (unsigned)b * 0x85ebca77u;
  h = (h ^ (h >> 16)) * 0x7feb352du;
  return h ^ (h >> 15);
}
void openOperator(Operator *op, Link *links, int *slots, int capacity, int used) {
  *op = (Operator){used, capacity, 0, links, slots};
  for (int i = 0; i < capacity * 2; i++)
    slots[i] = 0;
  for (int i = 0; i < used; i++) {
    Link l = links[i];
    unsigned index = hashPair(l.a, l.b) & (capacity * 2 - 1);
    while (slots[index])
      index = (index + 1) & (capacity * 2 - 1);
    slots[index] = i + 1;
  }
}
static Link *findLink(Operator *op, int a, int b) {
  unsigned mask = 2 * op->capacity - 1, index = hashPair(a, b) & mask;
  while (op->slots[index]) {
    Link *e = op->links + op->slots[index] - 1;
    if (e->a == a && e->b == b)
      return e;
    index = (index + 1) & mask;
  }
  if (op->used == op->capacity) {
    op->failed = 1;
    return 0;
  }
  Link *l = op->links + op->used;
  *l = (Link){a, b, 0, 0, 0, 0};
  op->slots[index] = ++op->used;
  return l;
}
static float coordinate(Project q, float x, float y) {
  return max(-1, min(1, q.mx * x + q.my * y + q.mc));
}
static void exchange(Poly *p, Project *q, int a, int b, float area, Operator *op, float x, float y,
                     float dx, float dy) {
  if (p[a].owner == p[b].owner)
    return;
  int sa = q[a].cosine < 0, sb = q[b].cosine > 0;
  if ((p[a].solid && sa) || (p[b].solid && sb))
    return;
  int i = 2 * a + sa, j = 2 * b + sb;
  float u = coordinate(q[a], x, y), v = coordinate(q[b], x, y);
  if (i > j) {
    int t = i;
    i = j;
    j = t;
    float f = u;
    u = v;
    v = f;
  }
  Link *l = findLink(op, i, j);
  if (!l)
    return;
  // The product of two affine functions also has a within-footprint covariance.
  float mixed = u * v + (q[a].mx * q[b].mx * dx * dx + q[a].my * q[b].my * dy * dy) / 12.f;
  mixed = max(-1 + __builtin_fabsf(u + v), min(1 - __builtin_fabsf(u - v), mixed));
  l->area += area;
  l->firstA += area * u;
  l->firstB += area * v;
  l->mixed += area * mixed;
}
void mergeLinks(Operator *op, const Link *src, int count) {
  for (int k = 0; k < count; k++) {
    Link e = src[k], *l = findLink(op, e.a, e.b);
    if (!l)
      return;
    l->area += e.area;
    l->firstA += e.firstA;
    l->firstB += e.firstB;
    l->mixed += e.mixed;
  }
}
// Certified parallel sheet pairs supply exact area; their moments come from
// integrating the same kernel on subintervals of the two basis coordinates.
int installExact(Operator *op, int a, int b, float area, float firstA, float firstB, float mixed) {
  if (a > b) {
    int t = a;
    a = b;
    b = t;
    float f = firstA;
    firstA = firstB;
    firstB = f;
  }
  Link *l = findLink(op, a, b);
  if (l)
    *l = (Link){a, b, area, firstA, firstB, mixed};
  return l ? l - op->links : -1;
}
__attribute__((visibility("default"))) void sweep(Poly *p, int count, V *verts, int nv, V *frame,
                                                  int dirs, int bins, Operator *op, int correction,
                                                  char *scratch, int firstDir, int endDir) {
  float *xs = (float *)scratch, *ys = xs + nv;
  Project *q = (Project *)(ys + nv);
  Edge *edges = (Edge *)(q + count);
  int *hits = (int *)(edges + 2 * count), *active = hits + count;
  Edge *starts = (Edge *)(active + count);
  Edge *cuts = starts + count;
  if (!count)
    return;
  for (int d = firstDir; d < endDir; d++) {
    V u = frame[3 * d], v = frame[3 * d + 1], w = frame[3 * d + 2];
    for (int i = 0; i < nv; i++) {
      xs[i] = dot(verts[i], u);
      ys[i] = dot(verts[i], v);
    }
    float bottom = 1e20f, top = -1e20f, staticLo = 1e20f, staticHi = -1e20f, moveLo = 1e20f,
          moveHi = -1e20f;
    for (int i = 0; i < count; i++) {
      Poly f = p[i];
      float lo = 1e20f, hi = -1e20f, left=1e20f, right=-1e20f;
      for (int j = f.start; j < f.start + f.count; j++) {
        lo = min(lo, ys[j]);
        hi = max(hi, ys[j]);
        left=min(left,xs[j]);right=max(right,xs[j]);
      }
      float g = dot(f.n, w);
      q[i] = (Project){lo, hi, -dot(f.n, u) / g, -dot(f.n, v) / g, dot(f.n, f.center) / g, g};
      {
        q[i].left=left;q[i].right=right;
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
    int nstart = count;
    for (int i = 0; i < count; i++)
      starts[i] = (Edge){q[i].lo, i};
    if (!nstart)
      continue;
    sort(starts, 0, nstart - 1);
    int begin = 0, na = 0;
    for (int row = first; row < end; row++) {
      float y0 = staticLo + row * step, y1 = y0 + step;
      while (begin < nstart && starts[begin].x < y1)
        active[na++] = starts[begin++].encoded;
      // Integrate thin projected polygons between their vertex heights. Use
      // each sub-strip's ordinate for every surface, including its blockers.
      int kept=0;
      float left=1e20f, right=-1e20f, thinLeft=1e20f, thinRight=-1e20f;
      for(int k=0;k<na;k++){
        int i=active[k];
        if(q[i].hi<=y0 || __builtin_fabsf(q[i].cosine)<1e-7f)continue;
        active[kept++]=i;
        if(!correction || p[i].solid){left=min(left,q[i].left);right=max(right,q[i].right);}
        if(q[i].hi-q[i].lo<step){thinLeft=min(thinLeft,q[i].left);thinRight=max(thinRight,q[i].right);}
      }
      na=kept;
      // Restrict extra ordinates to columns containing the narrow feature.
      // Fixed world-projection column boundaries are shared by both sweeps.
      const float span=8*step;
      thinLeft=__builtin_floorf(thinLeft/span)*span;
      thinRight=__builtin_ceilf(thinRight/span)*span;
      for(float x0=left;x0<right;){
        float x1=right;
        int refined=x0>=thinLeft && x0<thinRight;
        if(x0<thinLeft) x1=min(right,thinLeft);
        else if(refined) {
          float next=(__builtin_floorf(x0/span)+1)*span;
          if(next<=x0) next+=span;
          x1=min(right,next);
        }
        float clipLeft=x0,clipRight=x1;x0=x1;
        int nc=1;cuts[0]=(Edge){y0,0};
        if(refined)for(int k=0;k<na;k++){
          int i=active[k];
          if(q[i].right<=clipLeft || q[i].left>=clipRight || q[i].hi-q[i].lo>=step)continue;
          for(int j=p[i].start;j<p[i].start+p[i].count;j++)
            if(ys[j]>y0 && ys[j]<y1)cuts[nc++]=(Edge){ys[j],0};
        }
      if (nc > 1) sort(cuts, 1, nc - 1);
      cuts[nc++] = (Edge){y1,0};
      for (int c = 1; c < nc; c++) {
        const float low = cuts[c-1].x, high = cuts[c].x;
        if (high <= low) continue;
        const float y = (low + high) * .5f, dy = high - low;
        int ne = 0;
        for (int k = 0; k < na; k++) {
          int i = active[k];
          if (y < q[i].lo || y >= q[i].hi || q[i].right<=clipLeft || q[i].left>=clipRight) continue;
          float L = 1e20f, R = -1e20f;
        int start = p[i].start, end = start + p[i].count;
        for (int j = start, k = end - 1; j < end; k = j++) {
          if ((ys[k] > y) != (ys[j] > y)) {
            float x = xs[k] + (xs[j] - xs[k]) * ((y - ys[k]) / (ys[j] - ys[k]));
            L = min(L, x);
            R = max(R, x);
          }
        }
        L=max(L,clipLeft);R=min(R,clipRight);
        if (L >= R)
          continue;
        edges[ne++] = (Edge){L, i + 1};
        edges[ne++] = (Edge){R, -i - 1};
      }
      if (!ne)
        continue;
      if (correction) {
        float left = 1e20f, right = -1e20f;
        for (int i = 0; i < ne; i++) {
          int id = edges[i].encoded;
          int j = id > 0 ? id - 1 : -id - 1;
          if (p[j].solid) {
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
          for (int i = 1; i < nh; i++) {
            int s = hits[i], j = i;
            float z = q[s].a * mid + q[s].b * y + q[s].c;
            while (j > 0) {
              int t = hits[j - 1];
              if (q[t].a * mid + q[t].b * y + q[t].c <= z)
                break;
              hits[j] = t;
              j--;
            }
            hits[j] = s;
          }
          int previous = -1, stationary = -1;
          for (int i = 0; i < nh; i++) {
            int id = hits[i];
            float area = (x - prev) * dy * (2.f / dirs);
            if (previous >= 0 && (!correction || p[id].solid || p[previous].solid))
              exchange(p, q, previous, id, area, op, mid, y, x - prev, dy);
            if (correction && !p[id].solid) {
              if (stationary >= 0 && stationary != previous)
                exchange(p, q, stationary, id, -area, op, mid, y, x - prev, dy);
              stationary = id;
            }
            previous = id;
          }
        }
        int s = edges[e].encoded;
        if (s > 0) {
          hits[nh++] = s - 1;
          moving += p[s - 1].solid;
        } else {
          moving -= p[-s - 1].solid;
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
  }
}

// Convert the accumulated reciprocal measures to contiguous rows for iteration.
// Protected analytic couplings keep their mass; clamp only unresolved exchange.
typedef struct {
  int source;
  float mean, sourceMoment, receiverMoment, mixed;
} Weight;
static void put(Weight *rows, int *next, int i, int j, float area, float u, float v, float uv,
                float norm) {
  rows[next[i]++] = (Weight){j, area * norm, v * norm, u * norm, uv * norm};
}
void prepareRows(Poly *p, int n, Operator *op, int *offset, int *next, Weight *rows, int *exact,
                 float *mass, float *raw, double *bounds) {
  for (int i = 0; i <= n; i++)
    offset[i] = 0;
  for (int k = 0; k < op->used; k++) {
    Link l = op->links[k];
    if (l.area <= 0)
      continue;
    offset[l.a + 1]++;
    offset[l.b + 1]++;
    if (!exact[k]) {
      raw[l.a] += l.area / p[l.a / 2].area;
      raw[l.b] += l.area / p[l.b / 2].area;
    }
  }
  double maxRow = 0;
  for (int i = 0; i < n; i++) {
    offset[i + 1] += offset[i];
    next[i] = offset[i];
    maxRow = max(maxRow, raw[i] + mass[i]);
    raw[i] = raw[i] > 0 ? min(1, max(0, 1 - mass[i]) / raw[i]) : 1;
  }
  for (int k = 0; k < op->used; k++) {
    Link l = op->links[k];
    if (l.area <= 0)
      continue;
    float a = (exact[k] ? 1 : raw[l.a]) / p[l.a / 2].area,
          b = (exact[k] ? 1 : raw[l.b]) / p[l.b / 2].area;
    // Maintain nonnegative exchange for the four endpoint basis functions,
    // including tiny links left by subtracting a blocked stationary footprint.
    l.firstA = max(-l.area, min(l.area, l.firstA));
    l.firstB = max(-l.area, min(l.area, l.firstB));
    l.mixed = max(-l.area + __builtin_fabsf(l.firstA + l.firstB),
                  min(l.area - __builtin_fabsf(l.firstA - l.firstB), l.mixed));
    put(rows, next, l.a, l.b, l.area, l.firstA, l.firstB, l.mixed, a);
    put(rows, next, l.b, l.a, l.area, l.firstB, l.firstA, l.mixed, b);
  }
  bounds[1] = maxRow;
}
int solveLinks(Poly *p, int n, int *offset, Weight *rows, float *direct, float *directSlope,
               double *in, double *out, double *slope, double *outSlope, double *bounds) {
  for (int k = 0; k < n * 3; k++) {
    in[k] = slope[k] = 0;
    out[k] = direct[k];
    outSlope[k] = directSlope[k];
  }
  double omega = 1.25, previousChange = 1e30;
  for (int it = 1; it <= 512; it++) {
    double change = 0;
    for (int i = 0; i < n; i++) {
      double means[3] = {0}, moments[3] = {0};
      for (int k = offset[i]; k < offset[i + 1]; k++) {
        Weight w = rows[k];
        int j = 3 * w.source;
        for (int c = 0; c < 3; c++) {
          means[c] += w.mean * out[j + c] + w.sourceMoment * outSlope[j + c];
          moments[c] += w.receiverMoment * out[j + c] + w.mixed * outSlope[j + c];
        }
      }
      V color = p[i / 2].color;
      float rho[3] = {color.x, color.y, color.z};
      for (int c = 0; c < 3; c++) {
        int k = 3 * i + c;
        double a = means[c] > 0 ? means[c] : 0, b = max(-a, min(a, 3 * moments[c]));
        double da = __builtin_fabs(in[k] - a), db = __builtin_fabs(slope[k] - b);
        if (da > change)
          change = da;
        if (db > change)
          change = db;
        a = in[k] + omega * (a - in[k]);
        if (a < 0)
          a = 0;
        b = max(-a, min(a, slope[k] + omega * (b - slope[k])));
        in[k] = a;
        slope[k] = b;
        out[k] = direct[k] + rho[c] * a;
        outSlope[k] = directSlope[k] + rho[c] * b;
      }
    }
    if (it > 3 && change > 1.05 * previousChange)
      omega = 1;
    previousChange = change;
    if (change <= 1e-7) {
      bounds[0] = change;
      return it;
    }
  }
  return 0;
}
