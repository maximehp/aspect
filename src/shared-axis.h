// Integrate a finite receiver line and a parallel source generator together.
// For longitudinal difference z, all connecting segments are translates.
// The diffuse kernel is constant along this translation. A finite sheet cut
// by the common plane gives two endpoints (a,tau); the blocked source interval
// lies between the lines a+tau*z. No receiver position is chosen to find it.
typedef struct {
  double a, b;
} AxisLine;
typedef struct {
  AxisLine l, h;
} AxisBlock;
typedef struct {
  double v[5][5];
} AxisValue; // three physical values, mean, first moment
typedef struct {
  Pair lamp;
  const int *objects;
  int count;
  Vec eye, axis;
  double width;
  unsigned long long *work;
} AxisFamily;
typedef struct {
  AxisFamily *f;
  Vec source;
  double depth, offset, factor, t, lightAx, lightD2, lightHeight;
  AxisBlock *block;
  int blocks, nlit;
  Interval *lit;
} AxisSlice;
static double lineAt(AxisLine l, double z) { return l.a + l.b * z; }
static AxisValue axisAdd(AxisValue a, AxisValue b, double scale) {
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      a.v[i][j] += scale * b.v[i][j];
  return a;
}
static double axisError(AxisValue a, AxisValue b) {
  double e = 0;
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 5; j++)
      e = high(e, absd(a.v[i][j] - b.v[i][j]));
  return e;
}
// Position on the ray sheet is S + e*a + D*tau, 0<tau<1. Intersect this
// plane with the finite blocker and clip the resulting line segment. Its
// endpoints are independent of both axial coordinates.
static int axisBlock(const Shape *s, Vec S, Vec e, Vec D, AxisBlock *out) {
  Vec d = minus(S, s->c);
  double H = dp(d, s->ax[2]), A = dp(e, s->ax[2]), B = dp(D, s->ax[2]);
  double x, t, dx, dt;
  if (absd(A) > absd(B) && absd(A) > 1e-13) {
    x = -H / A;
    t = 0;
    dx = -B / A;
    dt = 1;
  } else if (absd(B) > 1e-13) {
    x = 0;
    t = -H / B;
    dx = 1;
    dt = -A / B;
  } else
    return absd(H) > 1e-10 ? 0 : -1; // coplanar family needs finite-segment predicates
  double lo = -FAR, hi = FAR;
  if (!dt && (t <= 0 || t >= 1))
    return 0;
  if (!clipPositive(t, dt, &lo, &hi) || !clipPositive(1 - t, -dt, &lo, &hi))
    return 0;
  for (int i = 0; i < 2; i++) {
    Vec u = s->ax[i];
    double c = dp(d, u) + x * dp(e, u) + t * dp(D, u), m = dx * dp(e, u) + dt * dp(D, u);
    if (!clipPositive(1 + c, m, &lo, &hi) || !clipPositive(1 - c, -m, &lo, &hi))
      return 0;
  }
  out->l = (AxisLine){x + lo * dx, t + lo * dt};
  out->h = (AxisLine){x + hi * dx, t + hi * dt};
  return 1;
}
static const double axisNodes[] = {-.7745966692414834, 0, .7745966692414834};
// On an event band each visible length endpoint is affine in z. After
// integrating the translation, a bilinear source times a linear receiver
// test leaves a polynomial of degree at most three over (z²+D²)².
// Two nodes constructed from the four rational moments integrate it exactly.
static int kernelNodes(double a, double b, double D, double z[2], double weights[2]) {
  const double h = .5 * (b - a), m = .5 * (b + a), q = D * D + m * m;
  double M[4] = {0};
  if (h * h < .0625 * q) {
    // Use dimensionless centered moments on narrow bands. At |s|=R/(2h)
    // the reciprocal squared kernel is bounded by 16 times its midpoint.
    // The tail of its power series is geometric on the unit interval.
    double A = 2 * m * h / q, B = h * h / q, ratio = 2 * h / ROOT(q), bound = 16 / (1 - ratio);
    double prev = 0, v = 1;
    for (int n = 0; n < 64; n++) {
      for (int j = n & 1; j < 4; j += 2)
        M[j] += 2 * v / (n + j + 1);
      double next = -(A * (n + 2) * v + B * (n + 3) * prev) / (n + 1);
      prev = v;
      v = next;
      bound *= ratio;
      if (bound < 2e-14)
        break;
    }
    for (int j = 0; j < 4; j++)
      M[j] *= h / (q * q);
  } else {
    double f0, f1, f2;
    moments(a, b, D, &f0, &f1, &f2);
    double Qa = a * a + D * D, Qb = b * b + D * D;
    double f3 = asinhValue((b - a) * (b + a) / (2 * ROOT(Qa * Qb))) - D * D * f1;
    M[0] = f0;
    M[1] = (f1 - m * f0) / h;
    M[2] = (f2 - m * (2 * f1 - m * f0)) / (h * h);
    M[3] = (f3 - m * (3 * f2 - m * (3 * f1 - m * f0))) / (h * h * h);
  }
  double u = M[1] / M[0], v = M[2] / M[0], var = v - u * u;
  if (!(var > 0 && M[0] > 0))
    return 0;
  double k = (M[3] / M[0] - 2 * u * v + u * u * u) / var, mid = .5 * (u + k);
  double r = ROOT(.25 * (k - u) * (k - u) + var), l = mid - r, R = mid + r;
  if (!(l > -1 && R < 1 && l < u && R > u))
    return 0;
  z[0] = m + h * l;
  z[1] = m + h * R;
  weights[0] = M[0] * (R - u) / (2 * r);
  weights[1] = M[0] - weights[0];

  return 1;
}
typedef AxisValue (*AxisEval)(void *, double);
typedef struct {
  int l, h, lamp;
} AxisBranch;
// Independent residual observations use the inverse of the same segment
// equations. They do not introduce artificial cuts in the finite receiver integral.
static void physicalSection(AxisSlice *s, AxisValue *out) {
  AxisFamily *f = s->f;
  const Chart *c = f->lamp.c;
  const double v = .5 + .5 * s->t / c->height, K = s->factor / PI;
  for (int i = 0; i < 3; i++) {
    double y = axisNodes[i] * f->width - s->offset;
    Interval dark[2 * s->blocks + 1], open[2 * s->blocks + 2];
    int n = 0;
    for (int j = 0; j < s->blocks; j++) {
      AxisLine l = s->block[j].l, h = s->block[j].h;
      for (int sign = -1; sign <= 1; sign += 2) {
        double a = -c->width, b = c->width;
        if (clipPositive(sign * (l.b * y - l.a), sign * (1 - l.b), &a, &b) &&
            clipPositive(-sign * (h.b * y - h.a), -sign * (1 - h.b), &a, &b))
          dark[n++] = (Interval){a, b, j};
      }
    }
    int no = openings(dark, n, -c->width, c->width, open);
    for (int j = 0; j < no; j++) {
      double a = open[j].lo, b = open[j].hi, m0, m1, m2;
      moments(a - y, b - y, s->depth, &m0, &m1, &m2);
      double u = .5 * (m0 + (m1 + y * m0) / c->width);
      for (int k = 0; k < 4; k++)
        out->v[i][k] += K * (k & 2 ? v : 1 - v) * (k & 1 ? u : m0 - u);
      for (int j = 0; j < s->nlit; j++) {
        double l = high(a, s->lit[j].lo), h = low(b, s->lit[j].hi);
        if (h > l)
          out->v[i][4] += K * s->lightHeight *
                          directPrimitive(s->lightAx, y, s->lightD2, s->depth * s->depth, 1, 0, l,
                                          h, 1e-11, f->lamp.stats);
      }
    }
  }
}
static int axisBranches(const AxisLine *lines, int blocks, int nlit, double z, AxisBranch *out) {
  double values[4 + 2 * (blocks + nlit)];
  for (int i = 0; i < 4 + 2 * (blocks + nlit); i++)
    values[i] = lineAt(lines[i], z);
  int left = values[0] > values[2] ? 0 : 2, end = values[1] < values[3] ? 1 : 3;
  if (values[left] >= values[end])
    return 0;
  AxisBranch dark[blocks + 1];
  int n = 0;
  for (int i = 0; i < blocks; i++) {
    int l = 4 + 2 * i, h = l + 1;
    if (values[l] > values[h]) {
      int t = l;
      l = h;
      h = t;
    }
    if (values[h] <= values[left] || values[l] >= values[end])
      continue;
    int j = n++;
    while (j && values[dark[j - 1].l] > values[l]) {
      dark[j] = dark[j - 1];
      j--;
    }
    dark[j] = (AxisBranch){l, h};
  }
  int no = 0;
  for (int j = 0; j <= n; j++) {
    int right = j < n ? dark[j].l : end;
    if (values[right] > values[left]) {
      out[no++] = (AxisBranch){left, right, 0};
      for (int i = 0; i < nlit; i++) {
        int l = 4 + 2 * (blocks + i), h = l + 1;
        if (values[l] < values[left])
          l = left;
        if (values[h] > values[right])
          h = right;
        if (values[h] <= values[l])
          continue;
        out[no++] = (AxisBranch){l, h, 1};
      }
    }
    if (j < n && values[dark[j].h] > values[left])
      left = dark[j].h;
    if (values[left] >= values[end])
      break;
  }
  return no;
}
typedef struct {
  double mean, first;
} LampEdge;
// Moments of the point-lamp primitive restricted to an affine endpoint.
static LampEdge axisLampEdge(const AxisSlice *s, AxisLine edge, double a, double b,
                             const double nodes[2], const double weights[2]) {
  const double L = s->lightD2, l = s->lightAx;
  if (!edge.b) {
    const double x = edge.a - l, inv = 1 / ROOT(x * x + L), p = x * inv / L;
    return (LampEdge){p * (weights[0] + weights[1]),
                      weights[0] * ((l + s->offset - nodes[0]) * p - inv) +
                          weights[1] * ((l + s->offset - nodes[1]) * p - inv)};
  }
  double mid = (a + b) * .5, h = (b - a) * .5, x = lineAt(edge, mid) - l, step = edge.b * h;
  double A = x * x + L, B = mid * mid + s->depth * s->depth;
  double ratio = high(absd(step) / ROOT(A), h / ROOT(B));
  if (ratio > .3) {
    double z[2] = {0}, w[2] = {0}; // Nonconstant endpoints use only their power series.
    LampEdge left = axisLampEdge(s, edge, a, mid, z, w),
             right = axisLampEdge(s, edge, mid, b, z, w);
    return (LampEdge){left.mean + right.mean, left.first + right.first};
  }
  const double u = 2 * x * step / A, v = step * step / A, c = 2 * mid * h / B, d = h * h / B;
  double den[] = {1, u + c, v + d + u * c, u * d + v * c, v * d};
  double num[] = {-.5 * u - 2 * c, -v - 4 * d - 2.5 * u * c, -4.5 * u * d - 3 * v * c, -5 * v * d};
  double f[4] = {1, 0, 0, 0}, M[3] = {0}, tail = 32, scale = 2 * ratio;
  // On |s|=1/(2*ratio) each complex distance factor is >=1/2.
  // The normalized integrand there has magnitude <=32.
  for (int n = 0;; n++) {
    for (int j = n & 1; j < 3; j += 2)
      M[j] += 2 * f[n & 3] / (n + j + 1);
    tail *= scale;
    if (tail / (1 - scale) < 2e-14)
      break;
    double next = 0;
    for (int k = 0; k < 4 && k <= n; k++)
      next += (num[k] - den[k + 1] * (n - k)) * f[(n - k) & 3];
    f[(n + 1) & 3] = next / (n + 1);
  }
  const double k = h / (ROOT(A) * B * B), p = (x * M[0] + step * M[1]) / L;
  return (LampEdge){k * p,
                    k * ((l + s->offset - mid) * p - h * (x * M[1] + step * M[2]) / L - M[0])};
}
static int integrateBand(AxisSlice *s, const AxisLine *lines, const AxisBranch *branches, int n,
                         double a, double b, AxisValue *out) {
  if (!n)
    return 1;
  double z[2], W[2];
  if (!kernelNodes(a, b, s->depth, z, W))
    return 0;
  AxisFamily *f = s->f;
  const Chart *c = f->lamp.c;
  double w = f->width, v = .5 + .5 * s->t / c->height, K = s->factor / PI;
  for (int j = 0; j < n; j++) {
    AxisBranch branch = branches[j];
    AxisLine l = lines[branch.l], r = lines[branch.h];
    if (branch.lamp) {
      LampEdge left = axisLampEdge(s, l, a, b, z, W), right = axisLampEdge(s, r, a, b, z, W);
      double k = K * s->lightHeight / (2 * w);
      out->v[3][4] += k * (right.mean - left.mean);
      out->v[4][4] += k * (right.first - left.first) / w;
    } else
      for (int h = 0; h < 2; h++) {
        double low = lineAt(l, z[h]), high = lineAt(r, z[h]), len = high - low,
               mid = (high + low) * .5, o = z[h] - s->offset;
        double mean = .5 + .5 * mid / c->width, y = (mid - o) / w,
               mixed = mean * y + len * len / (24 * c->width * w);
        for (int k = 0; k < 4; k++) {
          double scale = K * W[h] * (k & 2 ? v : 1 - v), p = k & 1 ? mean : 1 - mean;
          out->v[3][k] += scale * len * p / (2 * w);
          out->v[4][k] += scale * len * (k & 1 ? mixed : y - mixed) / (2 * w);
        }
      }
  }
  f->work[14]++;
  return 1;
}

static AxisValue axisGauss(AxisEval eval, void *ctx, double a, double b) {
  const double m = (a + b) * .5, h = (b - a) * .5;
  AxisValue value = {0};
  value = axisAdd(value, eval(ctx, m), h * 8 / 9);
  value = axisAdd(value, eval(ctx, m + h * axisNodes[0]), h * 5 / 9);
  return axisAdd(value, eval(ctx, m + h * axisNodes[2]), h * 5 / 9);
}
static AxisValue axisIntegrate(AxisEval eval, void *ctx, double a, double b, AxisValue coarse,
                               double tol, int depth, int *ok) {
  double m = (a + b) * .5;
  AxisValue l = axisGauss(eval, ctx, a, m), r = axisGauss(eval, ctx, m, b), fine = axisAdd(l, r, 1);
  if (axisError(coarse, fine) <= tol)
    return fine;
  if (!depth) {
    *ok = 0;
    return fine;
  }
  l = axisIntegrate(eval, ctx, a, m, l, tol * .5, depth - 1, ok);
  return axisAdd(l, axisIntegrate(eval, ctx, m, b, r, tol * .5, depth - 1, ok), 1);
}
typedef struct {
  AxisFamily *f;
  int ok;
} AxisIntegral;
static AxisValue axisTransverse(void *ctx, double t) {
  AxisIntegral *integral = ctx;
  AxisFamily *f = integral->f;
  const Chart *c = f->lamp.c;
  AxisValue out = {0};
  Vec S = plus(c->c, scale3(c->v, t)), dy = minus(f->eye, S),
      D = minus(dy, scale3(f->axis, dp(dy, f->axis)));
  double A = dp(dy, c->n), B = -dp(dy, f->lamp.y.n), depth = length(D), w = f->width;
  if (A <= 0 || B <= 0)
    return out;
  if (depth < 1e-8) {
    integral->ok = 0;
    return out;
  }
  AxisBlock blocks[f->count + 1];
  Interval lit[f->lamp.count[1] + 1];
  Vec dl = minus(f->lamp.lamp, S);
  AxisSlice s = {.f = f,
                 .source = S,
                 .depth = depth,
                 .offset = -dp(dy, f->axis),
                 .factor = A * B,
                 .t = t,
                 .block = blocks,
                 .lit = lit,
                 .lightAx = dp(dl, f->axis),
                 .lightHeight = dp(dl, c->n),
                 .lightD2 = dp(cross3(dl, f->axis), cross3(dl, f->axis))};
  if (!f->lamp.lampDark && s.lightHeight > 0)
    s.nlit = litGenerator(&f->lamp, t, S, lit);
  for (int j = 0; j < f->count; j++) {
    int used = axisBlock(f->lamp.s + f->objects[j], S, f->axis, D, blocks + s.blocks);
    if (used < 0) {
      integral->ok = 0;
      return out;
    }
    s.blocks += used;
  }
  physicalSection(&s, &out);
  // Only a change of visible endpoints can split the weak integral.
  int count = 0;
  AxisLine lines[4 + 2 * (s.blocks + s.nlit)];
  lines[count++] = (AxisLine){-c->width, 0};
  lines[count++] = (AxisLine){c->width, 0};
  lines[count++] = (AxisLine){-s.offset - w, 1};
  lines[count++] = (AxisLine){-s.offset + w, 1};
  for (int j = 0; j < s.blocks; j++) {
    lines[count++] = blocks[j].l;
    lines[count++] = blocks[j].h;
  }
  for (int j = 0; j < s.nlit; j++) {
    lines[count++] = (AxisLine){lit[j].lo, 0};
    lines[count++] = (AxisLine){lit[j].hi, 0};
  }
  double lo = s.offset - c->width - w, hi = s.offset + c->width + w;
  int n = 0;
  double cuts[2 + count * (count - 1) / 2];
  insert(cuts, &n, 0, lo, hi);
  for (int i = 0; i < count; i++)
    for (int j = 0; j < i; j++)
      if (lines[i].b != lines[j].b)
        insert(cuts, &n, (lines[j].a - lines[i].a) / (lines[i].b - lines[j].b), lo, hi);
  cuts[n++] = hi;
  // Every surviving opening is disjoint. Intersecting it with nlit disjoint
  // lamp intervals can introduce at most two boundaries per interval.
  const int capacity = 2 * (s.blocks + 1) + 2 * s.nlit + 2;
  AxisBranch Aset[capacity], Bset[capacity], *previous = Aset, *next = Bset;
  int last = -1;
  double start = lo;
  for (int i = 0; i < n; i++) {
    double b = cuts[i];
    int used = axisBranches(lines, s.blocks, s.nlit, .5 * (lo + b), next), same = used == last;
    for (int j = 0; same && j < used; j++)
      same = next[j].l == previous[j].l && next[j].h == previous[j].h &&
             next[j].lamp == previous[j].lamp;
    if (!same) {
      if (last >= 0 && !integrateBand(&s, lines, previous, last, start, lo, &out)) {
        integral->ok = 0;
        break;
      }
      AxisBranch *swap = next;
      next = previous;
      previous = swap;
      last = used;
      start = lo;
    }
    lo = b;
  }
  if (last >= 0 && !integrateBand(&s, lines, previous, last, start, lo, &out))
    integral->ok = 0;
  return out;
}
// Bind only the transverse floor coordinate. The coefficients delivered to
// equilibrium are the integral against 1 and r/width. Separately preserve the
// physical three point values for the spatial residual, including narrow light.
static int jointAxis(Pair *w, const Domain *g, const int *ids, int n, Vec center, double *result) {
  const Chart *c = w->c;
  AxisFamily family = {.lamp = *w,
                       .objects = ids,
                       .count = n,
                       .eye = center,
                       .axis = c->ax,
                       .width = length(g->a),
                       .work = w->reuse};
  AxisIntegral integral = {&family, 1};
  int cap = 32 + 32 * (n + w->count[1]);
  double a = -c->height, hi = c->height, cuts[cap + 1];
  Pair events = *w;
  events.count[0] = 0;
  events.warpCenter = 0;
  int nc = planarEvents(&events, cuts, cap, a, hi);
  // In the transverse cross-section the family pivots about the receiver
  // line. A blocker corner crossing that plane changes the sheet intersection.
  Vec eye = minus(center, c->c), N = cross3(c->ax, eye), N1 = cross3(c->ax, c->v);
  for (int j = 0; j < n; j++) {
    const Shape *s = w->s + ids[j];
    Vec U = cross3(s->ax[1], s->ax[2]), V = cross3(s->ax[2], s->ax[0]);
    U = scale3(U, 1 / dp(U, s->ax[0]));
    V = scale3(V, 1 / dp(V, s->ax[1]));
    for (int k = 0; k < 4; k++) {
      Vec d = plus(minus(s->c, center), plus(scale3(U, k & 1 ? 1 : -1), scale3(V, k & 2 ? 1 : -1)));
      double den = dp(N1, d);
      if (den && nc < cap)
        insert(cuts, &nc, dp(N, d) / den, a, hi);
    }
  }
  if (nc == cap)
    return 0;
  cuts[nc++] = hi;
  AxisValue total = {0};
  for (int j = 0; j < nc && integral.ok; j++) {
    double b = cuts[j], tol = .5 * w->tolerance * (b - a) / (2 * c->height);
    AxisValue coarse = axisGauss(axisTransverse, &integral, a, b);
    total = axisAdd(
        total, axisIntegrate(axisTransverse, &integral, a, b, coarse, tol, 8, &integral.ok), 1);
    a = b;
  }
  if (!integral.ok)
    return 0;
  int forward = dp(g->a, c->ax) > 0;
  for (int j = 0; j < 5; j++) {
    // A quadrature-equivalent signed correction is used only in the weak
    // receiver projection. Never apply it to physical point observations.
    double L = total.v[0][j], M = total.v[1][j], R = total.v[2][j];
    double d0 = total.v[3][j] - (5 * (L + R) + 8 * M) / 18;
    double d1 = total.v[4][j] - 5 * axisNodes[2] * (R - L) / 18;
    for (int p = 0; p < 3; p++) {
      int row = forward ? p : 2 - p;
      result[10 * row + j] = total.v[p][j];
      result[10 * row + 5 + j] = d0 + 3 * axisNodes[p] * d1;
    }
  }
  return 1;
}
