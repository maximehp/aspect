// Viewed along each box axis, the transverse body is a rounded rectangle.
// Its two support lines through the eye touch the cylindrical parts along
// finite straight generators. One plane per generator describes its shadow.
static int outlines(CylinderEdge *edges, const Shape *s, Vec O) {
  int count = 0;
  const double *o = (double *)&O, *h = (const double *)&s->h;
  double r = s->r;
  for (int i = 0; i < 3; i++) {
    int j = (i + 1) % 3, k = (i + 2) % 3, first = count;
    for (int sx = -1; sx <= 1; sx += 2)
      for (int sy = -1; sy <= 1; sy += 2) {
        double a = o[j] - sx * (h[j] - r), b = o[k] - sy * (h[k] - r), rr = a * a + b * b,
               disc = rr - r * r;
        if (absd(disc) < 1e-13 * (rr + r * r))
          return -1;
        if (disc < 0)
          continue;
        double m = ROOT(disc);
        for (int sign = -1; sign <= 1; sign += 2) {
          double nx = (r * a - sign * m * b) / rr, ny = (r * b + sign * m * a) / rr;
          if (sx * nx < -1e-13 || sy * ny < -1e-13)
            continue;
          if (absd(nx) < 1e-13 || absd(ny) < 1e-13 || count - first == 2)
            return -1;
          edges[count++] = (CylinderEdge){i, nx, ny, ny * a - nx * b};
        }
      }
  }
  return count;
}
// Each silhouette generator's tangent plane supports the *whole* rounded
// body. If its finite generator is reached before the source endpoint, the
// plane also witnesses an extreme of the exact blocked interval.
static int tangentInterval(const Cone *cone, Vec O, Vec D, Vec U, const double *h, double *lo,
                           double *hi, int *left, int *right) {
  if (cone->generators <= 0)
    return -1;
  double a = *lo, b = *hi;
  int at[2] = {-1, -1};
  const double *d = (double *)&D, *u = (double *)&U, *o = (double *)&O;
  for (int k = 0; k < cone->generators; k++) {
    CylinderEdge e = cone->edges[k];
    int i = (e.axis + 1) % 3, j = (e.axis + 2) % 3;
    double c = e.x * d[i] + e.y * d[j], m = e.x * u[i] + e.y * u[j];
    double va = c + a * m, vb = c + b * m;
    double eps =
        1e-11 * (1 + absd(d[i]) + absd(d[j]) + high(absd(a), absd(b)) * (absd(u[i]) + absd(u[j])));
    if (low(va, vb) > eps)
      return 0;
    if (high(va, vb) < -eps)
      continue;
    if (absd(m) < 1e-12 || absd(va) <= eps || absd(vb) <= eps)
      return -1;
    double x = -c / m;
    if (m < 0 && x > a) {
      a = x;
      at[0] = k;
    }
    if (m > 0 && x < b) {
      b = x;
      at[1] = k;
    }
  }
  if (b <= a)
    return -1;
  int *sides[2] = {left, right};
  for (int side = 0; side < 2; side++)
    if (at[side] >= 0) {
      CylinderEdge e = cone->edges[at[side]];
      int i = e.axis, j = (i + 1) % 3, k = (i + 2) % 3;
      double x = side ? b : a, ray = e.x * (d[k] + x * u[k]) - e.y * (d[j] + x * u[j]);
      double t = e.along / ray, loc = o[i] + t * (d[i] + x * u[i]);
      if (!ray || absd(t - 1) < 1e-11 || absd(absd(loc) - h[i]) < 1e-10)
        return -1;
      *sides[side] = t > 1e-10 && t < 1 && absd(loc) < h[i];
    }
  *lo = a;
  *hi = b;
  return 1;
}
