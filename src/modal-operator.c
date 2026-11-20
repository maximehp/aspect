// Compile W K W^-1 by receiver blocks. A source detail has vanishing zeroth
// and first moments; affine incident fields have zero receiving detail. These
// cancellations make distant modal interactions compressible in both variables.
typedef struct {
  int i, l, r, update;
  double a, b;
} Lift;
typedef struct {
  int col;
  float weight;
} Link;
static double mag(double x) { return __builtin_fabs(x); }
static int bin(double x) {
  union {
    double x;
    unsigned long long bits;
  } u = {x};
  int e = 1023 - (int)((u.bits >> 52) & 2047);
  return e < 0 ? 0 : e > 63 ? 63 : e;
}
int compileModes(double *block, int first, int nr, int n, const Lift *steps,
                 int begin, int end, int total, const double *amplitude,
                 double budget, Link *out, int *rows, int offset,
                 double *maxDiscard) {
  for (int k = begin; k < end; k++) {
    Lift s = steps[k];
    double *d = block + (s.i - first) * n, *l = block + (s.l - first) * n,
           *r = block + (s.r - first) * n;
    if (s.update) {
      for (int j = 0; j < n; j++) {
        double x = d[j];
        l[j] += s.a * x;
        r[j] += s.b * x;
      }
    } else {
      for (int j = 0; j < n; j++)
        d[j] -= s.a * l[j] + s.b * r[j];
    }
  }
  int used = offset;
  for (int i = 0; i < nr; i++) {
    double *v = block + i * n;
    for (int k = 0; k < total; k++) {
      Lift s = steps[k];
      if (s.update)
        v[s.i] -= s.a * v[s.l] + s.b * v[s.r];
      else {
        double d = v[s.i];
        v[s.l] += s.a * d;
        v[s.r] += s.b * d;
      }
    }
    double bins[64] = {0}, sum = 0;
    for (int j = 0; j < n; j++)
      if (v[j]) {
        double bound = mag(v[j]) * amplitude[j];
        bins[bin(bound)] += bound;
      }
    int cut = 63;
    while (cut >= 0 && sum + bins[cut] <= budget)
      sum += bins[cut--];
    for (int j = 0; j < n; j++)
      if (v[j] && bin(mag(v[j]) * amplitude[j]) <= cut) {
        float w = (float)v[j];
        out[used++] = (Link){j, w};
        sum += mag(v[j] - (double)w) * amplitude[j];
      }
    if (sum > *maxDiscard)
      *maxDiscard = sum;
    rows[first + i + 1] = used;
  }
  return used;
}
int restrictModes(int n, const int *rows, const Link *links,
                  const unsigned char *active, int *outRows, Link *out) {
  int used = 0;
  outRows[0] = 0;
  for (int i = 0; i < n; i++) {
    if (active[i])
      for (int j = rows[i]; j < rows[i + 1]; j++)
        if (active[links[j].col])
          out[used++] = links[j];
    outRows[i + 1] = used;
  }
  return used;
}
int applyModes(int n, const int *rows, const Link *links, const double *source,
               double *result) {
  int reads = 0;
  for (int i = 0; i < n; i++) {
    double a = 0, b = 0, c = 0;
    for (int j = rows[i]; j < rows[i + 1]; j++) {
      int k = 3 * links[j].col;
      double w = links[j].weight;
      a += w * source[k];
      b += w * source[k + 1];
      c += w * source[k + 2];
    }
    reads += rows[i + 1] - rows[i];
    result[3 * i] = a;
    result[3 * i + 1] = b;
    result[3 * i + 2] = c;
  }
  return reads;
}
void *memset(void *dst, int value, unsigned long size) {
  volatile unsigned char *p = dst;
  for (unsigned long i = 0; i < size; i++)
    p[i] = (unsigned char)value;
  return dst;
}
