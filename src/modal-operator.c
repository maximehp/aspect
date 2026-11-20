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
// A source chart's lifting steps never use coefficients from another chart.
// An identically empty chart pair stays zero under both basis transforms.
typedef struct {
  int first, length, begin, end, offset;
} SourceGroup;
int compileModes(double *block, int first, int nr, const Lift *steps, int begin, int end,
                 const SourceGroup *groups, int count, const double *amplitude, double budget,
                 Link *out, int *rows, int offset, double *maxDiscard) {
  for (int g = 0; g < count; g++) {
    SourceGroup src = groups[g];
    if (src.offset < 0)
      continue;
    double *data = block + src.offset;
    for (int k = begin; k < end; k++) {
      Lift s = steps[k];
      double *d = data + (s.i - first) * src.length;
      double *l = data + (s.l - first) * src.length;
      double *r = data + (s.r - first) * src.length;
      if (s.update) {
        for (int j = 0; j < src.length; j++) {
          l[j] += s.a * d[j];
          r[j] += s.b * d[j];
        }
      } else {
        for (int j = 0; j < src.length; j++)
          d[j] -= s.a * l[j] + s.b * r[j];
      }
    }
    for (int i = 0; i < nr; i++) {
      double *v = data + i * src.length;
      for (int k = src.begin; k < src.end; k++) {
        Lift s = steps[k];
        int d = s.i - src.first, l = s.l - src.first, r = s.r - src.first;
        if (s.update)
          v[d] -= s.a * v[l] + s.b * v[r];
        else {
          v[l] += s.a * v[d];
          v[r] += s.b * v[d];
        }
      }
    }
  }
  int used = offset;
  for (int i = 0; i < nr; i++) {
    double bins[64] = {0}, sum = 0;
    for (int g = 0; g < count; g++) {
      SourceGroup src = groups[g];
      if (src.offset < 0)
        continue;
      const double *v = block + src.offset + i * src.length;
      for (int j = 0; j < src.length; j++) {
        if (v[j])
          bins[bin(mag(v[j]) * amplitude[src.first + j])] += mag(v[j]) * amplitude[src.first + j];
      }
    }
    int cut = 63;
    while (cut >= 0 && sum + bins[cut] <= budget)
      sum += bins[cut--];
    for (int g = 0; g < count; g++) {
      SourceGroup src = groups[g];
      if (src.offset < 0)
        continue;
      const double *v = block + src.offset + i * src.length;
      for (int j = 0; j < src.length; j++) {
        int col = src.first + j;
        if (v[j] && bin(mag(v[j]) * amplitude[col]) <= cut) {
          float w = (float)v[j];
          out[used++] = (Link){col, w};
          sum += mag(v[j] - (double)w) * amplitude[col];
        }
      }
    }
    if (sum > *maxDiscard)
      *maxDiscard = sum;
    rows[first + i + 1] = used;
  }
  return used;
}
int restrictModes(int n, const int *rows, const Link *links, const unsigned char *active,
                  int *outRows, Link *out) {
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
int applyModes(int n, const int *rows, const Link *links, const double *source, double *result) {
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
