import { exact, addQ, subQ, mulQ, approximate } from './authored-scalar.js';
const zero = exact(0),
  two = exact(2);
const div = (a, b) => exact({ n: a.n * b.d, d: a.d * b.n });
const sign = (n) => (n < 0n ? -1 : n > 0n ? 1 : 0);
const compare = (a, b) => sign(a.n * b.d - b.n * a.d);
const dot = (a, b) => a.reduce((s, x, i) => addQ(s, mulQ(x, b[i])), zero);

// Rare ambiguous rays only. Split at the inner box's planes. On each
// interval the exact squared distance minus r² is A*t²+2*B*t+C.
// Compare a rational endpoint with either quadratic root by sign and squaring.
export function exactRoundedHit(box, origin, vector, tMin, tMax, endpoint = false) {
  const def = box.definition ?? {},
    r = exact(def.radius ?? box.radius),
    eye = Array.from(origin, exact),
    p = Array.from(vector, exact);
  const center = Array.from(def.center ?? box.center, exact),
    axes = Array.from(def.axes ?? box.axes, (a) => Array.from(a, exact));
  const H = Array.from(def.halfSize ?? box.halfSize, (h) => subQ(exact(h), r));
  const O = axes.map((a) =>
    dot(
      a,
      eye.map((x, i) => subQ(x, center[i])),
    ),
  );
  const D = axes.map((a) => dot(a, endpoint ? p.map((x, i) => subQ(x, eye[i])) : p));
  const lower = exact(tMin),
    upper = Number.isFinite(tMax) ? exact(tMax) : null,
    cuts = [lower];
  for (let i = 0; i < 3; i++)
    if (D[i].n)
      for (const h of [H[i], subQ(zero, H[i])]) {
        const t = div(subQ(h, O[i]), D[i]);
        if (compare(t, lower) > 0 && (!upper || compare(t, upper) < 0)) cuts.push(t);
      }
  if (upper) cuts.push(upper);
  cuts.sort(compare);
  // A finite ray's end serves as a last sample. For an infinite ray the
  // active feature after the last core crossing stays fixed to infinity.
  for (let k = 0; k < cuts.length; k++) {
    const a = cuts[k],
      b = cuts[k + 1];
    if (!b && upper) break;
    if (b && !compare(a, b)) continue;
    const mid = b ? div(addQ(a, b), two) : addQ(a, exact(1));
    let A = zero,
      B = zero,
      C = subQ(zero, mulQ(r, r));
    for (let i = 0; i < 3; i++) {
      const x = addQ(O[i], mulQ(mid, D[i]));
      if (compare(x, H[i]) <= 0 && compare(x, subQ(zero, H[i])) >= 0) continue;
      const q = x.n > 0n ? subQ(O[i], H[i]) : addQ(O[i], H[i]);
      A = addQ(A, mulQ(D[i], D[i]));
      B = addQ(B, mulQ(q, D[i]));
      C = addQ(C, mulQ(q, q));
    }
    if (!A.n) continue;
    const disc = subQ(mulQ(B, B), mulQ(A, C));
    if (disc.n < 0n) continue;
    const cmp = (x, branch) => {
      const v = addQ(mulQ(A, x), B);
      if (!disc.n) return sign(v.n);
      if (branch < 0) return v.n >= 0n ? 1 : compare(disc, mulQ(v, v));
      return v.n <= 0n ? -1 : compare(mulQ(v, v), disc);
    };
    for (const branch of [-1, 1]) {
      if (
        cmp(lower, branch) >= 0 ||
        (upper && cmp(upper, branch) <= 0) ||
        cmp(a, branch) > 0 ||
        (b && cmp(b, branch) < 0)
      )
        continue;
      const root = Math.sqrt(approximate(disc)),
        aa = approximate(A),
        bb = approximate(B);
      const q = -bb - (bb >= 0 ? root : -root);
      return branch < 0 === bb >= 0 || q === 0 ? q / aa : approximate(C) / q;
    }
  }
  return Infinity;
}
