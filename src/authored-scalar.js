// Decimal text is author input. A Number already denotes its IEEE-754 value.
// Preserve that distinction until a geometric predicate needs exact arithmetic.
const bits = new DataView(new ArrayBuffer(8));
function ratio(n, d = 1n) {
  if (d === 0n) throw RangeError('Zero denominator.');
  if (d < 0n) {
    n = -n;
    d = -d;
  }
  let a = n < 0n ? -n : n,
    b = d;
  while (b) {
    const t = a % b;
    a = b;
    b = t;
  }
  return { n: n / a, d: d / a };
}
export function exact(value) {
  if (typeof value === 'object' && value && typeof value.n === 'bigint' && typeof value.d === 'bigint')
    return ratio(value.n, value.d);
  if (typeof value === 'string') {
    const s = value.trim(),
      fraction = /^([+-]?\d+)\/(\d+)$/.exec(s);
    if (fraction) return ratio(BigInt(fraction[1]), BigInt(fraction[2]));
    const m = /^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$/.exec(s);
    if (!m || !(m[2] || m[3])) throw TypeError('Expected a decimal or rational dimension.');
    const power = Number(m[4] || 0) - (m[3]?.length || 0);
    if (!Number.isSafeInteger(power) || Math.abs(power) > 10000)
      throw RangeError('Decimal exponent outside supported range.');
    const n = BigInt((m[1] || '') + (m[2] || '') + (m[3] || ''));
    return power >= 0 ? ratio(n * 10n ** BigInt(power)) : ratio(n, 10n ** BigInt(-power));
  }
  if (!Number.isFinite(value)) throw TypeError('Expected a finite scalar.');
  bits.setFloat64(0, value);
  const word = bits.getBigUint64(0),
    e = Number((word >> 52n) & 2047n);
  const n = (word >> 63n ? -1n : 1n) * ((word & ((1n << 52n) - 1n)) + (e ? 1n << 52n : 0n));
  const k = (e || 1) - 1075;
  return k >= 0 ? ratio(n << BigInt(k)) : ratio(n, 1n << BigInt(-k));
}
export const addQ = (a, b) => ratio(a.n * b.d + b.n * a.d, a.d * b.d);
export const subQ = (a, b) => ratio(a.n * b.d - b.n * a.d, a.d * b.d);
export const mulQ = (a, b) => ratio(a.n * b.n, a.d * b.d);
export function approximate(q) {
  if (!q.n) return 0;
  const a = q.n < 0n ? -q.n : q.n;
  const k = Math.max(0, a.toString(2).length - 54),
    j = Math.max(0, q.d.toString(2).length - 54);
  return (((q.n < 0n ? -1 : 1) * Number(a >> BigInt(k))) / Number(q.d >> BigInt(j))) * 2 ** (k - j);
}

export function scalarValue(value) {
  if (typeof value === 'number') return value;
  const q = exact(value);
  return typeof value === 'string' && !value.includes('/') ? Number(value) : approximate(q);
}
