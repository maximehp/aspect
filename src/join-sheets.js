// Exact full-edge union of two equal Lambertian rectangles. Each retained
// coordinate must still denote the authored point in binary64 after the join.
import { exact, mulQ, subQ } from './authored-scalar.js';
const axes = ['axisU', 'axisV'];
const allowed = new Set(['id', 'center', 'axisU', 'axisV', 'color']);
function sum(a, b) {
  const x = a + b,
    z = x - a;
  return [x, a - (x - z) + (b - z)];
}
const exactVec = (a, b, k) => a.map((v, i) => sum(v, k * b[i]));
function matchesDirection(a, b) {
  if (!a.every((x, i) => x * b[(i + 1) % 3] === a[(i + 1) % 3] * b[i])) return false;
  return a.every((x, i) => {
    const j = (i + 1) % 3;
    return !subQ(mulQ(exact(x), exact(b[j])), mulQ(exact(a[j]), exact(b[i]))).n;
  });
}
export function joinSheets(rectangles) {
  const edges = new Map(),
    entries = rectangles.map((shape, i) => ({ shape, ids: [i], live: true }));
  // Gameplay identities stay in ids. Moving either author object recompiles
  // the union from that pose. An unrecognized surface property inhibits fusion.
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i],
      s = entry.shape;
    if (Object.keys(s).some((k) => !allowed.has(k))) continue;
    outer: for (let d = 0; d < 2; d++) {
      const along = s[axes[d]],
        cross = s[axes[1 - d]],
        first = cross.find((v) => v !== 0);
      if (first === undefined) continue;
      const prefix = [...s.color, ...cross.map((x) => (first < 0 ? -x : x))].join(',') + '|';
      for (const side of [-1, 1]) {
        const key = prefix + exactVec(s.center, along, side).flat().join(',');
        if (!edges.has(key)) edges.set(key, []);
        const bucket = edges.get(key);
        for (const neighbor of bucket) {
          if (!neighbor.entry.live) continue;
          const t = neighbor.entry.shape,
            v = t[axes[neighbor.d]],
            k = -side * neighbor.side;
          if (!along.some((x, i) => x * k * v[i] > 0) || !matchesDirection(along, v)) continue;
          const center = exactVec(s.center, v, -neighbor.side),
            extent = exactVec(along, v, k);
          if (center.some((x) => x[1]) || extent.some((x) => x[1])) continue;
          entry.live = neighbor.entry.live = false;
          const ids = [...entry.ids, ...neighbor.entry.ids].sort((a, b) => a - b);
          entries.push({
            shape: {
              ...s,
              id: rectangles[ids[0]].id,
              center: center.map((x) => x[0]),
              [axes[d]]: extent.map((x) => x[0]),
            },
            ids,
            live: true,
          });
          break outer;
        }
        bucket.push({ entry, d, side });
      }
    }
  }
  return entries.filter((e) => e.live).sort((a, b) => a.ids[0] - b.ids[0]);
}
