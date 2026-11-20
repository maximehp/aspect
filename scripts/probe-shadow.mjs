// Live room geometry, with two independent formulations: interval distance to
// the inner box and the tangent cone of its axial fillet. Probe both sides of
// each actual floor shadow edge at representable positions around its root.
import { buildRoom } from '../src/room.js';
import { roundedBoxDistance } from '../src/rounded-box.js';
import { exactRoundedHit } from '../src/exact-hit.js';
import { exact, addQ, mulQ, approximate } from '../src/authored-scalar.js';
import { silhouetteEquation } from '../src/silhouette.js';

const bits = new DataView(new ArrayBuffer(8));
function shift(x, ulps) {
  bits.setFloat64(0, x);
  bits.setBigUint64(0, bits.getBigUint64(0) + BigInt(x < 0 ? -ulps : ulps));
  return bits.getFloat64(0);
}
const edges = [],
  floor = [0, 0, -1.25],
  axis = [1, 0, 0];
let queries = 0,
  floatDisagreements = 0,
  exactDisagreements = 0;
for (let tick = 0; tick <= 20; tick++) {
  const angle = 0.299 + tick * 0.0001,
    scene = buildRoom({ angle, slats: 1, radius: '0.044' });
  const box = scene.roundedBoxes[0],
    eye = scene.lightPosition;
  const binaryTest = (x) =>
    roundedBoxDistance(box, eye, [x - eye[0], -eye[1], floor[2] - eye[2]], 1e-8, 1) < 1;
  const row = [];
  for (let signs = 0; signs < 4; signs++) {
    const f = silhouetteEquation(box, eye, floor, axis, 2, signs);
    if (f.discriminant.n <= 0n) continue;
    const a = approximate(f.A),
      b = approximate(f.B),
      c = approximate(f.C),
      q = -b - Math.sign(b) * Math.sqrt(approximate(f.discriminant));
    for (const root of [q / a, c / q]) {
      if (Math.abs(root) > 3.4 || binaryTest(root - 1e-7) === binaryTest(root + 1e-7)) continue;
      row.push(root);
      for (const ulps of [-16, -4, -1, 0, 1, 4, 16]) {
        const x = shift(root, ulps),
          u = exact(x),
          v = addQ(mulQ(addQ(mulQ(f.A, u), addQ(f.B, f.B)), u), f.C);
        const conic = v.n >= 0n,
          hit = exactRoundedHit(box, eye, [x, 0, floor[2]], 1e-8, 1, true) < 1;
        queries++;
        floatDisagreements += binaryTest(x) !== conic;
        exactDisagreements += hit !== conic;
      }
    }
  }
  edges.push(row.sort((a, b) => a - b));
}
const maxStep = Math.max(...edges.slice(1).flatMap((e, i) => e.map((x, j) => Math.abs(x - edges[i][j]))));
console.log(JSON.stringify({ queries, floatDisagreements, exactDisagreements, edges, maxStep }, null, 2));
