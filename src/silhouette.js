import { exact, addQ, subQ, mulQ, approximate } from './authored-scalar.js';
const zero = exact(0),
  sq = (x) => mulQ(x, x);
const dotQ = (a, b) => a.reduce((s, v, i) => addQ(s, mulQ(v, b[i])), zero);
const subV = (a, b) => a.map((v, i) => subQ(v, b[i]));

// The tangent cone and the finite-segment endpoint surface are quadrics.
// Substituting source position p+x*u retains an exact root description
// A*x²+2*B*x+C=0. signs and root ordering need no decimal root expansion.
export function silhouetteEquation(shape, lamp, p, u, axis, signs, endpoint = false) {
  const values = shape.definition ?? {},
    frame = (values.axes ?? shape.axes).map((a) => a.map(exact)),
    r = exact(values.radius ?? shape.radius);
  const half = (values.halfSize ?? shape.halfSize).map(exact),
    eye = lamp.map(exact),
    center = (values.center ?? shape.center).map(exact);
  const O = subV(eye, center),
    D = subV(p.map(exact), eye),
    U = u.map(exact);
  const q = [],
    d = [],
    v = [];
  for (let j = 0; j < 3; j++)
    if (j !== axis) {
      const h = subQ(half[j], r);
      q.push((signs & (1 << j) ? subQ : addQ)(dotQ(O, frame[j]), h));
      d.push(dotQ(D, frame[j]));
      v.push(dotQ(U, frame[j]));
    }
  const S = subQ(dotQ(q, q), sq(r)),
    qu = dotQ(q, v),
    qd = dotQ(q, d);
  const x = q.map((z, i) => addQ(z, d[i]));
  const A = endpoint ? dotQ(v, v) : subQ(sq(qu), mulQ(S, dotQ(v, v)));
  const B = endpoint ? dotQ(x, v) : subQ(mulQ(qu, qd), mulQ(S, dotQ(v, d)));
  const C = endpoint ? subQ(dotQ(x, x), sq(r)) : subQ(sq(qd), mulQ(S, dotQ(d, d)));
  return { A, B, C, discriminant: subQ(sq(B), mulQ(A, C)) };
}

// Called only when the native coefficient filter cannot settle a tangency.
export function conicImports(context) {
  return {
    refine: (id, axis, signs, px, py, pz, ux, uy, uz, endpoint, address) => {
      const { packet, memory } = context(),
        s = packet.shapes.subarray(19 * id, 19 * id + 19);
      const shape = {
        center: Array.from(s.subarray(0, 3)),
        axes: [3, 6, 9].map((k) => Array.from(s.subarray(k, k + 3))),
        halfSize: Array.from(s.subarray(12, 15)),
        radius: s[15],
        definition: packet.definitions?.[id],
      };
      const f = silhouetteEquation(
        shape,
        Array.from(packet.lamp),
        [px, py, pz],
        [ux, uy, uz],
        axis,
        signs,
        endpoint,
      );
      new Float64Array(memory.buffer, address, 4).set([f.A, f.B, f.C, f.discriminant].map(approximate));
    },
  };
}
