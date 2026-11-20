import { exact, addQ, subQ, mulQ, approximate } from './authored-scalar.js';
const zero = exact(0),
  sq = (x) => mulQ(x, x);
const dotQ = (a, b) => a.reduce((s, v, i) => addQ(s, mulQ(v, b[i])), zero);
const subV = (a, b) => a.map((v, i) => subQ(v, b[i]));

// The tangent cone and the finite-segment endpoint surface are quadrics.
// Substituting source position p+x*u retains an exact root description
// A*x²+2*B*x+C=0. signs and root ordering need no decimal root expansion.
export function filletBoundary(shape, lamp, axis, signs, endpoint = false) {
  const values = shape.definition ?? {},
    frame = (values.axes ?? shape.axes).map((a) => a.map(exact)),
    half = (values.halfSize ?? shape.halfSize).map(exact),
    r = exact(values.radius ?? shape.radius),
    eye = lamp.map(exact),
    origin = subV(eye, (values.center ?? shape.center).map(exact));
  const directions = [],
    q = [];
  for (let d = 0; d < 3; d++)
    if (d !== axis) {
      directions.push(frame[d]);
      q.push((signs & (1 << d) ? subQ : addQ)(dotQ(origin, frame[d]), subQ(half[d], r)));
    }
  const S = subQ(dotQ(q, q), sq(r));
  const local = (v) => directions.map((a) => dotQ(a, v));
  function form(a, b) {
    return endpoint ? dotQ(a, b) : subQ(mulQ(dotQ(q, a), dotQ(q, b)), mulQ(S, dotQ(a, b)));
  }
  function linear(d, u) {
    return endpoint ? addQ(dotQ(q, u), form(d, u)) : form(d, u);
  }
  function constant(d) {
    if (!endpoint) return form(d, d);
    const end = q.map((v, i) => addQ(v, d[i]));
    return subQ(dotQ(end, end), sq(r));
  }
  return {
    along(p, u) {
      const d = local(subV(p.map(exact), eye)),
        v = local(u.map(exact)),
        A = form(v, v),
        B = linear(d, v),
        C = constant(d);
      return { A, B, C, discriminant: subQ(sq(B), mulQ(A, C)) };
    },
    onPlane(p, u, v) {
      const d = local(subV(p.map(exact), eye)),
        x = local(u.map(exact)),
        y = local(v.map(exact));
      // Q = xx*u² + 2*xy*u*v + yy*v² + 2*x*u + 2*y*v + c
      return {
        xx: form(x, x),
        xy: form(x, y),
        yy: form(y, y),
        x: linear(d, x),
        y: linear(d, y),
        c: constant(d),
      };
    },
  };
}
export function silhouetteEquation(shape, lamp, p, u, axis, signs, endpoint = false) {
  return filletBoundary(shape, lamp, axis, signs, endpoint).along(p, u);
}

// Called only when the native coefficient filter cannot settle a tangency.
export function conicImports(context) {
  const poses = new WeakMap();
  return {
    refine: (id, axis, signs, px, py, pz, ux, uy, uz, endpoint, address) => {
      const { packet, memory } = context();
      let curves = poses.get(packet);
      if (!curves) {
        curves = new Map();
        poses.set(packet, curves);
      }
      const key = ((id * 4 + axis + 1) * 8 + signs) * 2 + endpoint;
      let boundary = curves.get(key);
      if (!boundary) {
        const s = packet.shapes.subarray(19 * id, 19 * id + 19);
        const shape = {
          center: Array.from(s.subarray(0, 3)),
          axes: [3, 6, 9].map((k) => Array.from(s.subarray(k, k + 3))),
          halfSize: Array.from(s.subarray(12, 15)),
          radius: s[15],
          definition: packet.definitions?.[id],
        };
        boundary = filletBoundary(shape, Array.from(packet.lamp), axis, signs, endpoint);
        curves.set(key, boundary);
      }
      const f = boundary.along([px, py, pz], [ux, uy, uz]);
      new Float64Array(memory.buffer, address, 4).set([f.A, f.B, f.C, f.discriminant].map(approximate));
    },
  };
}
