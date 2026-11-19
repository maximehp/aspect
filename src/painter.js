// Separation witnesses for convex objects. Any ray hitting both objects must
// cross their separating plane. Opposite orderings on two axes prove that the
// objects occupy disjoint directions. Cycles use the clipped visibility path.
import { dot, cross, sub } from './cpu-geometry.js';
function radius(s, n) {
  return s.axes
    ? s.axes.reduce(
        (a, x, i) => a + Math.abs(dot(x, n)) * (s.halfSize[i] - s.radius),
        s.radius * Math.sqrt(dot(n, n)),
      )
    : Math.abs(dot(s.axisU, n)) + Math.abs(dot(s.axisV, n));
}
export function painter(shapes, eye) {
  const edges = shapes.map(() => []),
    incoming = new Int32Array(shapes.length);
  for (let i = 0; i < shapes.length; i++)
    for (let j = 0; j < i; j++) {
      const a = shapes[i],
        b = shapes[j],
        delta = sub(b.center, a.center),
        view = sub(eye, a.center);
      const axes = (s) => s.axes || [s.axisU, s.axisV, cross(s.axisU, s.axisV)];
      let order = 0,
        separate = false,
        irrelevant = false;
      for (const n of [...axes(a), ...axes(b)]) {
        const d = dot(delta, n),
          ra = radius(a, n),
          rb = radius(b, n);
        if (Math.abs(d) < 1e-10 || Math.abs(d) < ra + rb - 1e-9) continue;
        separate = true;
        const v = dot(view, n) * Math.sign(d),
          rate = v <= ra ? 1 : v >= Math.abs(d) - rb ? -1 : 0;
        if (!rate || (order && order !== rate)) {
          irrelevant = true;
          break;
        }
        order = rate;
      }
      if (irrelevant) continue;
      if (!separate) {
        const n = a.axes ? null : cross(a.axisU, a.axisV);
        if (
          n &&
          !b.axes &&
          Math.abs(dot(n, delta)) < 1e-9 &&
          Math.abs(dot(n, b.axisU)) + Math.abs(dot(n, b.axisV)) < 1e-9
        )
          continue;
        // No separating witness. Both directions belong to the unresolved
        // group, but certified relations to other objects still apply.
        edges[i].push(j);
        edges[j].push(i);
        incoming[i]++;
        incoming[j]++;
        continue;
      }
      const [far, near] = order === 1 ? [j, i] : [i, j];
      edges[far].push(near);
      incoming[near]++;
    }
  const out = [];
  for (let i = 0; i < incoming.length; i++) if (!incoming[i]) out.push(i);
  for (let h = 0; h < out.length; h++) for (const i of edges[out[h]]) if (--incoming[i] === 0) out.push(i);
  if (out.length === shapes.length) return out;
  // A cycle needs exact clipping only among its own members. The strongly
  // connected components themselves admit a far-to-near order. Keep the
  // ordinary acyclic path above allocation-light.
  const reverse = shapes.map(() => []);
  for (let i = 0; i < edges.length; i++) for (const j of edges[i]) reverse[j].push(i);
  const visited = new Uint8Array(edges.length),
    next = new Int32Array(edges.length);
  const finished = [],
    stack = [];
  for (let i = 0; i < edges.length; i++) {
    if (visited[i]) continue;
    stack.push(i);
    visited[i] = 1;
    while (stack.length) {
      const v = stack[stack.length - 1];
      if (next[v] === edges[v].length) finished.push(stack.pop());
      else {
        const j = edges[v][next[v]++];
        if (!visited[j]) {
          visited[j] = 1;
          stack.push(j);
        }
      }
    }
  }
  visited.fill(0);
  const groups = [];
  while (finished.length) {
    const i = finished.pop();
    if (visited[i]) continue;
    const members = [];
    visited[i] = 1;
    stack.push(i);
    while (stack.length) {
      const v = stack.pop();
      members.push(v);
      for (const j of reverse[v])
        if (!visited[j]) {
          visited[j] = 1;
          stack.push(j);
        }
    }
    groups.push(members.length === 1 ? i : members.sort((a, b) => a - b));
  }
  return groups;
}
