// An axis split restricts the domain of a function; it never changes the
// authored surface equation. Each leaf has four bilinear Bernstein functions.
export const domain = () => ({ bounds: [0, 1, 0, 1] });
export function split(leaf, axis, cut) {
  const a = leaf.bounds.slice(),
    b = a.slice(),
    k = 2 * axis;
  if (leaf.children || !(cut > a[k] && cut < a[k + 1])) return;
  a[k + 1] = b[k] = cut;
  leaf.axis = axis;
  leaf.cut = cut;
  leaf.children = [{ bounds: a }, { bounds: b }];
}
export function leaves(root, out = []) {
  if (root.children) for (const c of root.children) leaves(c, out);
  else out.push(root);
  return out;
}
export function locate(root, u, v) {
  while (root.children) root = root.children[(root.axis ? v : u) < root.cut ? 0 : 1];
  return root;
}
export function hats(u, v) {
  return [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v];
}
export function fieldValue(chart, u, v, values) {
  if (chart.empty) return [0, 0, 0];
  const leaf = locate(chart.domain, u, v),
    [a, b, l, h] = leaf.bounds,
    phi = hats((u - a) / (b - a), (v - l) / (h - l));
  return [0, 1, 2].map((c) => leaf.ids.reduce((sum, id, k) => sum + phi[k] * values[3 * id + c], 0));
}

// A finer leaf meeting a coarse edge must give that edge exactly the same
// function. Eliminate each hanging coefficient by restriction of its two
// neighbours. Contacts select a side before imposing these constraints.
export function continuousBasis(charts, key, sector) {
  const vertices = [],
    byKey = new Map(),
    lines = new Map(),
    supports = [];
  function vertex(g, u, v, leaf, findOnly = false) {
    const region = sector(g, u, v, leaf);
    const name = key(g, u, v) + (region ? '/side/' + region : '');
    if (findOnly) return byKey.get(name);
    if (!byKey.has(name)) {
      byKey.set(name, vertices.length);
      vertices.push({ object: g.object, component: 2 * g.object + (g.shape.axes ? 0 : g.slot) });
    }
    return byKey.get(name);
  }
  function edge(g, leaf, d, fixed, a, b, i, j) {
    if (i === j) return;
    let name = g.offset + '/' + d + '/' + fixed;
    if (fixed === 0 || fixed === 1) {
      const end = (t) => (d ? key(g, fixed, t) : key(g, t, fixed)),
        p = end(0),
        q = end(1);
      name = p < q ? p + ';' + q : q + ';' + p;
      if (q < p) {
        [a, b] = [1 - b, 1 - a];
        [i, j] = [j, i];
      }
    }
    if (!lines.has(name)) lines.set(name, { points: new Map(), edges: [] });
    const line = lines.get(name);
    if (!line.points.has(a)) line.points.set(a, new Set());
    if (!line.points.has(b)) line.points.set(b, new Set());
    line.points.get(a).add(i);
    line.points.get(b).add(j);
    line.edges.push([
      a,
      b,
      i,
      j,
      g.contacts?.length && fixed > 0 && fixed < 1
        ? (t) => vertex(g, d ? fixed : t, d ? t : fixed, leaf, true)
        : null,
    ]);
  }
  for (const g of charts)
    if (!g.empty)
      for (const leaf of leaves(g.domain)) {
        const [a, b, l, h] = leaf.bounds,
          [i, j, k, m] = (leaf.ids = [
            vertex(g, a, l, leaf),
            vertex(g, b, l, leaf),
            vertex(g, a, h, leaf),
            vertex(g, b, h, leaf),
          ]);
        edge(g, leaf, 0, l, a, b, i, j);
        edge(g, leaf, 0, h, a, b, k, m);
        edge(g, leaf, 1, a, l, h, i, k);
        edge(g, leaf, 1, b, l, h, j, m);
      }
  for (const line of lines.values()) {
    const points = line.points;
    for (const [a, b, i, j, trace] of line.edges)
      for (const [t, ids] of points)
        for (const k of ids) {
          if (t <= a || t >= b || k === i || k === j || (trace && trace(t) !== k)) continue;
          if (!vertices[k].constraint || b - a > vertices[k].constraint.width)
            vertices[k].constraint = { i, j, f: (t - a) / (b - a), width: b - a };
        }
  }
  function resolve(v) {
    if (v.basis) return v.basis;
    if (v.visiting) throw Error('Cyclic surface edge restriction');
    v.visiting = true;
    if (!v.constraint) {
      v.basis = [[supports.length, 1]];
      supports.push({ object: v.object, component: v.component });
    } else {
      const { i, j, f } = v.constraint,
        terms = new Map();
      for (const [k, w] of [
        [i, 1 - f],
        [j, f],
      ])
        for (const [p, s] of resolve(vertices[k])) terms.set(p, (terms.get(p) || 0) + w * s);
      v.basis = [...terms];
    }
    return v.basis;
  }
  for (const v of vertices) resolve(v);
  return { vertices, supports };
}
