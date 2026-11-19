// Center samples carry the smooth remainder of the diffuse solution. Each edge
// has its own knots so two differently sized faces can share an identical
// boundary curve even where a third face terminates partway along that edge.
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a, b) => a.map((v, i) => v - b[i]);
const scaled = (v, k) => v.map((x) => x * k);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const clamp = (x, min, max) => Math.max(min, Math.min(x, max));
const interpolate = (a, b, t) => a.map((v, i) => v + t * (b[i] - v));
const paramTolerance = 128 * Number.EPSILON;

function world(rect, u, v) {
  return rect.center.map((x, k) => x + (2 * u - 1) * rect.axisU[k] + (2 * v - 1) * rect.axisV[k]);
}

function edgeUV(side, t) {
  return side < 2 ? [side, t] : [t, side - 2];
}

function makeFace(surface, index) {
  const { rect, bounce } = surface;
  const { nu, nv, front, back } = bounce || { nu: 0, nv: 0, front: [], back: [] };
  if (
    bounce &&
    (!Number.isInteger(nu) ||
      !Number.isInteger(nv) ||
      nu < 1 ||
      nv < 1 ||
      front.length !== 3 * nu * nv ||
      back.length !== front.length)
  ) {
    throw new Error('Invalid reflected-light sample grid on ' + rect.id);
  }
  const n = cross(rect.axisU, rect.axisV),
    area = Math.hypot(...n);
  if (!(area > 0)) throw new Error('Degenerate surface: ' + rect.id);
  const normal = scaled(n, 1 / area);
  const face = {
    index,
    rect,
    bounce,
    nu,
    nv,
    normal,
    dualU: scaled(cross(rect.axisV, normal), 1 / area),
    dualV: scaled(cross(normal, rect.axisU), 1 / area)
  };
  face.edges = Array.from({ length: 4 }, (_, side) => {
    const start = world(rect, ...edgeUV(side, 0)),
      end = world(rect, ...edgeUV(side, 1));
    const delta = sub(end, start),
      length = Math.hypot(...delta),
      dir = scaled(delta, 1 / length);
    const count = side < 2 ? nv : nu;
    return {
      face,
      side,
      start,
      end,
      dir,
      length,
      knots: [0, ...Array.from({ length: count }, (_, k) => (k + 0.5) / count), 1]
    };
  });
  return face;
}

function addKnot(edge, t) {
  t = clamp(t, 0, 1);
  const old = edge.knots.find((x) => Math.abs(x - t) <= paramTolerance);
  if (old !== undefined) return old;
  edge.knots.push(t);
  return t;
}

const pointAt = (e, t) => interpolate(e.start, e.end, t);
const paramAt = (e, p) => dot(sub(p, e.start), e.dir) / e.length;

function barrierIntervals(a, b, p, q, faces, tolerance) {
  const blocked = [[], []];
  for (const wall of faces) {
    if (wall === a || wall === b || Math.hypot(...cross(a.normal, wall.normal)) <= paramTolerance) continue;
    const x = sub(p, wall.rect.center),
      y = sub(q, wall.rect.center);
    if (Math.abs(dot(x, wall.normal)) > tolerance || Math.abs(dot(y, wall.normal)) > tolerance) continue;
    // Intersect the shared line segment with a third, noncoplanar sheet.
    let lo = 0,
      hi = 1;
    for (const dual of [wall.dualU, wall.dualV]) {
      const u = dot(x, dual),
        v = dot(y, dual),
        d = v - u;
      if (Math.abs(d) <= paramTolerance) {
        if (Math.abs(u) > 1 + paramTolerance) {
          hi = -1;
          break;
        }
      } else {
        const t0 = (-1 - u) / d,
          t1 = (1 - u) / d;
        lo = Math.max(lo, Math.min(t0, t1));
        hi = Math.min(hi, Math.max(t0, t1));
      }
    }
    if (hi - lo <= paramTolerance) continue;
    const h = dot(sub(wall.rect.center, p), a.normal);
    const radius = Math.abs(dot(wall.rect.axisU, a.normal)) + Math.abs(dot(wall.rect.axisV, a.normal));
    if (h + radius > tolerance) blocked[0].push([lo, hi]);
    if (h - radius < -tolerance) blocked[1].push([lo, hi]);
  }
  return blocked;
}

function findJoins(faces) {
  const joins = [];
  for (let i = 0; i < faces.length; i++)
    for (let j = i + 1; j < faces.length; j++) {
      const a = faces[i],
        b = faces[j];
      if (!a.bounce || !b.bounce || a.rect.color.some((v, k) => v !== b.rect.color[k])) continue;
      if (Math.hypot(...cross(a.normal, b.normal)) > paramTolerance) continue;
      const tol =
        paramTolerance *
        Math.max(
          1,
          Math.hypot(...a.rect.center),
          Math.hypot(...b.rect.center),
          ...a.edges.map((e) => e.length),
          ...b.edges.map((e) => e.length)
        );
      if (Math.abs(dot(sub(a.rect.center, b.rect.center), a.normal)) > tol) continue;
      for (const ea of a.edges)
        for (const eb of b.edges) {
          if (
            Math.hypot(...cross(ea.dir, eb.dir)) > paramTolerance ||
            Math.hypot(...cross(sub(ea.start, eb.start), ea.dir)) > tol
          )
            continue;
          const inward = cross(a.normal, ea.dir);
          if (dot(sub(a.rect.center, ea.start), inward) * dot(sub(b.rect.center, ea.start), inward) >= 0)
            continue;
          const t0 = paramAt(ea, eb.start),
            t1 = paramAt(ea, eb.end);
          const lo = Math.max(0, Math.min(t0, t1)),
            hi = Math.min(1, Math.max(t0, t1));
          if ((hi - lo) * ea.length <= tol) continue;
          const blocked = barrierIntervals(a, b, pointAt(ea, lo), pointAt(ea, hi), faces, tol);
          const flip = dot(a.normal, b.normal) < 0 ? 1 : 0;
          for (let side = 0; side < 2; side++) {
            const cuts = [0, 1, ...blocked[side].flat()].sort((x, y) => x - y);
            for (let k = 1; k < cuts.length; k++) {
              const l = cuts[k - 1],
                h = cuts[k],
                mid = (l + h) / 2;
              if (h - l <= paramTolerance || blocked[side].some(([x, y]) => x < mid && mid < y)) continue;
              const start = addKnot(ea, lo + (hi - lo) * l),
                end = addKnot(ea, lo + (hi - lo) * h);
              addKnot(eb, paramAt(eb, pointAt(ea, start)));
              addKnot(eb, paramAt(eb, pointAt(ea, end)));
              joins.push({ ea, eb, start, end, side, otherSide: side ^ flip });
            }
          }
        }
    }
  for (const { ea, eb, start, end } of joins) {
    for (const t of [...ea.knots])
      if (t >= start - paramTolerance && t <= end + paramTolerance) addKnot(eb, paramAt(eb, pointAt(ea, t)));
    for (const s of eb.knots) {
      const t = paramAt(ea, pointAt(eb, s));
      if (t >= start - paramTolerance && t <= end + paramTolerance) addKnot(ea, t);
    }
  }
  return joins;
}

function patchRemainder(face, u, v, side, useFirst) {
  const { nu, nv, bounce } = face;
  const total = side ? bounce.back : bounce.front;
  const first = useFirst && (side ? bounce.firstBack : bounce.firstFront);
  const x = clamp(u * nu - 0.5, 0, nu - 1),
    y = clamp(v * nv - 0.5, 0, nv - 1);
  const i = Math.floor(x),
    j = Math.floor(y);
  const val = (a, b, c) => {
    const k = 3 * (b * nu + a) + c;
    return total[k] - (first ? first[k] : 0);
  };
  const i1 = Math.min(i + 1, nu - 1),
    j1 = Math.min(j + 1, nv - 1);
  return [0, 1, 2].map(
    (c) =>
      (1 - y + j) * ((1 - x + i) * val(i, j, c) + (x - i) * val(i1, j, c)) +
      (y - j) * ((1 - x + i) * val(i, j1, c) + (x - i) * val(i1, j1, c))
  );
}

function root(n) {
  while (n.parent !== n) {
    n.parent = n.parent.parent;
    n = n.parent;
  }
  return n;
}

export function buildReconstruction(surfaces, field, maxSize) {
  const faces = surfaces.map(makeFace),
    joins = findJoins(faces);
  const all = [];
  for (const face of faces) {
    if (!face.bounce) continue;
    const cornerNodes = new Map();
    const node = (u, v) => {
      // A corner belongs to two local edge curves.
      const corner = (u === 0 || u === 1) && (v === 0 || v === 1) ? u + 2 * v : null;
      if (corner !== null && cornerNodes.has(corner)) return cornerNodes.get(corner);
      const sides = [0, 1].map((side) => {
        const n = { face, u, v, side };
        n.parent = n;
        all.push(n);
        return n;
      });
      if (corner !== null) cornerNodes.set(corner, sides);
      return sides;
    };
    face.centers = [];
    for (let j = 0; j < face.nv; j++)
      for (let i = 0; i < face.nu; i++) face.centers.push(node((i + 0.5) / face.nu, (j + 0.5) / face.nv));
    for (const edge of face.edges) {
      edge.knots.sort((a, b) => a - b);
      edge.nodes = edge.knots.map((t) => node(...edgeUV(edge.side, t)));
    }
  }
  for (const { ea, eb, start, end, side, otherSide } of joins) {
    for (const [i, t] of ea.knots.entries()) {
      if (t < start - paramTolerance || t > end + paramTolerance) continue;
      const s = paramAt(eb, pointAt(ea, t)),
        j = eb.knots.findIndex((x) => Math.abs(x - s) <= paramTolerance);
      if (j < 0) continue;
      root(ea.nodes[i][side]).parent = root(eb.nodes[j][otherSide]);
    }
  }
  for (const n of all) {
    const group = root(n),
      useFirst = field && n.face.bounce.firstFront && n.face.bounce.firstBack;
    const value = patchRemainder(n.face, n.u, n.v, n.side, useFirst);
    if (!group.count) {
      group.count = 0;
      group.sum = [0, 0, 0];
      // A query exactly on a floor/wall intersection can see both rooms
      // through a zero-length segment. Take the limit from this sheet's
      // interior. A shared coplanar knot still has one common RGB value.
      const inward = (t) => (t === 0 ? 1e-8 : t === 1 ? 1 - 1e-8 : t);
      group.first = useFirst
        ? field.at(
            n.face.index,
            world(n.face.rect, inward(n.u), inward(n.v)),
            n.side ? scaled(n.face.normal, -1) : n.face.normal
          )
        : [0, 0, 0];
    }
    for (let k = 0; k < 3; k++) group.sum[k] += value[k];
    group.count++;
  }
  const records = [],
    layouts = [];
  for (const face of faces) {
    if (!face.bounce) {
      layouts.push([0, 0, 0, 0]);
      continue;
    }
    const stride = face.centers.length + 4 + face.edges.reduce((n, e) => n + e.knots.length, 0);
    layouts.push([records.length / 4, stride, face.nu, face.nv]);
    const write = (nodes, side, t = 0) => {
      const g = root(nodes[side]);
      records.push(...g.sum.map((x, k) => Math.max(0, g.first[k] + x / g.count)), t);
    };
    for (let side = 0; side < 2; side++) {
      for (const nodes of face.centers) write(nodes, side);
      let offset = face.centers.length + 4;
      for (const e of face.edges) {
        records.push(offset, e.knots.length, 0, 0);
        offset += e.knots.length;
      }
      for (const e of face.edges)
        for (let k = 0; k < e.knots.length; k++) write(e.nodes[k], side, e.knots[k]);
    }
  }
  const count = records.length / 4;
  if (count > Math.min(maxSize * maxSize, 2 ** 24))
    throw new Error('Reflected-light field exceeds GPU addressing limits.');
  const width = Math.max(1, Math.ceil(Math.sqrt(count))),
    height = Math.max(1, Math.ceil(count / width));
  const pixels = new Float32Array(width * height * 4);
  pixels.set(records);
  return { pixels, width, height, layouts };
}
