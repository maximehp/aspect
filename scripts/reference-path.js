// Isolate named bounces in the independent reference. The planar sequence
// uses cosine-direction sampling. At its endpoint, sample the small rounded
// sources by true area to resolve their much weaker contribution.
export function referencePath({
  room,
  faces,
  target,
  side,
  path,
  paths,
  trace,
  random,
  dot,
  scale,
  advance,
}) {
  const charts = [],
    halfpi = Math.PI / 2;
  let area = 0;
  for (const box of room.roundedBoxes) {
    const h = box.halfSize.map((x) => x - box.radius),
      r = box.radius;
    function add(a, get) {
      if (a > 0) {
        area += a;
        charts.push({ end: area, box, get });
      }
    }
    for (let d = 0; d < 3; d++) {
      const i = (d + 1) % 3,
        j = (d + 2) % 3;
      for (const sign of [-1, 1])
        add(4 * h[i] * h[j], (a, b) => {
          const p = [0, 0, 0],
            n = [0, 0, 0];
          n[d] = sign;
          p[d] = sign * (h[d] + r);
          p[i] = (2 * a - 1) * h[i];
          p[j] = (2 * b - 1) * h[j];
          return [p, n];
        });
      for (const si of [-1, 1])
        for (const sj of [-1, 1])
          add(2 * h[d] * halfpi * r, (a, b) => {
            const p = [0, 0, 0],
              n = [0, 0, 0];
            n[i] = si * Math.cos(halfpi * b);
            n[j] = sj * Math.sin(halfpi * b);
            p[d] = (2 * a - 1) * h[d];
            p[i] = si * h[i] + r * n[i];
            p[j] = sj * h[j] + r * n[j];
            return [p, n];
          });
    }
    for (const sx of [-1, 1])
      for (const sy of [-1, 1])
        for (const sz of [-1, 1])
          add(halfpi * r * r, (a, b) => {
            const q = Math.sqrt(1 - a * a),
              n = [sx * q * Math.cos(b * halfpi), sy * q * Math.sin(b * halfpi), sz * a];
            return [[sx, sy, sz].map((v, k) => v * h[k] + r * n[k]), n];
          });
  }
  function solid() {
    const selector = random() * area,
      s = charts.find((c) => c.end >= selector),
      { box } = s,
      [p, n] = s.get(random(), random());
    const rotate = (v) => [0, 1, 2].map((k) => v.reduce((a, x, j) => a + x * box.axes[j][k], 0));
    return {
      point: advance(box.center, rotate(p), 1),
      normal: rotate(n),
      color: box.color,
      id: faces.length + room.roundedBoxes.indexOf(box),
    };
  }
  function cosineDirection(f, sign) {
    const a = random(),
      b = random() * Math.PI * 2,
      r = Math.sqrt(a);
    return advance(
      advance(scale(f.tangent, r * Math.cos(b)), f.bitangent, r * Math.sin(b)),
      f.normal,
      sign * Math.sqrt(1 - a),
    );
  }
  const via = path.via.map((id) => faces.findIndex((f) => f.id === id));
  if (!via.length || via.some((i) => i < 0)) throw Error('Supply a sequence of existing planar surface IDs.');
  const wall = via.at(-1),
    point = path.point ?? [0.5, 0.5];
  const receiver = faces[target],
    origin = advance(
      advance(receiver.center, receiver.axisU, 2 * point[0] - 1),
      receiver.axisV,
      2 * point[1] - 1,
    );
  const sum = [
      [0, 0, 0],
      [0, 0, 0],
    ],
    squares = [
      [0, 0, 0],
      [0, 0, 0],
    ];
  function deposit(k, throughput, factor) {
    for (let c = 0; c < 3; c++) {
      let x = throughput[c] * room.lampRGB[c] * factor;
      sum[k][c] += x;
      squares[k][c] += x * x;
    }
  }
  let wallHits = 0;
  for (let it = 0; it < paths; it++) {
    let p = origin,
      f = receiver,
      weight = f.color.slice(),
      skip = target,
      sign = side === 'front' ? 1 : -1;
    for (const goal of via) {
      const dir = cosineDirection(f, sign),
        [hit, dist, face] = trace(p, dir, skip);
      if (hit !== goal) {
        skip = -1;
        break;
      }
      p = advance(p, dir, dist);
      skip = hit;
      f = face;
      sign = dot(f.normal, dir) < 0 ? 1 : -1;
      weight = weight.map((v, i) => v * f.color[i]);
    }
    if (skip !== wall) continue;
    wallHits++;
    const l = room.lightPosition.map((v, j) => v - p[j]),
      l2 = dot(l, l),
      m = sign * dot(f.normal, l);
    if (m > 0 && !trace(p, l, wall, true)) deposit(0, weight, m / (l2 * Math.sqrt(l2)));
    if (!area) continue;
    const s = solid(),
      d = s.point.map((v, i) => v - p[i]),
      L = room.lightPosition.map((v, i) => v - s.point[i]),
      d2 = dot(d, d),
      r2 = dot(L, L),
      cWall = sign * dot(f.normal, d),
      cSlat = -dot(s.normal, d),
      cLamp = dot(s.normal, L);
    if (cWall <= 0 || cSlat <= 0 || cLamp <= 0 || trace(p, d, wall, true) || trace(s.point, L, s.id, true))
      continue;
    deposit(
      1,
      weight.map((v, i) => v * s.color[i]),
      ((((area / Math.PI) * cWall * cSlat) / (d2 * d2)) * cLamp) / (r2 * Math.sqrt(r2)),
    );
  }
  const mean = sum.map((v) => v.map((x) => x / paths)),
    se = squares.map((v, k) =>
      v.map((x, c) => Math.sqrt(Math.max(0, x / paths - mean[k][c] ** 2) / (paths - 1))),
    );
  return {
    path,
    paths,
    arrivals: wallHits,
    fromLamp: { mean: mean[0], standardError: se[0] },
    fromRoundedSurfaces: { mean: mean[1], standardError: se[1] },
  };
}
