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
  const region = path.region;
  function checkRegion(bounds) {
    if (
      bounds &&
      (bounds.length !== 4 ||
        !bounds.every((x) => Number.isFinite(x) && x >= 0 && x <= 1) ||
        bounds[0] >= bounds[1] ||
        bounds[2] >= bounds[3])
    )
      throw RangeError('Expected increasing U and V bounds in the unit square');
  }
  checkRegion(region);
  const sum = [
      [0, 0, 0],
      [0, 0, 0],
    ],
    squares = [
      [0, 0, 0],
      [0, 0, 0],
    ];
  const collect = path.receiverMoments === true;
  if (collect && !region) throw RangeError('Receiver moments require a finite region');
  const moments = Array.from({ length: 2 }, () => Array.from({ length: 4 }, () => [0, 0, 0]));
  const momentSquares = moments.map((m) => m.map((row) => row.slice()));
  const observers = (path.observers || []).map(({ id, side = 0, uv = [0.5, 0.5], region: observation }) => {
    const face = faces.find((f) => f.id === id);
    if (!region || !face || (side !== 0 && side !== 1))
      throw Error('Expected a receiver region and a planar observer');
    checkRegion(observation);
    const position = (u, v) => advance(advance(face.center, face.axisU, 2 * u - 1), face.axisV, 2 * v - 1);
    return {
      sample:
        observation &&
        (() =>
          position(
            observation[0] + random() * (observation[1] - observation[0]),
            observation[2] + random() * (observation[3] - observation[2]),
          )),
      color: face.color,
      normal: scale(face.normal, side ? -1 : 1),
      p: position(...uv),
      sum: Array.from({ length: 2 }, () => [0, 0, 0]),
      squares: Array.from({ length: 2 }, () => [0, 0, 0]),
    };
  });
  const regionArea = region && 4 * receiver.jacobian * (region[1] - region[0]) * (region[3] - region[2]);
  let start,
    u = 0,
    v = 0;
  function deposit(k, throughput, factor) {
    for (const o of observers)
      if (o.factor === undefined) {
        const d = (o.sample ? o.sample() : o.p).map((x, i) => x - start[i]),
          r2 = dot(d, d);
        const cosA = Math.max(0, (side === 'front' ? 1 : -1) * dot(receiver.normal, d)),
          cosB = Math.max(0, -dot(o.normal, d));
        o.factor =
          cosA * cosB > 0 && !trace(start, d, target, true)
            ? (regionArea * cosA * cosB) / (Math.PI * r2 * r2)
            : 0;
      }
    for (let c = 0; c < 3; c++) {
      let x = throughput[c] * room.lampRGB[c] * factor;
      sum[k][c] += x;
      squares[k][c] += x * x;
      for (const o of observers) {
        const value = x * o.factor * o.color[c];
        o.sum[k][c] += value;
        o.squares[k][c] += value * value;
      }
      if (collect)
        for (let j = 0; j < 4; j++) {
          const test = x * (j & 1 ? u : 1 - u) * (j & 2 ? v : 1 - v);
          moments[k][j][c] += test;
          momentSquares[k][j][c] += test * test;
        }
    }
  }
  let wallHits = 0;
  for (let it = 0; it < paths; it++) {
    if (region) {
      u = random();
      v = random();
    }
    let p = region
        ? advance(
            advance(receiver.center, receiver.axisU, 2 * (region[0] + u * (region[1] - region[0])) - 1),
            receiver.axisV,
            2 * (region[2] + v * (region[3] - region[2])) - 1,
          )
        : origin,
      f = receiver,
      weight = f.color.slice(),
      skip = target,
      sign = side === 'front' ? 1 : -1;
    start = p;
    for (const o of observers) o.factor = undefined;
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
    ...(collect && {
      receiverMoments: moments.map((m, k) =>
        m.map((row, j) => ({
          mean: row.map((x) => x / paths),
          standardError: row.map((x, c) =>
            Math.sqrt(Math.max(0, momentSquares[k][j][c] / paths - (x / paths) ** 2) / (paths - 1)),
          ),
        })),
      ),
    }),
    ...(observers.length && {
      followingReflection: observers.map((o) =>
        o.sum.map((row, k) => ({
          mean: row.map((x) => x / paths),
          standardError: row.map((x, c) =>
            Math.sqrt(Math.max(0, o.squares[k][c] / paths - (x / paths) ** 2) / (paths - 1)),
          ),
        })),
      ),
    }),
    fromLamp: { mean: mean[0], standardError: se[0] },
    fromRoundedSurfaces: { mean: mean[1], standardError: se[1] },
  };
}
