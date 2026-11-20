// Browser-side physical comparison for a named path over a finite receiver.
// Integrate the playable bilinear field over the same area as the path sampler.
import { PlaneComposition } from '../src/plane-composition.js';

function checkRegion(region) {
  if (
    !Array.isArray(region) ||
    region.length !== 4 ||
    !region.every((x) => Number.isFinite(x) && x >= 0 && x <= 1) ||
    region[0] >= region[1] ||
    region[2] >= region[3]
  )
    throw RangeError('Invalid receiver region');
}
function average(chart, bounds, field) {
  const [u, v] = chart.knots,
    mean = [0, 0, 0];
  const area = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]);
  for (let j = 0; j < chart.rows; j++)
    for (let i = 0; i < chart.cols; i++) {
      const a = Math.max(u[i], bounds[0]),
        b = Math.min(u[i + 1], bounds[1]),
        l = Math.max(v[j], bounds[2]),
        h = Math.min(v[j + 1], bounds[3]);
      if (b <= a || h <= l) continue;
      const s = ((a + b) / 2 - u[i]) / (u[i + 1] - u[i]),
        t = ((h + l) / 2 - v[j]) / (v[j + 1] - v[j]);
      for (let y = 0; y < 2; y++)
        for (let x = 0; x < 2; x++) {
          const k = chart.ids[(j + y) * (chart.cols + 1) + i + x],
            weight = ((b - a) * (h - l) * (x ? s : 1 - s) * (y ? t : 1 - t)) / area;
          for (let c = 0; c < 3; c++) mean[c] += weight * field[3 * k + c];
        }
    }
  return mean;
}
// Compare two projections to composing the actual kernels through the region.
function projected(moments, weights, color) {
  const field = moments.map((_, i) =>
    [0, 1, 2].map((c) =>
      moments.reduce((sum, m, j) => sum + ((i ^ j) & 1 ? -2 : 4) * ((i ^ j) & 2 ? -2 : 4) * m[c], 0),
    ),
  );
  return {
    weights,
    field,
    rgb: color.map((r, c) => r * weights.reduce((sum, w, k) => sum + w * field[k][c], 0)),
    constantRGB: color.map(
      (r, c) => r * weights.reduce((s, w) => s + w, 0) * moments.reduce((s, m) => s + m[c], 0),
    ),
  };
}
export async function comparePlanePath(
  renderer,
  scene,
  {
    wall = 'north-wall',
    wallSide = 0,
    floor = 'floor',
    floorSide = 1,
    region = [0.5, 0.75, 0.4, 0.6],
    observers = [],
    panels,
    tolerance,
  } = {},
) {
  const solution = await renderer.prepareScene(scene),
    { geometry, modal } = solution;
  const middle = geometry.shapes.findIndex((s) => s.id === wall),
    dest = geometry.shapes.findIndex((s) => s.id === floor);
  if (!geometry.shapes[middle]?.axisU || !geometry.shapes[dest]?.axisU)
    throw TypeError('Choose two authored sheets in the scene');
  const chart = geometry.maps[middle][wallSide],
    receiver = geometry.maps[dest][floorSide],
    owner = geometry.shapes[dest];
  if (chart?.type !== 0 || receiver?.type !== 0)
    throw RangeError('Choose populated sides of the intermediate sheet and receiver');
  const [l, h, b, t] = region;
  checkRegion(region);
  const target = {
    owner,
    n: receiver.n,
    center: owner.center.map((x, k) => x + (l + h - 1) * owner.axisU[k] + (b + t - 1) * owner.axisV[k]),
    axisU: owner.axisU.map((x) => x * (h - l)),
    axisV: owner.axisV.map((x) => x * (t - b)),
  };
  const operator = await PlaneComposition.create(renderer.gl, geometry, chart, target, scene.roundedBoxes);
  const next = observers.map(({ id, side = 0, uv = [0.5, 0.5] }) => {
    const object = geometry.shapes.findIndex((s) => s.id === id),
      shape = geometry.shapes[object];
    if (!shape?.axisU || !geometry.maps[object][side]?.n) throw Error('Expected a planar observer side');
    return {
      object,
      n: geometry.maps[object][side].n,
      p: shape.center.map((c, d) => c + (2 * uv[0] - 1) * shape.axisU[d] + (2 * uv[1] - 1) * shape.axisV[d]),
    };
  });
  let composed;
  try {
    composed = panels ? panels.map((p) => operator.integrate(p, next)) : operator.solve(tolerance, next);
    const aperture = next.map((receiver) => operator.receiverWeights(receiver));
    for (const s of Array.isArray(composed) ? composed : composed.estimates)
      if (next.length)
        s.followingReflection = next.map((receiver, k) => {
          const shade = (m) => projected(m, aperture[k], geometry.shapes[receiver.object].color);
          return { direct: shade(s.directMoments), reflected: shade(s.reflectedMoments) };
        });
  } finally {
    operator.dispose();
  }
  async function isolated(keep) {
    const mask = (s) => (keep(s) ? s : { ...s, color: [0, 0, 0] });
    return (
      await renderer.prepareScene({
        ...scene,
        rectangles: scene.rectangles.map(mask),
        roundedBoxes: scene.roundedBoxes.map(mask),
      })
    ).firstBounce;
  }
  const direct = await isolated((s) => s === geometry.shapes[middle]),
    slat = await isolated((s) => !!s.axes);
  const outgoing = new Float64Array(slat.length);
  // The production modal coefficients encode illumination in the same
  // object-local basis used by the playable equilibrium.
  const encoded = modal.basis.encode(slat);
  for (let i = 0; i < geometry.receivers.length; i++)
    if (geometry.receivers[i].object === middle) {
      for (let c = 0; c < 3; c++) outgoing[3 * i + c] = encoded[3 * i + c] * geometry.shapes[middle].color[c];
    }
  const reflected = modal.basis.decode(modal.transfer(outgoing));
  const observe = (f) => average(receiver, region, f).map((x, c) => x * owner.color[c]);
  return {
    region,
    composed,
    playable: { directWallRGB: observe(direct), reflectedWallRGB: observe(reflected) },
  };
}

// Isolate the first three reflected orders in the playable operator. The
// remainder includes all later orders and the production solver's residual.
export async function reflectedOrders(renderer, scene, observations) {
  const { geometry, modal, firstBounce, incident } = await renderer.prepareScene(scene);
  const orders = [firstBounce];
  let coefficients = modal.firstCoefficients;
  for (let order = 2; order <= 3; order++) {
    const outgoing = coefficients.map(
      (value, i) => value * geometry.shapes[geometry.receivers[Math.floor(i / 3)].object].color[i % 3],
    );
    coefficients = modal.transfer(outgoing, false);
    orders.push(modal.basis.decode(coefficients));
  }
  return observations.map(({ id, side, region }) => {
    const object = geometry.shapes.findIndex((s) => s.id === id),
      chart = geometry.maps[object]?.[side];
    checkRegion(region);
    if (!chart?.n || chart.type !== 0) throw Error('Expected a planar receiver side');
    const sample = (field) =>
      average(chart, region, field).map((v, c) => v * geometry.shapes[object].color[c]);
    const rgb = orders.map(sample),
      total = sample(incident);
    rgb.push(total.map((v, c) => v - rgb.reduce((sum, x) => sum + x[c], 0)));
    return { id, side, region, orders: rgb, total };
  });
}

// Portable source fields for the independent reference. Multiplying by a
// source reflectance and the receiving material happens once, on each path.
export async function captureSourceField(renderer, scene, source, observation) {
  const { geometry, firstBounce, incident, modal, rows } = await renderer.prepareScene(scene);
  const { shapes, receivers, maps } = geometry,
    from = shapes.findIndex((s) => s.id === source),
    to = shapes.findIndex((s) => s.id === observation.id);
  if (!shapes[from]?.axisU || !shapes[to]?.axisU || !['front', 'back'].includes(observation.side))
    throw Error('Select two authored sheets and a receiving side');
  checkRegion(observation.region);
  const target = maps[to][observation.side === 'front' ? 0 : 1];
  if (target.type !== 0) throw Error('Select a populated receiver side');
  const outgoing = modal.firstCoefficients.map(
    (x, i) => x * shapes[receivers[Math.floor(i / 3)].object].color[i % 3],
  );
  const fields = [firstBounce, modal.basis.decode(modal.transfer(outgoing, false)), incident];
  const charts = maps[from].map((chart) => ({
    cols: chart.cols,
    rows: chart.rows,
    knots: chart.knots,
    ids: chart.ids.map((id, i) => (id < 0 ? -1 : i)),
    fields: fields.map((field) =>
      chart.ids.flatMap((id) => (id < 0 ? [0, 0, 0] : Array.from(field.subarray(3 * id, 3 * id + 3)))),
    ),
  }));
  const production = fields.map((field) => {
    const result = new Float64Array(field.length);
    for (const i of target.ids) {
      const part = rows[i % rows.length],
        row = (i - part.first) / part.stride;
      for (let j = part.starts[row]; j < part.starts[row + 1]; j++) {
        const id = part.col[j];
        if (receivers[id].object !== from) continue;
        for (let c = 0; c < 3; c++)
          result[3 * i + c] += part.weights[j] * field[3 * id + c] * shapes[from].color[c];
      }
    }
    return average(target, observation.region, result).map((x, c) => x * shapes[to].color[c]);
  });
  return { scene, source, observation, charts, production };
}
