// Browser-side physical comparison for a named path over a finite receiver.
// Integrate the playable bilinear field over the same area as the path sampler.
import { PlaneComposition } from '../src/plane-composition.js';

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
  if (![l, h, b, t].every((x) => Number.isFinite(x) && x >= 0 && x <= 1) || l >= h || b >= t)
    throw RangeError('Invalid receiver region');
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
