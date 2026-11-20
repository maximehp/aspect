// Reproduce the unoccluded pair integral independently of the equilibrium solve.
import { buildRoom } from '../src/room.js';
import { compileSurfaces } from '../src/surface-functions.js';
import { parallelBilinearExchange } from '../src/rectangle-exchange.js';

const [sourceName = 'print-paper', receiverName = 'north-wall'] = process.argv.slice(2);
const geometry = compileSurfaces(buildRoom({ angle: (24 * Math.PI) / 180 }));
const shapes = [sourceName, receiverName].map((name) => geometry.shapes.find((s) => s.id === name));
if (shapes.some((s) => !s || s.axes)) throw Error('Expected two room rectangles');
const axes = shapes[0].axisU.map((_, k) => k).filter((k) => shapes[0].axisU[k] || shapes[0].axisV[k]);
if (
  axes.length !== 2 ||
  shapes.some((s) => [s.axisU, s.axisV].some((a) => a.filter((x) => x !== 0).length !== 1))
)
  throw Error('Expected axis-aligned planes');
const normal = 3 - axes[0] - axes[1];
if (shapes[1].axisU[normal] || shapes[1].axisV[normal]) throw Error('Expected parallel planes');
const gap = Math.abs(shapes[0].center[normal] - shapes[1].center[normal]);
const bounds = shapes.map((s) =>
  axes.map((k) => {
    const c = s.center[k],
      h = Math.abs(s.axisU[k]) + Math.abs(s.axisV[k]);
    return [c - h, c + h];
  }),
);
const [source, receiver] = bounds;
const kernel = parallelBilinearExchange(gap),
  matrix = kernel.grid(source, receiver);

// At a fixed y, integrate the four bilinear source hats via independent
// primitives for (1,x,y,xy) K. x and y are displacements from the receiver.
function fromSource(point) {
  const [w, h] = source.map(([a, b]) => b - a);
  const u0 = (point[0] - source[0][0]) / w,
    v0 = (point[1] - source[1][0]) / h;
  let mass = 0,
    xMass = 0,
    yMass = 0,
    xyMass = 0;
  for (let i = 0; i < 2; i++)
    for (let j = 0; j < 2; j++) {
      const x = source[0][i] - point[0],
        y = source[1][j] - point[1],
        a = Math.hypot(gap, x),
        b = Math.hypot(gap, y);
      const angleX = Math.atan(x / b),
        angleY = Math.atan(y / a),
        sign = i === j ? 1 : -1;
      mass += (sign * ((x / a) * angleY + (y / b) * angleX)) / (2 * Math.PI);
      xMass -= (sign * gap * gap * angleY) / (2 * Math.PI * a);
      yMass -= (sign * gap * gap * angleX) / (2 * Math.PI * b);
      xyMass -= (sign * gap * gap * Math.log(x * x + y * y + gap * gap)) / (4 * Math.PI);
    }
  const u = u0 * mass + xMass / w,
    v = v0 * mass + yMass / h,
    uv = u0 * v0 * mass + (xMass * v0) / w + (yMass * u0) / h + xyMass / (w * h);
  return [mass - u - v + uv, u - uv, v - uv, uv];
}

const chart = geometry.maps[geometry.shapes.indexOf(shapes[1])].find(
  (g) => g.n?.[normal] * (shapes[0].center[normal] - shapes[1].center[normal]) > 0,
);
const nodal = chart.ids.map((id) =>
  fromSource(axes.map((k) => geometry.receivers[id].p[k])).reduce((a, b) => a + b, 0),
);
const chartAxes = [chart.ax, chart.v].map((v) => axes.findIndex((k) => v[k] !== 0));
const positions = chartAxes.map((k, d) =>
  chart.knots[d].map(
    (u) =>
      chart.c[axes[k]] + (2 * u - 1) * (d ? chart.height : chart.width) * (d ? chart.v : chart.ax)[axes[k]],
  ),
);
function nodalField(point) {
  const cell = positions.map((knots, d) => {
    const x = point[chartAxes[d]],
      sign = Math.sign(knots.at(-1) - knots[0]);
    let i = 0;
    while (i + 2 < knots.length && sign * (x - knots[i + 1]) > 0) i++;
    return [i, (x - knots[i]) / (knots[i + 1] - knots[i])];
  });
  const [[i, u], [j, v]] = cell,
    k = i + j * (chart.cols + 1);
  return (
    (1 - v) * ((1 - u) * nodal[k] + u * nodal[k + 1]) +
    v * ((1 - u) * nodal[k + chart.cols + 1] + u * nodal[k + chart.cols + 2])
  );
}

function quadrature(axis, split = false) {
  const [lo, hi] = receiver[axis],
    cuts = new Set([lo, hi, ...positions[chartAxes.indexOf(axis)]]);
  for (const x of source[axis]) {
    if (lo < x && x < hi) cuts.add(x);
    for (let d = gap; d < Math.max(Math.abs(hi - x), Math.abs(lo - x)); d *= 2) {
      for (const p of [x - d, x + d]) if (lo < p && p < hi) cuts.add(p);
    }
  }
  if (split) {
    const sorted = [...cuts].sort((a, b) => a - b);
    for (let i = 1; i < sorted.length; i++) cuts.add((sorted[i] + sorted[i - 1]) / 2);
  }
  const knots = [...cuts].sort((a, b) => a - b),
    out = [];
  const nodes = [0.1834346424956498, 0.525532409916329, 0.7966664774136267, 0.9602898564975363];
  const weights = [0.362683783378362, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763];
  for (let i = 1; i < knots.length; i++) {
    const mid = (knots[i - 1] + knots[i]) / 2,
      half = (knots[i] - knots[i - 1]) / 2;
    for (let j = 0; j < 4; j++)
      for (const sign of [-1, 1]) out.push([mid + sign * half * nodes[j], half * weights[j]]);
  }
  return out;
}
function integrate(split) {
  const audit = new Float64Array(16),
    returned = new Float64Array(16),
    xs = quadrature(0, split),
    ys = quadrature(1, split);
  let nodalReturn = 0;
  for (const [x, wx] of xs)
    for (const [y, wy] of ys) {
      const f = fromSource([x, y]),
        u = (x - receiver[0][0]) / (receiver[0][1] - receiver[0][0]),
        v = (y - receiver[1][0]) / (receiver[1][1] - receiver[1][0]);
      const hats = [(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v];
      const aperture = f.reduce((a, b) => a + b, 0);
      nodalReturn += wx * wy * aperture * nodalField([x, y]);
      for (let i = 0; i < 4; i++)
        for (let j = 0; j < 4; j++) {
          audit[4 * i + j] += wx * wy * hats[i] * f[j];
          returned[4 * i + j] += wx * wy * f[i] * f[j];
        }
    }
  return { audit, returned, nodalReturn, count: xs.length * ys.length };
}
const coarse = integrate(false),
  { audit, returned, nodalReturn, count } = integrate(true);
let interpolated = 0;
for (const tile of geometry.tiles.filter((t) => t.offset === chart.offset))
  for (const id of tile.nodes) {
    const p = geometry.receivers[id].p;
    interpolated += tile.width * tile.height * fromSource(axes.map((k) => p[k])).reduce((a, b) => a + b, 0);
  }
const reverse = parallelBilinearExchange(gap).grid(receiver, source);
console.log(
  JSON.stringify(
    {
      sourceName,
      receiverName,
      gap,
      endpointEvaluations: kernel.evaluations,
      independentPoints: count,
      returnThroughReceiver: {
        unitDiffuseResponse: true,
        retainedFunction: returned.reduce((a, b) => a + b, 0),
        interpolatedFunction: nodalReturn,
        bilinearWeights: [...returned],
        largestChangeAfterHalvingPanels: Math.max(
          ...returned.map((w, i) => Math.abs(w - coarse.returned[i])),
        ),
        interpolatedChangeAfterHalvingPanels: Math.abs(nodalReturn - coarse.nodalReturn),
      },
      bilinearWeights: [...matrix],
      maxAbsoluteMomentDifference: Math.max(...matrix.map((v, i) => Math.abs(v - audit[i]))),
      maxReciprocityDifference: Math.max(
        ...matrix.map((v, i) => Math.abs(v - reverse[4 * (i % 4) + (i >> 2)])),
      ),
      integratedExchange: matrix.reduce((a, b) => a + b, 0),
      integratedNodalInterpolant: interpolated,
    },
    null,
    2,
  ),
);
