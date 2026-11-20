// Compare the same final leg three ways: production rows, independent cosine
// sampling of the production source field, and the complete physical paths.
import { evaluateChart } from '../src/optical-operations.js';

export function sourceExchangeProbe(snapshot, room, observation, paths) {
  const { source, charts } = snapshot;
  if (
    JSON.stringify(snapshot.scene) !== JSON.stringify(room) ||
    JSON.stringify(snapshot.observation) !== JSON.stringify(observation)
  )
    throw Error('Source field and observation must come from this exact scene and receiver region');
  const id = room.rectangles.findIndex((s) => s.id === source);
  if (id < 0) throw Error('Expected a planar emitting surface');
  const bin = () => ({ sum: [0, 0, 0], squares: [0, 0, 0] });
  const collect = () => Array.from({ length: 3 }, bin);
  const checked = { sampledField: collect(), physical: collect(), pairedDifference: collect() };
  const moments = Array.from({ length: 4 }, bin);
  let hits = 0,
    sampled = null;
  function add(bin, rgb, weight = 1) {
    rgb.forEach((v, i) => {
      const x = v * weight;
      bin.sum[i] += x;
      bin.squares[i] += x * x;
    });
  }
  function estimate(bin) {
    const mean = bin.sum.map((x) => x / paths);
    return {
      mean,
      standardError: mean.map((x, c) => Math.sqrt(Math.max(0, bin.squares[c] / paths - x * x) / (paths - 1))),
    };
  }
  return {
    firstHit(i, position, face, sign, weight) {
      if (i !== id) return;
      const chart = charts[sign > 0 ? 0 : 1],
        p = position.map((x, k) => x - face.center[k]);
      const coordinate = (axis) => 0.5 + 0.5 * p.reduce((s, x, k) => s + x * axis[k], 0);
      sampled = chart.fields.map((field) =>
        evaluateChart(chart, coordinate(face.dualU), coordinate(face.dualV), field).map(
          (v, c) => v * weight[c] * face.color[c],
        ),
      );
      hits++;
    },
    finishPath(byDepth, u, v) {
      if (!sampled) return;
      const reference = [
        byDepth[1].path,
        byDepth[2].path,
        byDepth[1].path.map((x, c) => x + byDepth[2].path[c] + byDepth[3].path[c]),
      ];
      sampled.forEach((rgb, i) => {
        add(checked.sampledField[i], rgb);
        add(checked.physical[i], reference[i]);
        add(
          checked.pairedDifference[i],
          rgb.map((x, c) => x - reference[i][c]),
        );
      });
      const [l, h, b, t] = observation.region;
      const x = ((u + 1) / 2 - l) / (h - l),
        y = ((v + 1) / 2 - b) / (t - b);
      [(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y].forEach((w, i) => add(moments[i], sampled[2], w));
      sampled = null;
    },
    result() {
      return {
        source,
        hits,
        paths,
        production: snapshot.production,
        measured: ['second reflection', 'third reflection', 'all reflections from second onward'],
        ...Object.fromEntries(Object.entries(checked).map(([k, v]) => [k, v.map(estimate)])),
        sampledReceiverMoments: moments.map(estimate),
      };
    },
  };
}
