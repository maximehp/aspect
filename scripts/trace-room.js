// Browser-side audit of two additive light paths through the playable operator.
// Muting a material isolates the first reflected source while retaining every
// occluder. All later handoffs use the operator from the original colored pose.

export async function traceRoom(renderer, scene) {
  const solution = await renderer.prepareScene(scene);
  const { modal, geometry } = solution,
    { basis } = modal;
  const { shapes, receivers } = geometry;
  const objectId = (name) => shapes.findIndex((s) => s.id === name);
  const indices = (object) => new Set(receivers.flatMap((p, i) => (p.object === object ? [i] : [])));
  const wall = objectId('north-wall'),
    floor = objectId('floor'),
    ceiling = objectId('ceiling');
  function outgoing(field, object) {
    const v = new Float64Array(field.length),
      rho = shapes[object].color;
    for (const i of indices(object)) for (let c = 0; c < 3; c++) v[3 * i + c] = rho[c] * field[3 * i + c];
    return v;
  }
  // Return pi times outgoing radiance, consistent with the independent room probe.
  function witness(field, object, side) {
    const chart = geometry.maps[object][side];
    const samples = [
      [0.29, 0.4],
      [0.5, 0.5],
      [0.69, 0.57],
    ];
    const rgb = samples.map(([u, v]) =>
      basis.at(chart, u, v, field).map((x, c) => x * shapes[object].color[c]),
    );
    return { samples, rgb };
  }
  function expanded(source) {
    const out = new Float64Array(source.length);
    for (const p of solution.rows)
      for (let r = 0; r + 1 < p.starts.length; r++) {
        const i = 3 * (p.first + r * p.stride);
        for (let k = p.starts[r]; k < p.starts[r + 1]; k++) {
          const j = 3 * p.col[k],
            w = p.weights[k];
          for (let c = 0; c < 3; c++) out[i + c] += w * source[j + c];
        }
      }
    return out;
  }
  const branches = {};
  for (const branch of ['slat emission', 'lit wall around slat']) {
    const choose = (o) => (branch === 'slat emission' ? !!o.axes : o.id === 'north-wall');
    const isolate = (o) => (choose(o) ? o : { ...o, color: [0, 0, 0] });
    const masked = await renderer.prepareScene({
      ...scene,
      rectangles: scene.rectangles.map(isolate),
      roundedBoxes: scene.roundedBoxes.map(isolate),
    });
    let field = basis.encode(masked.firstBounce),
      ref = masked.firstBounce;
    const records = [];
    let src = branch === 'slat emission' ? wall : floor;
    records.push({
      arrivedAt: shapes[src].id,
      activeModes: [...indices(src)].filter((i) => modal.active[i]).length,
      ...witness(field, src, src === floor ? 1 : 0),
    });
    for (const dest of branch === 'slat emission' ? [floor, ceiling] : [ceiling]) {
      field = modal.transfer(outgoing(field, src));
      ref = expanded(outgoing(ref, src));
      const at = witness(field, dest, dest === floor ? 1 : 0),
        match = witness(basis.encode(ref), dest, dest === floor ? 1 : 0);
      records.push({
        from: shapes[src].id,
        to: shapes[dest].id,
        ...modal.inspectLinks(indices(dest), indices(src)),
        ...at,
        expandedRGB: match.rgb,
      });
      src = dest;
    }
    branches[branch] = records;
  }
  const before = JSON.stringify([solution.counts, modal.counts]),
    chart = geometry.maps[floor][1];
  let maximumDifference = 0;
  for (let i = 0; i < 1024; i++) {
    const u = ((i % 32) + 0.5) / 32,
      v = (Math.floor(i / 32) + 0.5) / 32,
      rgb = basis.at(chart, u, v, modal.coefficients);
    const [ix, iy] = [u, v].map((t, d) => {
      const stops = chart.knots[d];
      let l = 0;
      while (l + 2 < stops.length && t > stops[l + 1]) l++;
      return [l, (t - stops[l]) / (stops[l + 1] - stops[l])];
    });
    for (let c = 0; c < 3; c++) {
      let expected = 0;
      for (let a = 0; a < 2; a++)
        for (let b = 0; b < 2; b++)
          expected +=
            (a ? ix[1] : 1 - ix[1]) *
            (b ? iy[1] : 1 - iy[1]) *
            solution.incident[3 * chart.ids[(iy[0] + b) * (chart.cols + 1) + ix[0] + a] + c];
      maximumDifference = Math.max(maximumDifference, Math.abs(expected - rgb[c]));
    }
  }
  return {
    branches,
    floorQueries: 1024,
    queryCountersUnchanged: before === JSON.stringify([solution.counts, modal.counts]),
    maximumDifference,
    counts: modal.counts,
    spatial: solution.counts,
    geometry: {
      filletSurfaces: geometry.charts.filter((g) => shapes[g.object].axes).length,
      cylinder: geometry.charts.filter((g) => g.type === 1).length,
      sphere: geometry.charts.filter((g) => g.type === 2).length,
    },
    modalResidual: modal.modalResidual,
    physicalResidual: solution.residual,
    discreteSolutionBound: solution.equilibriumError,
  };
}
