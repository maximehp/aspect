// Local coordinates (height, U, V) of a point on an authored plane relative
// to each finite blocking sheet. Each coordinate is affine in both plane
// variables. The same table serves either end of a reflection.
import { dot, sub } from './cpu-geometry.js';
export function sheetBoundaryForms(surface, records) {
  const { charts, shapes } = surface;
  const data = Array(charts.length).fill(0);
  for (const g of charts) {
    if (g.type || g.empty) continue;
    data[g.offset] = data.length;
    for (let i = 0; i < shapes.length; i++) {
      const s = shapes[i];
      if (s.axes) {
        data.push(0, 0, 0, 0, 0, 0, 0, 0, 0);
        continue;
      }
      const delta = sub(g.c, s.center);
      for (const k of [9, 3, 6]) {
        const dual = records.subarray(19 * i + k, 19 * i + k + 3);
        data.push(dot(delta, dual), dot(g.ax, dual), dot(g.v, dual));
      }
    }
  }
  return new Float64Array(data);
}
