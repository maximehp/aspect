// Independent CPU reference over a chosen area on a planar receiver. The
// integration includes both planar sheets and the actual filleted slat solids.
import { buildRoom } from '../src/room.js';
import { hitRoundedBox } from '../src/rounded-box.js';

const [id = 'bench-seat-bottom', u = '0', v = '0', side = 'front', degrees = '24', count = '524288'] = process.argv.slice(2);
const iu = Number(u), iv = Number(v), angle = Number(degrees), paths = Number(count);
if (!['front', 'back'].includes(side) || !Number.isFinite(angle) || !Number.isSafeInteger(paths) || paths < 2) {
  throw new Error('Expected a front/back side, a finite blind angle and at least two paths.');
}
// The optional final JSON argument accepts the same authored room edits as play.
const edits = process.argv[8] ? JSON.parse(process.argv[8]) : {};
const room = buildRoom({ ...edits, angle: angle*Math.PI/180 });
const target = room.rectangles.findIndex(r => r.id === id);
if (target < 0) throw new Error('Unknown planar receiver: ' + id);
// Keep the original observation areas comparable when the renderer changes.
const bins = axis => Math.min(5, Math.max(1, Math.ceil(Math.hypot(...axis))));
const grid = { nu: bins(room.rectangles[target].axisU), nv: bins(room.rectangles[target].axisV) };
if (![iu, iv].every(Number.isInteger) || iu < 0 || iu >= grid.nu || iv < 0 || iv >= grid.nv) {
  throw new Error('Area index outside the ' + grid.nu + ' × ' + grid.nv + ' surface grid.');
}

const dot = (a,b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a,b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const scale = (a,s) => a.map(x => x*s);
const unit = a => scale(a, 1/Math.hypot(...a));
const advance = (a,b,s) => a.map((x,k) => x + s*b[k]);
const faces = room.rectangles.map(rect => {
  const area = cross(rect.axisU,rect.axisV), normal = unit(area), tangent = unit(rect.axisU);
  const jacobian = Math.hypot(...area);
  return { ...rect, normal, tangent, bitangent: cross(normal,tangent),
    dualU: scale(cross(rect.axisV,normal),1/jacobian), dualV: scale(cross(normal,rect.axisU),1/jacobian) };
});

// Quickly reject closed solids before evaluating the analytic fillet.
function intersectsBounds({min,max}, origin, direction, maximum) {
  let start = 0, end = maximum;
  for (let k=0; k<3; k++) {
    if (direction[k] === 0) {
      if (origin[k]<min[k] || origin[k]>max[k]) return false;
      continue;
    }
    const a = (min[k]-origin[k])/direction[k], b = (max[k]-origin[k])/direction[k];
    start = Math.max(start,Math.min(a,b));
    end = Math.min(end,Math.max(a,b));
    if (start>end) return false;
  }
  return true;
}
function trace(origin, direction, skip, shadow = false) {
  let hit = -1, distance = shadow ? 1-1e-10 : Infinity;
  for (let i=0; i<faces.length; i++) {
    if (i === skip) continue;
    const f = faces[i], den = dot(f.normal,direction);
    if (den === 0) continue;
    const x = f.center[0]-origin[0], y = f.center[1]-origin[1], z = f.center[2]-origin[2];
    const t = (x*f.normal[0] + y*f.normal[1] + z*f.normal[2])/den;
    if (!(t > 1e-10 && t < distance)) continue;
    const a = t*direction[0]-x, b = t*direction[1]-y, c = t*direction[2]-z;
    const inside = dual => Math.abs(a*dual[0] + b*dual[1] + c*dual[2]) <= 1;
    if (!inside(f.dualU) || !inside(f.dualV)) continue;
    if (shadow) return true;
    hit = i;
    distance = t;
  }
  let curvedFace;
  for (let j=0; j<room.roundedBoxes.length; j++) {
    const box = room.roundedBoxes[j], i = faces.length+j;
    if (i===skip || !intersectsBounds(box.bounds,origin,direction,distance)) continue;
    const result = hitRoundedBox(box,origin,direction,1e-10,distance);
    if (!result) continue;
    if (shadow) return true;
    const normal = result.normal;
    const tangent = unit(cross(Math.abs(normal[0]) < 0.8 ? [1,0,0] : [0,1,0],normal));
    hit = i;
    distance = result.t;
    curvedFace = { normal, tangent, bitangent: cross(normal,tangent), color: box.color };
  }
  return shadow ? false : [hit, distance, curvedFace || faces[hit]];
}

let state = 123456789;
function random() {
  state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
  return ((state >>> 0) + 0.5)/4294967296;
}
if (process.argv[9]) {
  const { referencePath } = await import('./reference-path.js');
  const path = JSON.parse(process.argv[9]);
  console.log(JSON.stringify(referencePath({room,faces,target,side,path,paths,trace,random,dot,scale,advance})));
  process.exit(0);
}

const first = [0,0,0], later = [0,0,0], sumSquares = [0,0,0];
for (let sample=0; sample<paths; sample++) {
  let i = target, f = faces[i], sign = side === 'front' ? 1 : -1;
  const u = 2*(iu+random())/grid.nu-1, v = 2*(iv+random())/grid.nv-1;
  let position = advance(advance(f.center,f.axisU,u),f.axisV,v);
  const weight = [1,1,1], energy = [0,0,0];
  for (let depth=0; ; depth++) {
    if (depth > 0) {
      const d = room.lightPosition.map((x,k) => x-position[k]), r2 = dot(d,d);
      const cosine = sign*dot(d,f.normal)/Math.sqrt(r2);
      if (cosine > 0 && !trace(position,d,i,true)) {
        for (let c=0; c<3; c++) {
          const value = weight[c]*f.color[c]*room.lampRGB[c]*cosine/r2;
          (depth === 1 ? first : later)[c] += value;
          energy[c] += value;
        }
      }
    }
    for (let c=0; c<3; c++) weight[c] *= f.color[c];
    if (depth >= 2) {
      const survival = Math.min(.95, Math.max(...f.color));
      if (random() >= survival) break;
      for (let c=0; c<3; c++) weight[c] /= survival;
    }
    const r2 = random(), phi = 2*Math.PI*random(), r = Math.sqrt(r2);
    const direction = advance(advance(scale(f.tangent,r*Math.cos(phi)),f.bitangent,r*Math.sin(phi)),
      f.normal,sign*Math.sqrt(1-r2));
    const [next, distance, nextFace] = trace(position,direction,i);
    if (next < 0) break;
    position = advance(position,direction,distance);
    i = next;
    f = nextFace;
    sign = dot(f.normal,direction) < 0 ? 1 : -1;
  }
  for (let c=0; c<3; c++) sumSquares[c] += energy[c]*energy[c];
}

const reference = first.map((value,c) => (value+later[c])/paths);
const standardError = reference.map((mean,c) => Math.sqrt(Math.max(0,sumSquares[c]/paths-mean*mean)/(paths-1)));
console.log(JSON.stringify({ id, iu, iv, side, angle, paths, ...(process.argv[8] ? { edits } : {}), reference, standardError,
  firstReflected: scale(first,1/paths), laterReflected: scale(later,1/paths) }, null, 2));
