// A row gathers radiance from other patch sides. Side 2*p faces axisU × axisV.
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const scale = (a, s) => a.map(v => v*s);
const offset = (o, u, a, v, b) => o.map((c, i) => c + a*u[i] + b*v[i]);
const maxAxisCells = 5;
const patchBudget = 350;
const samplesPerSide = 256;

function makePatches(rectangles) {
  const frames = rectangles.map(({ id, center, axisU, axisV }) => {
    const areaVector = cross(axisU, axisV);
    const jacobian = Math.hypot(...areaVector);
    if (!(jacobian > 0) || !Number.isFinite(4*jacobian) || !center.every(Number.isFinite)) {
      throw new Error('Invalid diffuse surface geometry: ' + id);
    }
    const normal = scale(areaVector, 1/jacobian);
    const tangent = scale(axisU, 1/Math.hypot(...axisU));
    return { id, center, axisU, axisV, normal, tangent, bitangent: cross(normal, tangent),
      area: 4*jacobian, dualU: scale(cross(axisV, normal), 1/jacobian),
      dualV: scale(cross(normal, axisU), 1/jacobian) };
  });
  let pitch = 2;
  for (;;) {
    const count = axis => Math.min(maxAxisCells, Math.max(1, Math.ceil(2*Math.hypot(...axis)/pitch)));
    let total = 0;
    for (const frame of frames) {
      frame.nu = count(frame.axisU);
      frame.nv = count(frame.axisV);
      total += frame.nu * frame.nv;
    }
    if (total <= Math.max(patchBudget, frames.length)) break;
    pitch *= Math.max(1.25, Math.sqrt(total/patchBudget));
  }
  const patches = [];
  for (const [rectIndex, face] of frames.entries()) {
    face.index = rectIndex;
    const { nu, nv } = face;
    face.firstPatch = patches.length;
    face.stepU = scale(face.axisU, 1/nu);
    face.stepV = scale(face.axisV, 1/nv);
    for (let iv=0; iv<nv; iv++) for (let iu=0; iu<nu; iu++) {
      const center = offset(face.center, face.stepU, 2*iu+1-nu, face.stepV, 2*iv+1-nv);
      patches.push({ rectIndex, iu, iv, nu, nv, center, normal: face.normal, area: face.area/(nu*nv) });
    }
  }
  return { frames, patches };
}

function radicalInverse(n, base) {
  let value = 0, place = 1/base;
  while (n) {
    value += (n % base)*place;
    place /= base;
    n = Math.floor(n/base);
  }
  return value;
}

// A shifted Halton point selects two surface coordinates and a cosine-weighted
// hemisphere direction. The shifts depend on the authored id and local grid
// coordinates, so an unrelated edit or a reordered input cannot reseed a face.
const samples = Array.from({length: samplesPerSide}, (_, i) => {
  const phi = 2*Math.PI*radicalInverse(i, 7);
  return [radicalInverse(i, 2), radicalInverse(i, 3), radicalInverse(i, 5), Math.cos(phi), Math.sin(phi)];
});
function shifts(id, { iu, iv, nu, nv }, side) {
  let state = 2166136261;
  const key = JSON.stringify([id, iu, iv, nu, nv, side]);
  for (let i=0; i<key.length; i++) state = Math.imul(state^key.charCodeAt(i), 16777619);
  return Array.from({length: 4}, () => {
    state = (state + 0x6d2b79f5) | 0;
    let x = Math.imul(state ^ (state >>> 15), 1 | state);
    x ^= x + Math.imul(x ^ (x >>> 7), 61 | x);
    return (((x ^ (x >>> 14)) >>> 0) + 0.5) / 4294967296;
  });
}
const wrap = x => x >= 1 ? x-1 : x;

function candidates(frames, index, side) {
  const receiver = frames[index], n = receiver.normal, sign = side ? -1 : 1;
  return frames.filter(f => {
    if (f.index === index) return false;
    const x = f.center[0]-receiver.center[0], y=f.center[1]-receiver.center[1], z=f.center[2]-receiver.center[2];
    const distance = sign * (x*n[0]+y*n[1]+z*n[2]);
    const reach = Math.abs(dot(n,f.axisU)) + Math.abs(dot(n,f.axisV));
    return distance + reach > -32*Number.EPSILON*(Math.abs(x)+Math.abs(y)+Math.abs(z)+reach);
  });
}

function firstHit(frames, ox, oy, oz, dx, dy, dz) {
  let nearest = Infinity, hitSide = -1, hitFrame;
  for (let i=0; i<frames.length; i++) {
    const f = frames[i], n = f.normal;
    const cx = f.center[0]-ox, cy = f.center[1]-oy, cz = f.center[2]-oz;
    const hx = cx*n[0], hy = cy*n[1], hz = cz*n[2];
    const height = hx+hy+hz;
    // A ray starting on a touching or coplanar sheet can have a signed
    // height within dot-product rounding. Use the local coordinate scale.
    const heightTolerance = 32*Number.EPSILON*(Math.abs(hx)+Math.abs(hy)+Math.abs(hz));
    if (Math.abs(height) <= heightTolerance) continue;
    const facing = dx*n[0]+dy*n[1]+dz*n[2];
    const t = height/facing;
    if (!(t>0 && t<=nearest) || !Number.isFinite(t)) continue;
    if (t === nearest && String(f.id) >= String(hitFrame.id)) continue;
    const x = dx*t-cx, y = dy*t-cy, z = dz*t-cz;
    const u = x*f.dualU[0]+y*f.dualU[1]+z*f.dualU[2];
    if (Math.abs(u)>1) continue;
    const v = x*f.dualV[0]+y*f.dualV[1]+z*f.dualV[2];
    if (Math.abs(v)>1) continue;
    const iu = Math.min(f.nu-1, Math.floor((u+1)*f.nu/2));
    const iv = Math.min(f.nv-1, Math.floor((v+1)*f.nv/2));
    hitSide = 2*(f.firstPatch + iv*f.nu + iu) + Number(facing > 0);
    nearest = t;
    hitFrame = f;
  }
  return hitSide;
}

function sampleExchange(frames, patches) {
  const sideCount = 2*patches.length;
  const counts = Array.from({length: sideCount}, () => new Map());
  const escape = new Float64Array(sideCount);
  let escapedSamples = 0, raysCast = 0;
  const visibleFrames = frames.map((f,i)=>[candidates(frames,i,0),candidates(frames,i,1)]);
  for (let i=0; i<sideCount; i++) {
    const p = patches[i>>1], f = frames[p.rectIndex], sign = (i&1) ? -1 : 1;
    const possible = visibleFrames[p.rectIndex][i&1];
    if (!possible.length) {
      escape[i]=1;
      escapedSamples+=samplesPerSide;
      continue;
    }
    raysCast += samplesPerSide;
    const [uShift, vShift, rShift, phiShift] = shifts(f.id, p, i&1);
    const c = Math.cos(2*Math.PI*phiShift), s = Math.sin(2*Math.PI*phiShift);
    const { stepU, stepV, tangent, bitangent, normal } = f;
    let misses = 0;
    for (const sample of samples) {
      const u = 2*wrap(sample[0]+uShift)-1, v = 2*wrap(sample[1]+vShift)-1;
      const ox = p.center[0]+u*stepU[0]+v*stepV[0];
      const oy = p.center[1]+u*stepU[1]+v*stepV[1];
      const oz = p.center[2]+u*stepU[2]+v*stepV[2];
      const radial2 = wrap(sample[2]+rShift), r = Math.sqrt(radial2);
      const tx = r*(c*sample[3]-s*sample[4]), ty = r*(s*sample[3]+c*sample[4]);
      const tz = sign*Math.sqrt(1-radial2);
      const j = firstHit(possible, ox, oy, oz,
        tx*tangent[0]+ty*bitangent[0]+tz*normal[0], tx*tangent[1]+ty*bitangent[1]+tz*normal[1],
        tx*tangent[2]+ty*bitangent[2]+tz*normal[2]);
      if (j < 0) { misses++; continue; }
      const row = counts[Math.min(i,j)], other = Math.max(i,j);
      row.set(other, (row.get(other) ?? 0)+1);
    }
    escape[i] = misses/samplesPerSide;
    escapedSamples += misses;
  }
  const links = Array.from({length: sideCount}, () => []);
  let exchangeLinks = 0;
  for (let i=0; i<sideCount; i++) for (const [j, hits] of counts[i]) {
    const a = patches[i>>1].area, b = patches[j>>1].area;
    // The two hit counts estimate the same exchange area A_i F_ij = A_j F_ji.
    // Their combined expectation is N * area * (1/A_i + 1/A_j).
    const hitRate = hits/samplesPerSide;
    links[i].push([j, hitRate/(1+a/b)]);
    links[j].push([i, hitRate/(1+b/a)]);
    exchangeLinks++;
  }
  return { links, escape, escapedSamples, exchangeLinks, raysCast };
}

function balanceExchange(links, escape) {
  let scale = new Float64Array(links.length).fill(1), next = new Float64Array(links.length);
  let maxUncorrectedRowSum = 0;
  // Symmetric matrix scaling (one factor per side) preserves reciprocity.
  // An escaped ray contributes to a separate opening term with unit exterior
  // scale. This leaves genuine openings and completely isolated sides dark.
  for (let k=0; k<16; k++) {
    for (let i=0; i<links.length; i++) {
      let sum = 0;
      for (const [j,w] of links[i]) sum += w*scale[j];
      if (k===0) maxUncorrectedRowSum = Math.max(maxUncorrectedRowSum, sum);
      next[i] = sum+escape[i] > 0 ? Math.sqrt(scale[i]/(sum+escape[i])) : 1;
    }
    [scale,next] = [next,scale];
  }
  const sums = new Float64Array(links.length);
  for (let i=0; i<links.length; i++) {
    for (const entry of links[i]) {
      entry[1] *= scale[i]*scale[entry[0]];
      sums[i] += entry[1];
    }
  }
  // Finite samples can give incompatible target row sums, e.g. on two
  // isolated parallel sheets. Enforce passivity on both ends of each link.
  const limit = 1 - (links.length+4)*Number.EPSILON;
  for (let i=0; i<links.length; i++) for (const entry of links[i]) {
    const sum = Math.max(sums[i],sums[entry[0]]);
    if (sum>limit) entry[1] *= limit/sum;
  }
  return maxUncorrectedRowSum;
}

export function buildExchange(rectangles) {
  const { frames, patches } = makePatches(rectangles);
  const { links, escape, exchangeLinks, escapedSamples, raysCast } = sampleExchange(frames, patches);
  const maxUncorrectedRowSum = balanceExchange(links, escape);
  return { patches, links, stats: { patchCount: patches.length, exchangeLinks,
    segmentTests: raysCast, samplesPerSide, escapedSamples, maxUncorrectedRowSum } };
}
