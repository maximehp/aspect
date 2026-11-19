// First diffuse reflection from the continuous directly lit parts of the scene.
// Integrate in both emitter area and receiver direction. Their balance weights
// add to one for every visible light path. Small bright sheets get explicit
// source samples; nearby walls that cover much of the hemisphere get angular samples.
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const scale = (v, s) => v.map(x => x*s);
const diff = (a, b) => a.map((x, i) => x-b[i]);
const pointOn = (r, u, v) => r.center.map((x,k) => x + u*r.axisU[k] + v*r.axisV[k]);
const angularSamples = 64;
const samplesPerSteradian = 96;
const receiverSamples = 64;
const interleave = 16;
const roundoff = 64 * Number.EPSILON;

function radicalInverse(n, base) {
  let value = 0, place = 1 / base;
  for (; n; n = Math.floor(n/base), place /= base) value += (n % base)*place;
  return value;
}

function geometryFrame({rect, cells}, lamp) {
  const crossUV = cross(rect.axisU, rect.axisV), jacobian = Math.hypot(...crossUV);
  const normal = scale(crossUV, 1/jacobian);
  const dualU = scale(cross(rect.axisV, normal), 1/jacobian);
  const dualV = scale(cross(normal, rect.axisU), 1/jacobian);
  const height = dot(diff(lamp, rect.center), normal);
  return { rect, cells, normal, dualU, dualV, jacobian, height, regions: [] };
}

// For a point lamp, D(x) dA is proportional to the solid angle subtended
// by dA. Uniform sampling of that solid angle concentrates points where the
// lamp actually sends its power, even on a very large nearby reflector.
function litRegion(frame, uv, lamp) {
  const a = uv[0], triangles = [], cumulative = [];
  const rays = uv.map(([u,v]) => diff(pointOn(frame.rect,u,v),lamp));
  let solidAngle = 0, areaUV = 0;
  let uMin = Infinity, uMax = -Infinity, vMin = Infinity, vMax = -Infinity;
  for (const [u,v] of uv) {
    uMin = Math.min(uMin,u); uMax = Math.max(uMax,u);
    vMin = Math.min(vMin,v); vMax = Math.max(vMax,v);
  }
  for (let k = 1; k+1 < uv.length; k++) {
    const b = uv[k], c = uv[k+1];
    const twiceArea = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
    if (!(twiceArea > 0)) continue;

    const A = rays[0], B = rays[k], C = rays[k+1];
    const la = Math.hypot(...A), lb = Math.hypot(...B), lc = Math.hypot(...C);
    const unitB = scale(B,1/lb);
    const sumAB = unitB.map((v,i) => A[i]/la + v);
    // The triangle solid-angle formula uses |det(A,B,C)|. With coplanar
    // endpoints we can compute it from planar height and UV area directly.
    const determinant = Math.abs(frame.height)*frame.jacobian*twiceArea;
    const denominator = la*lb*dot(sumAB, C.map((v,i) => v + unitB[i]*lc));
    const omega = 2*Math.atan2(determinant, denominator);
    if (!(omega > 0)) continue;

    areaUV += twiceArea/2;
    solidAngle += omega;
    triangles.push({a,b,c,A,edge:diff(C,A),la,lb,unitB,sumAB,determinant});
    cumulative.push(solidAngle);
  }
  if (!(solidAngle > 0)) return null;
  const count = Math.max(4, Math.min(32,Math.ceil(areaUV*frame.jacobian)),
    Math.ceil(solidAngle*samplesPerSteradian));
  return {uv, cumulative, triangles, solidAngle, count, density:count/solidAngle,
    uMin,uMax,vMin,vMax};
}

// Arvo's area-preserving spherical triangle map (SIGGRAPH 1995).
// Invert the subtriangle solid angle along AC, then sample 1-cos(theta)
// uniformly along the arc from B. Reconstruct in planar barycentrics so a
// grazing direction always gives a point on the original polygon.
// https://www.graphics.cornell.edu/pubs/1995/Arv95c.pdf
function solidSample(triangle, targetAngle, fraction, frame) {
  const {A,edge,la,lb,unitB,sumAB,determinant} = triangle;
  const q = [0,0,0];
  let lo = 0, hi = 1;
  for (let iteration = 0; iteration < 40; iteration++) {
    const t = (lo+hi)/2;
    for (let i=0; i<3; i++) q[i] = A[i] + t*edge[i];
    const lq = Math.hypot(...q);
    const denominator = la*lb*(sumAB[0]*(q[0]+unitB[0]*lq) +
      sumAB[1]*(q[1]+unitB[1]*lq) + sumAB[2]*(q[2]+unitB[2]*lq));
    if (2*Math.atan2(t*determinant,denominator) < targetAngle) lo = t;
    else hi = t;
  }
  const t = (lo+hi)/2;
  for (let i=0; i<3; i++) q[i] = A[i] + t*edge[i];
  const lq = Math.hypot(...q), unitQ = scale(q,1/lq);
  const delta = diff(unitQ,unitB);
  const oneMinusCos = dot(delta,delta)/2;
  const cosTheta = 1-fraction*oneMinusCos;
  const sinTheta = Math.sqrt(Math.max(0,fraction*oneMinusCos*(2-fraction*oneMinusCos)));
  const sinArc = Math.hypot(...cross(unitB,unitQ));
  const weightB = Math.max(0,sinArc*cosTheta-dot(unitB,unitQ)*sinTheta)/lb;
  const weightQ = sinTheta/lq;
  const mix = weightB + weightQ > 0 ? weightQ/(weightB+weightQ) : Math.sqrt(fraction);

  const {a,b,c} = triangle;
  const u = b[0] + mix*(a[0] + t*(c[0]-a[0]) - b[0]);
  const v = b[1] + mix*(a[1] + t*(c[1]-a[1]) - b[1]);
  return pointOn(frame.rect,u,v);
}

function regionAt(frame, u, v) {
  for (const r of frame.regions) {
    if (u<r.uMin || u>r.uMax || v<r.vMin || v>r.vMax) continue;
    let inside=true;
    for (let i=0, j=r.uv.length-1; i<r.uv.length; j=i++) {
      const a=r.uv[j], b=r.uv[i];
      if ((b[0]-a[0])*(v-a[1]) < (b[1]-a[1])*(u-a[0])) { inside=false; break; }
    }
    if (inside) return r;
  }
  return null;
}

function frameCoordinates(point, frames, result = new Float64Array(3*frames.length)) {
  for (let i=0; i<frames.length; i++) {
    const f=frames[i], center=f.rect.center, n=f.normal, u=f.dualU, v=f.dualV;
    const x=point[0]-center[0], y=point[1]-center[1], z=point[2]-center[2];
    const h=x*n[0]+y*n[1]+z*n[2];
    result[i*3]=Math.abs(h)<=roundoff*(1+Math.abs(x)+Math.abs(y)+Math.abs(z)) ? 0 : h;
    result[i*3+1]=x*u[0]+y*u[1]+z*u[2];
    result[i*3+2]=x*v[0]+y*v[1]+z*v[2];
  }
  return result;
}

function blocked(a, b, receiver, source) {
  for (let k=0; k<a.length; k+=3) {
    if (k===3*receiver || k===3*source) continue;
    const h0=a[k], h1=b[k];
    if ((h0>0&&h1<0)||(h0<0&&h1>0)) {
      const t=h0/(h0-h1);
      if (Math.abs(a[k+1]+t*(b[k+1]-a[k+1]))<=1 &&
          Math.abs(a[k+2]+t*(b[k+2]-a[k+2]))<=1) return true;
    }
  }
  return false;
}

function makeSamples(frames, lamp) {
  const samples=[];
  for (const [i,f] of frames.entries()) {
    if (f.height===0 || !f.rect.color.some(c=>c>0)) continue;
    for (const cell of f.cells) {
      if (!cell.lit) continue;
      const r=litRegion(f, cell.uv, lamp);
      if (!r) continue;
      const {uv, density, uMin, uMax, vMin, vMax} = r;
      f.regions.push({uv, density, uMin, uMax, vMin, vMax});
      for (let s=0; s<r.count; s++) {
        const target=(s+.5)*r.solidAngle/r.count;
        const t=r.cumulative.findIndex(end=>end>=target);
        const start=t ? r.cumulative[t-1] : 0;
        const fraction=(radicalInverse(s,2)+.5/r.count)%1;
        const x=solidSample(r.triangles[t],target-start,fraction,f);
        const d=diff(x,lamp), d2=dot(d,d);
        const direct=Math.abs(f.height)/(d2*Math.sqrt(d2));
        samples.push({ i, x, sign:Math.sign(f.height),
          coordinates:frameCoordinates(x,frames), density:r.density*direct,
          color:f.rect.color, illumination:direct });
      }
    }
  }
  return samples;
}

function makeHemisphere(normal, frames) {
  // Anchor quadrature to the world normal so coplanar sheets share directions,
  // even when their local parameter axes or normal conventions differ.
  const helper=Math.abs(normal[0])<.75 ? [1,0,0] : [0,1,0];
  const a=cross(normal,helper), tangent=scale(a,1/Math.hypot(...a));
  const bitangent=cross(normal,tangent), samples=[];
  for (let s=0; s<angularSamples; s++) {
    const z=Math.sqrt((s+.5)/angularSamples), r=Math.sqrt(1-z*z);
    const phi=2*Math.PI*(radicalInverse(s,2)+.5/angularSamples);
    const dx=r*Math.cos(phi), dy=r*Math.sin(phi);
    const d=normal.map((v,k)=>v*z+tangent[k]*dx+bitangent[k]*dy);
    const inFrames=new Float64Array(3*frames.length);
    for (let i=0;i<frames.length;i++) {
      const f=frames[i];
      inFrames[3*i]=dot(d,f.normal);
      inFrames[3*i+1]=dot(d,f.dualU);
      inFrames[3*i+2]=dot(d,f.dualV);
    }
    samples.push({d,z,inFrames});
  }
  return samples;
}

export function createGather(surfaces, lamp, lampRGB) {
  const frames=surfaces.map(s=>geometryFrame(s,lamp));
  const sources=lampRGB.some(value=>value>0) ? makeSamples(frames,lamp) : [];
  const hemispheres=new Map();
  function estimate(index, point, normal, origin, start, stride) {
    const result=[0,0,0];
    // In area measure, K = cos(receiver) cos(source) / (π r²). The
    // angular rule has sample density angularSamples*K at this source.
    // The emitter samples have density (count / solid angle) * D(x),
    // where D(x) = cos(theta_lamp)/r_lamp². Both
    // quadratures divide by the sum to avoid counting a path twice.
    for (let source=start;source<sources.length;source+=stride) {
      const s=sources[source];
      if (s.i===index || origin[3*s.i]*s.sign<=0) continue;
      const x=s.x[0]-point[0], y=s.x[1]-point[1], z=s.x[2]-point[2];
      const height=x*normal[0]+y*normal[1]+z*normal[2];
      if (height<=0) continue;
      const areaHeight=Math.abs(origin[3*s.i]);
      const d2=x*x+y*y+z*z;
      const kernel=height*areaHeight/(Math.PI*d2*d2);
      if (blocked(origin,s.coordinates,index,s.i)) continue;
      const weight=s.illumination*kernel/(s.density+angularSamples*kernel);
      for (let c=0;c<3;c++) result[c]+=weight*s.color[c];
    }

    const key=normal.join(',');
    if (!hemispheres.has(key)) hemispheres.set(key,makeHemisphere(normal,frames));
    const directions=hemispheres.get(key);
    for (let sample=start;sample<directions.length;sample+=stride) {
      const {d,z,inFrames}=directions[sample];
      let distance=Infinity, nearest=-1, hitU=0, hitV=0;
      for (let k=0;k<origin.length;k+=3) {
        const denominator=inFrames[k];
        if (k===index*3 || !denominator) continue;
        const t=-origin[k]/denominator;
        if (!(t>0 && t<distance)) continue;
        const u=origin[k+1]+t*inFrames[k+1], v=origin[k+2]+t*inFrames[k+2];
        if (Math.abs(u)>1 || Math.abs(v)>1) continue;
        distance=t; nearest=k/3; hitU=u; hitV=v;
      }
      if (nearest<0) continue;
      const f=frames[nearest], incidence=-inFrames[nearest*3];
      if (incidence*f.height<=0) continue;
      const region=regionAt(f,hitU,hitV);
      if (!region) continue;
      const x=point[0]+distance*d[0]-lamp[0], y=point[1]+distance*d[1]-lamp[1],
        dz=point[2]+distance*d[2]-lamp[2], l2=x*x+y*y+dz*dz;
      const direct=Math.abs(f.height)/(l2*Math.sqrt(l2));
      const kernel=z*Math.abs(incidence)/(Math.PI*distance*distance);
      const weight=direct*kernel/(angularSamples*kernel+region.density*direct);
      for (let c=0;c<3;c++) result[c]+=weight*f.rect.color[c];
    }
    const rho=frames[index].rect.color;
    for (let c=0;c<3;c++) result[c]*=rho[c]*lampRGB[c]*stride;
    return result;
  }

  function at(index, point, normal) {
    if (!sources.length) return [0,0,0];
    return estimate(index,point,normal,frameCoordinates(point,frames),0,1);
  }

  function patchMeans(patches) {
    const field=new Float64Array(6*patches.length);
    if (!sources.length) return field;
    const o=new Float64Array(3*frames.length);
    for (let i=0; i<patches.length; i++) {
      const p=patches[i], f=frames[p.rectIndex], n=p.normal, back=scale(n,-1);
      // Integrate over position as well as direction. Interleave direction
      // and emitter samples across the patch so a few fixed receiver points
      // cannot entirely decide how much a narrow bright stripe contributes.
      for (let j=0;j<receiverSamples;j++) {
        const du=2*(j+.5)/receiverSamples-1, dv=2*((radicalInverse(j,2)+.5/receiverSamples)%1)-1;
        const x=p.center.map((v,k)=>v+du*f.rect.axisU[k]/p.nu+dv*f.rect.axisV[k]/p.nv);
        frameCoordinates(x,frames,o);
        const start=(Math.floor(j/interleave)*5+j)%interleave;
        const a=estimate(p.rectIndex,x,n,o,start,interleave);
        const b=estimate(p.rectIndex,x,back,o,start,interleave);
        for (let c=0;c<3;c++) {
          field[i*6+c]+=a[c]/receiverSamples;
          field[i*6+3+c]+=b[c]/receiverSamples;
        }
      }
    }
    return field;
  }
  return { at, patchMeans };
}
