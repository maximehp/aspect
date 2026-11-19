import { buildExchange } from './exchange.js';
import { createGather } from './gather.js';
import { DEFAULT_LAMP_RGB } from './lamp.js';

// The numerical certificate below applies to this finite, stored transport
// operator and its patch-average source vector. It does not certify geometric
// quadrature, visibility sampling, or interpolation of patches on the GPU.
const eps = Number.EPSILON;
const tiny = Number.MIN_VALUE;
const nextBuffer = new ArrayBuffer(8);
const nextFloat = new Float64Array(nextBuffer);
const nextBits = new BigUint64Array(nextBuffer);
function nextUp(value) {
  if (value === Infinity) return value;
  if (value === 0) return tiny;
  nextFloat[0] = value;
  nextBits[0] += value > 0 ? 1n : -1n;
  return nextFloat[0];
}
function nextDown(value) { return -nextUp(-value); }
function gamma(operations) {
  const e = operations * eps;
  if (e >= 0.25) throw new Error('Transport row is too large to certify.');
  return nextUp(e / (1 - e));
}
const sameArray = (a, b) => a && a.length === b.length && a.every((x, i) => x === b[i]);

function solveTransport(firstBounce, patches, links, surfaces) {
  const sides = patches.length*2, rho = new Float64Array(sides*3);
  let firstBounceMax = 0, firstBounceLitSides = 0, q = 0, linkCount = 0, maxRowLength=0;
  const operationErrors = new Float64Array(sides);
  for (let i=0;i<sides;i++) {
    const color = surfaces[patches[i>>1].rectIndex].rect.color;
    rho.set(color,i*3);
    const row = links[i];
    if (!row) throw new Error('Missing diffuse exchange row.');
    let rowSum = 0;
    for (const [j,w] of row) {
      if (!Number.isInteger(j) || j<0 || j>=sides || !Number.isFinite(w) || w<0) {
        throw new Error('Invalid diffuse exchange coefficient.');
      }
      rowSum += w;
    }
    linkCount += row.length;
    maxRowLength=Math.max(maxRowLength,row.length);
    // Bound sum of stored, rounded coefficients fl(reflectance * weight).
    const sumUpper = nextUp((rowSum + row.length*tiny) / (1-gamma(row.length+1)));
    const maxRho = Math.max(...color);
    const rowUpper = maxRho === 0 ? 0 : nextUp(nextUp(sumUpper*maxRho) * (1+eps) + row.length*tiny);
    q = Math.max(q,rowUpper);
    operationErrors[i] = gamma(2*row.length+2);
    const peak = Math.max(firstBounce[i*3],firstBounce[i*3+1],firstBounce[i*3+2]);
    if (peak>0) firstBounceLitSides++;
    firstBounceMax = Math.max(firstBounceMax,peak);
  }
  if (!Number.isFinite(firstBounceMax)) throw new Error('Nonfinite first-bounce radiance.');
  const commonStats = { patchCount: patches.length, sideCount: sides, linkCount,
    contractionBound: q, firstBounceMax, firstBounceLitSides };
  // Zero illumination has the exact zero least-energy solution, even when a
  // perfectly reflecting closed component makes the fixed point non-unique.
  if (firstBounceMax === 0) return { indirect: new Float64Array(firstBounce.length),
    stats: { ...commonStats, iterations:0, residualBound:0, discreteErrorBound:0, indirectSolveBound:0, tolerance:0 } };
  if (!(q<1)) throw new Error('Cannot certify this diffuse solve: the transport contraction bound is not below one.');

  function apply(x, output) {
    let maxRounding = 0;
    for (let i=0;i<sides;i++) {
      const base=i*3, row=links[i];
      let r=0,g=0,b=0;
      const rr=rho[base], rg=rho[base+1], rb=rho[base+2];
      for (const [j,w] of row) {
        const j3=j*3;
        r += (rr*w)*x[j3]; g += (rg*w)*x[j3+1]; b += (rb*w)*x[j3+2];
      }
      r+=firstBounce[base]; g+=firstBounce[base+1]; b+=firstBounce[base+2];
      output[base]=r; output[base+1]=g; output[base+2]=b;
      const peak=Math.max(r,g,b), ge=operationErrors[i];
      // Nonnegative sums permit a relative enclosure without cancellation;
      // the tiny term also encloses gradual-underflow errors.
      const rounding=(ge*peak + (2*row.length+2)*tiny)/(1-ge);
      maxRounding=Math.max(maxRounding,rounding);
    }
    return nextUp(nextUp(maxRounding));
  }

  // Solve all bounce orders to a residual tolerance. This cap detects failure;
  // it never turns an unconverged finite-bounce image into an accepted result.
  const denominator = nextDown(1-q);
  const tolerance = Math.max(firstBounceMax*1e-6,16*(2*maxRowLength+2)*tiny/denominator);
  let current=firstBounce.slice(), next=new Float64Array(firstBounce.length);
  let iterations=0, residualBound=Infinity, discreteErrorBound=Infinity;
  for (;iterations<10000;iterations++) {
    const rounding=apply(current,next);
    let residual=0;
    for (let k=0;k<current.length;k++) residual=Math.max(residual,Math.abs(next[k]-current[k]));
    residualBound=nextUp(nextUp(residual)+rounding);
    discreteErrorBound=nextUp(residualBound/denominator);
    if (!Number.isFinite(discreteErrorBound)) throw new Error('Diffuse transport overflowed.');
    if (discreteErrorBound<=tolerance) break;
    [current,next]=[next,current];
  }
  if (iterations===10000) throw new Error('Diffuse transport failed to meet its certified residual tolerance.');
  return { indirect: current, stats: { ...commonStats, iterations: iterations + 1,
    residualBound, discreteErrorBound, indirectSolveBound: discreteErrorBound, tolerance } };
}

export class DiffuseTransport {
  constructor() {
    this.geometryKey=null;
    this.exchange=null;
    this.field=null;
    this.previous=null;
    this.bounces=null;
    this.stats=null;
  }

  update(surfaces, lightPosition, lampRGB=DEFAULT_LAMP_RGB) {
    const started=performance.now();
    if (lightPosition.length!==3 || !lightPosition.every(Number.isFinite) ||
        lampRGB.length!==3 || !lampRGB.every(x=>Number.isFinite(x)&&x>=0)) {
      throw new Error('Invalid diffuse lamp parameters.');
    }
    for (const {rect} of surfaces) if (rect.color.length!==3 ||
      !rect.color.every(x=>Number.isFinite(x)&&x>=0&&x<=1)) {
      throw new Error('Diffuse reflectance must be between zero and one: '+rect.id);
    }
    const geometryKey=JSON.stringify(surfaces.map(({rect:r})=>[r.id,r.center,r.axisU,r.axisV]));
    const exchangeRebuilt=geometryKey!==this.geometryKey;
    const previous=this.previous;
    const unchanged=!exchangeRebuilt && previous && sameArray(previous.light,lightPosition) &&
      sameArray(previous.lampRGB,lampRGB) && surfaces.every((s,i)=>
        s.cells===previous.cells[i] && sameArray(s.rect.color,previous.colors[i]));
    if (unchanged) return { field: this.field, surfaces:surfaces.map((s,i)=>({...s,bounce:this.bounces[i]})),
      stats:{...this.stats,exchangeRebuilt:false,solved:false,updateMs:performance.now()-started} };
    const exchange = exchangeRebuilt ? buildExchange(surfaces.map(s => s.rect)) : this.exchange;
    const {patches,links}=exchange;
    const field = createGather(surfaces, lightPosition, lampRGB);
    const firstBounce = field.patchMeans(patches);
    const {indirect, stats} = solveTransport(firstBounce, patches, links, surfaces);
    const bounces=surfaces.map(()=>null);
    let storageRoundingBound=0, maxIndirect=0;
    for (let pi=0;pi<patches.length;pi++) {
      const {rectIndex,iu,iv,nu,nv}=patches[pi];
      if (!bounces[rectIndex]) bounces[rectIndex]={nu,nv,
        front:new Float32Array(nu*nv*3),back:new Float32Array(nu*nv*3),
        firstFront:new Float32Array(nu*nv*3),firstBack:new Float32Array(nu*nv*3)};
      for (let side=0;side<2;side++) {
        const destination=side===0?bounces[rectIndex].front:bounces[rectIndex].back;
        const first=side===0?bounces[rectIndex].firstFront:bounces[rectIndex].firstBack;
        for (let c=0;c<3;c++) {
          const value=indirect[(pi*2+side)*3+c], index=(iv*nu+iu)*3+c;
          destination[index]=value;
          first[index]=firstBounce[(pi*2+side)*3+c];
          if (!Number.isFinite(destination[index])) throw new Error('Diffuse field exceeds GPU storage range.');
          storageRoundingBound=Math.max(storageRoundingBound,Math.abs(destination[index]-value));
          maxIndirect=Math.max(maxIndirect,value);
        }
      }
    }
    this.field=field;
    this.exchange=exchange;
    this.geometryKey=geometryKey;
    this.bounces=bounces;
    storageRoundingBound=storageRoundingBound===0?0:nextUp(storageRoundingBound);
    const indirectErrorBound=stats.indirectSolveBound+storageRoundingBound;
    this.stats={...this.exchange.stats,...stats,exchangeRebuilt,solved:true,
      storageRoundingBound,
      indirectErrorBound:indirectErrorBound===0?0:nextUp(indirectErrorBound),
      maxIndirect,updateMs:performance.now()-started};
    this.previous={light:[...lightPosition],lampRGB:[...lampRGB],
      cells:surfaces.map(s=>s.cells),colors:surfaces.map(s=>[...s.rect.color])};
    return {field, surfaces:surfaces.map((s,i)=>({...s,bounce:bounces[i]})),stats:this.stats};
  }
}
