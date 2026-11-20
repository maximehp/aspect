import {exact, scalarValue} from './authored-scalar.js';
// A box with a circular fillet on every edge and a spherical octant at each
// corner. halfSize is measured to its exterior, in an orthonormal local frame.
export function roundedBox(id, center, axes, halfSize, radius, color) {
  const definition={};
  function coordinates(values, key){
    if(values.every(x=>typeof x==='number'))return values;
    definition[key]=values;return values.map(scalarValue);
  }
  center=coordinates(center,'center');halfSize=coordinates(halfSize,'halfSize');
  if(axes.some(a=>a.some(x=>typeof x!=='number'))){definition.axes=axes;axes=axes.map(a=>a.map(scalarValue));}
  if(typeof radius!=='number'){definition.radius=radius;radius=scalarValue(radius);}
  if (!(radius > 0 && radius <= Math.min(...halfSize))) {
    throw new RangeError('Fillet radius must fit within every box half-extent.');
  }
  const extent = roundedBoxExtent({ axes, halfSize, radius });
  return { id, center, axes, halfSize, radius, color, ...(Object.keys(definition).length?{definition}:{}),
    bounds: { min: center.map((x,k) => x-extent[k]), max: center.map((x,k) => x+extent[k]) } };
}

export function roundedBoxExtent({ axes, halfSize, radius }) {
  return [0, 1, 2].map(k =>
    radius + axes.reduce((sum, axis, i) => sum + Math.abs(axis[k]) * (halfSize[i] - radius), 0));
}

// A plain scene edit can copy a shape and replace one coordinate. Reuse its
// authored rational only while it still describes the current numeric pose.
export function exactBoxGeometry(shape) {
  function coordinate(current, authored) {
    if (Array.isArray(current) || ArrayBuffer.isView(current))
      return Array.from(current, (value, i) => coordinate(value, authored?.[i]));
    return exact(authored !== undefined && scalarValue(authored) === current ? authored : current);
  }
  const saved = shape.definition;
  return {
    center: coordinate(shape.center, saved?.center),
    axes: coordinate(shape.axes, saved?.axes),
    halfSize: coordinate(shape.halfSize, saved?.halfSize),
    radius: coordinate(shape.radius, saved?.radius),
  };
}

const dot = (a,b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];

// Squared distance to the inner box is quadratic between consecutive core
// planes. Solve each of those intervals analytically, in order along the ray.
// Returns the first boundary crossing in (tMin,tMax], including exit rays.
function localDistance(o,d,h,r,tMin,tMax) {
  let entry = -Infinity, exit = Infinity;
  for (let i = 0; i < 3; i++) {
    if (d[i] === 0) {
      if (Math.abs(o[i]) > h[i]) return Infinity;
    } else {
      const a = (-h[i]-o[i])/d[i], b = (h[i]-o[i])/d[i];
      entry = Math.max(entry,Math.min(a,b));
      exit = Math.min(exit,Math.max(a,b));
    }
  }
  let from = Math.max(tMin,entry);
  const end = Math.min(tMax,exit);
  if (from > end || exit <= tMin || !Number.isFinite(exit)) return Infinity;
  const core = h.map(x => x-r);
  const onFace = t => {
    let outside = 0;
    for (let i = 0; i < 3; i++) {
      if (Math.abs(o[i]+d[i]*t) > core[i]) outside++;
    }
    return outside <= 1;
  };
  if (entry > tMin) {
    const offset = o.map((v,i) => Math.max(Math.abs(v+entry*d[i])-core[i],0));
    if (onFace(entry) || dot(offset,offset) <= r*r) return entry;
  }
  const events = [];
  for (let i = 0; i < 3; i++) {
    if (d[i] === 0) continue;
    for (const s of [-1,1]) {
      const t = (s*core[i]-o[i])/d[i];
      if (t > from && t < end) events.push(t);
    }
  }
  events.sort((a,b) => a-b);
  events.push(end);
  for (const until of events) {
    if (until <= from) continue;
    const mid = from + (until-from)*.5;
    let a = 0, b = 0, c = -r*r;
    for (let i = 0; i < 3; i++) {
      const p = o[i]+d[i]*mid;
      if (Math.abs(p) <= core[i]) continue;
      const q = (o[i]-Math.sign(p)*core[i])+d[i]*from;
      a += d[i]*d[i];
      b += q*d[i];
      c += q*q;
    }
    const disc = b*b-a*c;
    if (a > 0 && disc >= 0) {
      const s = Math.sqrt(disc);
      // Compute the root farther from zero first to avoid cancellation.
      const q = -b-(b < 0 ? -s : s);
      const x = q/a, y = q === 0 ? x : c/q;
      for (const root of [Math.min(x,y),Math.max(x,y)]) {
        const t = from+root;
        if (root >= 0 && t <= until && t > tMin) return t;
      }
    }
    from = until;
  }
  return exit <= tMax && onFace(exit) ? exit : Infinity;
}

// Rays may be unnormalized, e.g. direction = light - position and tMax = 1.
export function roundedBoxDistance(box,origin,direction,tMin=0,tMax=Infinity) {
  const relative = origin.map((v,k) => v-box.center[k]);
  const o = box.axes.map(axis => dot(relative,axis));
  const d = box.axes.map(axis => dot(direction,axis));
  return localDistance(o,d,box.halfSize,box.radius,tMin,tMax);
}

export function hitRoundedBox(box,origin,direction,tMin=0,tMax=Infinity) {
  const t = roundedBoxDistance(box,origin,direction,tMin,tMax);
  if (!Number.isFinite(t)) return null;
  const point = origin.map((v,k) => v-box.center[k]+t*direction[k]);
  const n = box.axes.map((axis,k) => {
    const v = dot(point,axis);
    return Math.sign(v)*Math.max(0,Math.abs(v)-(box.halfSize[k]-box.radius));
  });
  const length = Math.hypot(...n);
  const normal = point.map((_,k) => (n[0]*box.axes[0][k] + n[1]*box.axes[1][k] + n[2]*box.axes[2][k])/length);
  return { t, normal };
}

// Same piecewise-quadratic intersection in the local frame, shared by direct
// shadows and primary visibility. The exterior-box slab test and face test
// handle broad misses and the planar part of each slat before curved work.
export const roundedBoxGLSL = `
vec3 roundedBoxNormal(vec3 p, vec3 h, float r) {
  return normalize(sign(p)*max(abs(p)-(h-vec3(r)),vec3(0.)));
}

// Distance from a closed segment to the inner box, minus the fillet radius.
// The segment origin is fixed. At the minimizing parameter t the derivative
// with respect to the segment's endpoint is t times the separation normal.
float roundedBoxSegmentDistance(vec3 o, vec3 d, vec3 h, float r, out vec3 gradEndpoint) {
  vec3 core=h-vec3(r);
  gradEndpoint=vec3(0.);
  vec3 q0=o-clamp(o,-core,core);
  float slope0=dot(q0,d);
  if(slope0>=0.) return length(q0)-r;
  vec3 endPoint=o+d, q1=endPoint-clamp(endPoint,-core,core);
  float slope1=dot(q1,d);
  if(slope1<=0.) {
    float separation=length(q1);
    if(separation>0.) gradEndpoint=q1/separation;
    return separation-r;
  }

  // A line missing the XY core rectangle has a closest supporting corner.
  // When that point lies on the segment and within the extrusion, its 2D
  // distance is the exact 3D minimum. This covers long-edge silhouettes.
  float speed2=dot(d.xy,d.xy);
  if(speed2>0.) {
    vec2 perp=vec2(-d.y,d.x);
    float signedHeight=dot(perp,o.xy);
    float gap=abs(signedHeight)-dot(abs(perp),core.xy);
    if(gap>0.) {
      perp*=sign(signedHeight);
      vec2 corner=sign(perp)*core.xy;
      float t=dot(corner-o.xy,d.xy)/speed2;
      if(t>=0. && t<=1. && abs(o.z+t*d.z)<=core.z) {
        float invSpeed=inversesqrt(speed2);
        gradEndpoint=vec3(t*invSpeed*perp,0.);
        return gap*invSpeed-r;
      }
    }
  }

  // Squared distance has a continuous piecewise-linear derivative. On each
  // interval use only its active coordinates. Evaluating the slope on a
  // core-plane event itself can admit a spurious axis after FP32 rounding.
  bvec3 parallel=equal(d,vec3(0.));
  vec3 inv=1./mix(d,vec3(1.),parallel);
  vec3 a=mix((-core-o)*inv,vec3(2.),parallel);
  vec3 b=mix((core-o)*inv,vec3(2.),parallel);
  vec3 starts=mix(min(a,b),vec3(0.),parallel);
  vec3 stops=mix(max(a,b),vec3(1.),parallel);
  float enter=max(0.,max(starts.x,max(starts.y,starts.z)));
  float leave=min(1.,min(stops.x,min(stops.y,stops.z)));
  bool outsideParallel=(parallel.x && abs(o.x)>core.x) ||
                       (parallel.y && abs(o.y)>core.y) ||
                       (parallel.z && abs(o.z)>core.z);
  if(!outsideParallel && enter<=leave) return -r;

  float from=0., best=dot(q0,q0), bestT=0.;
  for(int i=0;i<7;++i) {
    a=mix(a,vec3(2.),lessThanEqual(a,vec3(from)));
    b=mix(b,vec3(2.),lessThanEqual(b,vec3(from)));
    vec3 next=min(a,b);
    float until=min(1.,min(next.x,min(next.y,next.z)));
    vec3 mid=o+(from+.5*(until-from))*d;
    bvec3 activeAxes=greaterThan(abs(mid),core);
    vec3 v=mix(vec3(0.),d,activeAxes);
    vec3 offset=mix(vec3(0.),(o-sign(mid)*core)+from*d,activeAxes);
    float curvature=dot(v,v);
    float localMin=curvature>0. ? -dot(offset,v)/curvature : 0.;
    float t=from+clamp(localMin,0.,until-from);
    vec3 p=o+t*d;
    vec3 q=p-clamp(p,-core,core);
    float distance2=dot(q,q);
    if(distance2<best) { best=distance2; bestT=t; }
    if(until>=1.) break;
    from=until;
  }
  vec3 p=o+bestT*d;
  float separation=sqrt(best);
  if(separation>0.) gradEndpoint=bestT*(p-clamp(p,-core,core))/separation;
  return separation-r;
}

bool roundedFlat(vec3 p, vec3 core) {
  bvec3 outside = greaterThan(abs(p), core);
  return !(outside.x && outside.y || outside.x && outside.z || outside.y && outside.z);
}

bool intersectRoundedBox(vec3 o, vec3 d, vec3 h, float r,
                         float tMin, float tMax, out float t) {
  const float farValue = 1.e30;
  bvec3 parallel = equal(d,vec3(0.));
  if ((parallel.x && abs(o.x)>h.x) || (parallel.y && abs(o.y)>h.y) ||
      (parallel.z && abs(o.z)>h.z) || all(parallel)) return false;
  vec3 inv = 1./mix(d,vec3(1.),parallel);
  vec3 a = (-h-o)*inv, b = (h-o)*inv;
  vec3 low = mix(min(a,b),vec3(-farValue),parallel);
  vec3 high = mix(max(a,b),vec3(farValue),parallel);
  float enter = max(max(low.x,low.y),low.z), leave = min(min(high.x,high.y),high.z);
  float from = max(enter,tMin), end = min(leave,tMax);
  if (from>end || leave<=tMin) return false;
  vec3 core = h-vec3(r);
  if (enter>tMin) {
    vec3 p=o+enter*d, q=max(abs(p)-core,vec3(0.));
    // A nearly flat curved contact can round to just inside the shell at
    // slab entry. Return its front surface rather than crossing to the back.
    if (roundedFlat(p,core) || dot(q,q)<=r*r) { t=enter; return true; }
  }

  a = mix((-core-o)*inv,vec3(farValue),parallel);
  b = mix((core-o)*inv,vec3(farValue),parallel);
  for (int i=0; i<7; ++i) {
    a = mix(a,vec3(farValue),lessThanEqual(a,vec3(from)));
    b = mix(b,vec3(farValue),lessThanEqual(b,vec3(from)));
    vec3 events = min(a,b);
    float until = min(end,min(events.x,min(events.y,events.z)));
    vec3 mid = o+(.5*(until-from)+from)*d;
    bvec3 activeAxes = greaterThan(abs(mid),core);
    vec3 q = mix(vec3(0.),(o-sign(mid)*core)+from*d,activeAxes);
    vec3 v = mix(vec3(0.),d,activeAxes);
    float aa = dot(v,v), bb = dot(q,v), cc = dot(q,q)-r*r;
    float disc = bb*bb-aa*cc;
    if (aa>0. && disc>=0.) {
      float sq = sqrt(disc);
      float bigRoot = -bb-(bb<0. ? -sq : sq);
      float x = bigRoot/aa, y = bigRoot==0. ? x : cc/bigRoot;
      float first = min(x,y), second = max(x,y);
      float tolerance=2.4e-7*max(abs(from),abs(until));
      float candidate=from+first;
      t=clamp(candidate,from,until);
      if (candidate>=from-tolerance && candidate<=until+tolerance && t>tMin) return true;
      candidate=from+second;
      t=clamp(candidate,from,until);
      if (candidate>=from-tolerance && candidate<=until+tolerance && t>tMin) return true;
    }
    if (until>=end) break;
    from=until;
  }
  t=leave;
  return leave<=tMax && roundedFlat(o+leave*d,core);
}`;
