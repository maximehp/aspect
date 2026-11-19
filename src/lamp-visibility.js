// Gnomonic tiles index the directions leaving the point light. Each entry is a
// conservative caster list. The actual light segment still intersects the
// original surface, so the tile size does not set the resolution of a shadow.
const size = 32;
const faceCount = 6;
const pad = 1e-5; // widen angular bounds for the CPU-double / shader-float transition
export const boxFaces = [[0,4,6,2],[1,3,7,5],[0,1,5,4],[2,6,7,3],[0,2,3,1],[4,5,7,6]];

export function boxVertices(box, inset = 0) {
  return Array.from({ length: 8 }, (_, bits) => box.center.map((c,k) =>
    c + box.axes.reduce((sum,axis,i) => sum + ((bits >> i)&1 ? 1 : -1)*(box.halfSize[i]-inset)*axis[k], 0)));
}

export function sheetVertices(sheet) {
  return [[-1,-1],[1,-1],[1,1],[-1,1]].map(([u,v]) =>
    sheet.center.map((x,k) => x + u*sheet.axisU[k] + v*sheet.axisV[k]));
}

// Matching coordinate swizzles are in the shader below.
function project(p, face) {
  const i = face >> 1;
  return [p[(i+1)%3],p[(i+2)%3],(face&1 ? -1 : 1)*p[i]];
}

function clip(poly, axis, sign) {
  const output = [];
  for (let i=0; i<poly.length; i++) {
    const p=poly[i], q=poly[(i+1)%poly.length];
    const a=p[2]+sign*p[axis], b=q[2]+sign*q[axis];
    if (a>=0) output.push(p);
    if ((a>0&&b<0)||(a<0&&b>0)) {
      const t=a/(a-b);
      output.push(p.map((x,k)=>x+t*(q[k]-x)));
    }
  }
  return output;
}

const turn = (a,b,c) => (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
function chain(points) {
  const out=[];
  for (const p of points) {
    while (out.length>=2 && turn(out.at(-2),out.at(-1),p)<=0) out.pop();
    out.push(p);
  }
  out.pop();
  return out;
}

function faceHull(vertices, facets, face) {
  const pts=vertices.map(p=>project(p,face));
  if (pts.every(p=>p[2]<=0)) return null;
  let projected=[];
  for (const facet of facets) {
    let poly=facet.map(i=>pts[i]);
    for (let axis=0; axis<2; axis++) for (const sign of [-1,1]) poly=clip(poly,axis,sign);
    for (const p of poly) if (p[2]>0) projected.push([p[0]/p[2],p[1]/p[2]]);
  }
  if (projected.length<3) return null;
  projected.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  const hull=chain(projected).concat(chain(projected.reverse()));
  if (hull.length<3) return null;
  const sides=hull.map((p,i)=>{
    const q=hull[(i+1)%hull.length], a=p[1]-q[1], b=q[0]-p[0];
    return [a,b,-a*p[0]-b*p[1], (Math.abs(a)+Math.abs(b))*(1.0625/size+pad)];
  });
  const cell=v=>Math.max(0, Math.min(size-1,Math.floor((v+1)*.5*size)));
  return {sides, minX:cell(Math.min(...hull.map(p=>p[0]))-pad-.0625/size), maxX:cell(Math.max(...hull.map(p=>p[0]))+pad+.0625/size),
    minY:cell(Math.min(...hull.map(p=>p[1]))-pad-.0625/size), maxY:cell(Math.max(...hull.map(p=>p[1]))+pad+.0625/size),
    near:Math.max(0,Math.min(...pts.map(p=>p[2]))), far:Math.max(...pts.map(p=>p[2]))};
}

function covers(sides,x,y) {
  return sides.every(([a,b,c,r])=>a*x+b*y+c>=r);
}

function casterTiles(shape, lamp) {
  const solid=Boolean(shape.axes);
  const facets=solid?boxFaces:[[0,1,2,3]];
  const relative=points=>points.map(p=>p.map((v,k)=>v-lamp[k]));
  const vertices=relative(solid?boxVertices(shape):sheetVertices(shape));
  const inset=solid?relative(boxVertices(shape,shape.radius)):vertices;
  const out=[];
  for (let face=0; face<faceCount; face++) {
    const hull=faceHull(vertices,facets,face);
    if (!hull) continue;
    const core=solid?faceHull(inset,facets,face):hull;
    for (let y=hull.minY;y<=hull.maxY;y++) for (let x=hull.minX;x<=hull.maxX;x++) {
      const u=(2*x+1)/size-1, v=(2*y+1)/size-1;
      if (hull.sides.some(([a,b,c,r])=>a*u+b*v+c < -r)) continue;
      out.push({tile:(face*size+y)*size+x, full:core && covers(core.sides,u,v), near:hull.near, far:hull.far});
    }
  }
  return out;
}

export class LampVisibility {
  constructor() {
    this.lamp=[];
    this.cache=new WeakMap();
  }

  build(shapes, lamp, maxTextureSize) {
    if (lamp.some((x,i)=>x!==this.lamp[i])) {
      this.cache=new WeakMap(); this.lamp=[...lamp];
    }
    const tileCount=size*size*faceCount, lists=Array.from({length:tileCount},()=>[]);
    const cutoff=new Float64Array(tileCount).fill(Infinity);
    shapes.forEach((shape,id)=>{
      let coverage=this.cache.get(shape);
      if (!coverage) { coverage=casterTiles(shape,lamp); this.cache.set(shape,coverage); }
      for (const tile of coverage) {
        if (tile.full) cutoff[tile.tile]=Math.min(cutoff[tile.tile], tile.far);
        lists[tile.tile].push({id, tile});
      }
    });
    const data=new Uint32Array(2*(tileCount+lists.reduce((n,l)=>n+l.length,0)));
    let ptr=tileCount, maxCandidates=0;
    lists.forEach((list,cell)=>{
      const start=ptr;
      list.sort((a,b)=>Number(Boolean(shapes[b.id].axes))-Number(Boolean(shapes[a.id].axes)) || a.tile.near-b.tile.near);
      let boxes=0;
      for (const {id,tile} of list) {
        if (tile.near>cutoff[cell]+pad*(1+cutoff[cell])) continue;
        if(shapes[id].axes) boxes++;
        data[2*ptr++]=2*id+Number(Boolean(tile.full));
      }
      if (ptr-start>=65536) throw new Error('Too many overlapping lamp blockers in one angular tile.');
      data[2*cell]=start;data[2*cell+1]=65536*boxes+ptr-start;
      maxCandidates=Math.max(maxCandidates,ptr-start);
    });
    const width=Math.min(256,maxTextureSize), height=Math.ceil(ptr/width);
    if (height>maxTextureSize) throw new Error('Lamp visibility lists exceed maximum texture size.');
    const pixels=new Uint32Array(2*width*height);
    pixels.set(data.subarray(0,2*ptr));
    return {width,height,pixels,maxCandidates};
  }
}

export const visibilityGLSL = `
uniform highp usampler2D lampTiles;
uvec2 lampRead(int a) {
  int w=textureSize(lampTiles,0).x;
  return texelFetch(lampTiles,ivec2(a%w,a/w),0).xy;
}
uvec2 lampRange(vec3 d) {
  vec3 magnitude=abs(d);
  int axis=magnitude.x>=magnitude.y && magnitude.x>=magnitude.z?0:(magnitude.y>=magnitude.z?1:2);
  vec3 p=axis==0?d.yzx:(axis==1?d.zxy:d);
  int face=2*axis+int(p.z<0.);
  ivec2 tile=clamp(ivec2((p.xy/abs(p.z)+1.)*float(${size})*.5), ivec2(0), ivec2(${size-1}));
  return lampRead((face*${size}+tile.y)*${size}+tile.x);
}
bool seesLamp(vec3 world, int self, bool testSheets) {
  vec3 d=world-lampPosition;uvec2 range=lampRange(d);
  int end=int(range.x + (testSheets ? (range.y&65535u) : range.y>>16u));
  for(int i=int(range.x);i<end;++i) {
    int code=int(lampRead(i).x), id=code/2;
    if (id==self) continue;
    vec4 center=shapeRow(id,0);
    vec3 diff=lampPosition-center.xyz;
    vec4 z=shapeRow(id,3);
    if (center.w==0.) {
      vec4 plane=shapeRow(id,1); float denom=dot(plane.xyz,d);
      if(denom==0.)continue;
      float t=-dot(plane.xyz,diff)/denom;
      if(t<=0.||t>=.999999)continue;
      if ((code & 1)!=0) return false;
      vec3 hit=diff+t*d;
      if(abs(dot(shapeRow(id,2).xyz,hit))<=1. && abs(dot(z.xyz,hit))<=1.) return false;
    } else {
      vec4 x=shapeRow(id,1),y=shapeRow(id,2);
      mat3 axes=mat3(x.xyz,y.xyz,z.xyz);
      float hit;
      if(intersectRoundedBox(diff*axes,d*axes,vec3(x.w,y.w,z.w),center.w,0.,.999999,hit))return false;
    }
  }
  return true;
}

// For a single silhouette the projection of a square pixel onto its normal
// is a trapezoidal distribution. Its integral is the covered area.
float halfplaneArea(vec3 line){
  vec2 g=abs(line.xy);
  float a=max(g.x,g.y),b=min(g.x,g.y),tail=.5*(a+b)-abs(line.z);
  float minority=tail<=b?tail*tail/(2.*max(a*b,1.e-25)):(tail-.5*b)/a;
  return line.z>=0.?1.-minority:minority;
}

// Integrate the oriented edges of the intersection (Green's theorem).
// Along each edge only the segment inside both halfplanes contributes.
void clipInterval(float origin,float delta,inout vec2 interval){
  if(delta>0.)interval.x=max(interval.x,-origin/delta);
  else if(delta<0.)interval.y=min(interval.y,-origin/delta);
  else if(origin<0.)interval=vec2(1.,0.);
}
float squareEdge(vec2 origin,vec2 delta,vec3 a,vec3 b){
  vec2 interval=vec2(0.,1.);
  clipInterval(dot(a.xy,origin)+a.z,dot(a.xy,delta),interval);
  clipInterval(dot(b.xy,origin)+b.z,dot(b.xy,delta),interval);
  return .5*max(0.,interval.y-interval.x);
}
float silhouetteEdge(vec3 line,vec3 other){
  vec2 origin=-line.z*line.xy;
  vec2 delta=vec2(line.y,-line.x);
  vec2 span=vec2(-2.,2.);
  clipInterval(.5+origin.x,delta.x,span);clipInterval(.5-origin.x,-delta.x,span);
  clipInterval(.5+origin.y,delta.y,span);clipInterval(.5-origin.y,-delta.y,span);
  clipInterval(dot(other.xy,origin)+other.z,dot(other.xy,delta),span);
  return line.z*max(0.,span.y-span.x);
}
float jointArea(vec3 a,vec3 b){
  a/=length(a.xy);b/=length(b.xy);
  float parallel=a.x*b.y-a.y*b.x;
  if(abs(parallel)<1.e-6){
    if(dot(a.xy,b.xy)>0.)return halfplaneArea(a.z<b.z?a:b);
    return max(0.,halfplaneArea(a)+halfplaneArea(b)-1.);
  }
  float edges=squareEdge(vec2(-.5,-.5),vec2(1.,0.),a,b)
             +squareEdge(vec2(.5,-.5),vec2(0.,1.),a,b)
             +squareEdge(vec2(.5,.5),vec2(-1.,0.),a,b)
             +squareEdge(vec2(-.5,.5),vec2(0.,-1.),a,b);
  return clamp(.5*(edges+silhouetteEdge(a,b)+silhouetteEdge(b,a)),0.,1.);
}
float junctionCoverage(vec3 p,vec3 dx,vec3 dy,int self){
  float visible=0.;
  for(int i=0;i<16;i++){
    vec2 offset=(vec2(i%4,i/4)+.5)*.25-.5;
    visible+=float(seesLamp(p+dx*offset.x+dy*offset.y,self,true));
  }
  return visible/16.;
}

bool segmentBox(vec3 o,vec3 d,vec3 h){
  if(any(lessThan(h,vec3(0))))return false;
  bvec3 parallel=equal(d,vec3(0));
  if(any(bvec3(parallel.x&&abs(o.x)>h.x,parallel.y&&abs(o.y)>h.y,parallel.z&&abs(o.z)>h.z)))return false;
  vec3 inv=1./mix(d,vec3(1.),parallel),a=(-h-o)*inv,b=(h-o)*inv;
  vec3 lo=mix(min(a,b),vec3(-1.e28),parallel),hi=mix(max(a,b),vec3(1.e28),parallel);
  return max(0.,max(lo.x,max(lo.y,lo.z)))<=min(1.,min(hi.x,min(hi.y,hi.z)));
}

float filteredVisibility(vec3 world,vec3 dx,vec3 dy,int self,float exposed,bool nearSheetEdge) {
  // Away from projected cell edges the entire footprint has one known
  // planar visibility. Only the moving solids can change it there.
  if(!nearSheetEdge && exposed==0.)return 0.;
  vec3 d=world-lampPosition;
  uvec2 range=lampRange(d);
  int start=int(range.x), end=int(range.x+(range.y>>16u));
  if(start==end)return exposed;
  vec3 lineA=vec3(0),lineB=vec3(0);int count=0;
  float margin=.5*(length(dx)+length(dy));
  // A subpixel opening can be bounded by a flat sill on one side and a
  // curved slat on the other. Intersect both constraints within the pixel.
  int stop=int(range.x+(range.y&65535u));
  if(nearSheetEdge)for(int i=end;i<stop;i++){
    int code=int(lampRead(i).x),id=code/2;
    if(id==self)continue;
    vec3 n=shapeRow(id,1).xyz;
    float den=dot(n,d);
    if(den==0.)continue;
    vec3 o=lampPosition-shapeRow(id,0).xyz;
    float t=-dot(n,o)/den;
    if(t<=0.||t>=.999999)continue;
    if((code&1)!=0)return 0.;
    vec3 q=o+t*d,u=shapeRow(id,2).xyz,v=shapeRow(id,3).xyz;
    vec2 ab=vec2(dot(u,q),dot(v,q));
    vec3 du=t*(dx-d*(dot(n,dx)/den)),dv=t*(dy-d*(dot(n,dy)/den));
    vec2 gu=sign(ab.x)*vec2(dot(u,du),dot(u,dv)),gv=sign(ab.y)*vec2(dot(v,du),dot(v,dv));
    vec2 reach=.5*vec2(abs(gu.x)+abs(gu.y),abs(gv.x)+abs(gv.y)),gap=abs(ab)-1.;
    if(any(greaterThanEqual(gap,reach)))continue;
    if(all(lessThanEqual(gap,-reach)))return 0.;
    if(all(greaterThan(gap,-reach)) || count==2)return junctionCoverage(world,dx,dy,self);
    bool useU=gap.x>-reach.x;
    vec3 line=useU?vec3(gu,gap.x):vec3(gv,gap.y);
    if(count++==0)lineA=line;else lineB=line;
  }
  for(int i=start;i<end;i++) {
    int id=int(lampRead(i).x)/2;
    vec4 c=shapeRow(id,0),x=shapeRow(id,1),y=shapeRow(id,2),z=shapeRow(id,3);
    mat3 axes=mat3(x.xyz,y.xyz,z.xyz);
    vec3 o=(lampPosition-c.xyz)*axes, localD=d*axes,h=vec3(x.w,y.w,z.w);
    if(!segmentBox(o,localD,h+margin))continue;
    // The inset box lies completely inside the radius-eroded rounded body.
    if(segmentBox(o,localD,h-c.w-margin))return 0.;
    vec3 grad;
    float gap=roundedBoxSegmentDistance(o,localD,h,c.w,grad);
    // If the center segment crosses the inner box, minimum distance is
    // constant there. A wide pixel may still reach the straight silhouette
    // of a long edge. Recover its signed projected gap and first derivative.
    if(margin>c.w && gap<=-c.w && all(equal(grad,vec3(0.)))) {
      vec2 v=vec2(-localD.y,localD.x);
      float speed=length(v);
      bool insideSpan=false;
      if(speed>0.) {
        v/=speed;
        float offset=dot(v,o.xy);
        v*=sign(offset);
        vec3 core=h-c.w;
        float t=dot(sign(v)*core.xy-o.xy,localD.xy)/(speed*speed);
        if(t>=0. && t<=1. && abs(o.z+t*localD.z)+margin<=core.z) {
          gap=abs(offset)-dot(abs(v),core.xy)-c.w;
          grad=vec3(t*v,0.);
          insideSpan=true;
        }
      }
      if(!insideSpan)return junctionCoverage(world,dx,dy,self);
    }
    vec3 worldGrad=axes*grad;
    vec2 gradient=vec2(dot(dx,worldGrad),dot(dy,worldGrad));
    float reach=.5*(abs(gradient.x)+abs(gradient.y));
    if(gap<=-reach)return 0.;
    if(gap>=reach)continue;
    if(count==2)return junctionCoverage(world,dx,dy,self);
    vec3 line=vec3(gradient,gap);
    if(count++==0)lineA=line;else lineB=line;
  }
  if(count==0)return 1.;
  return count==1?halfplaneArea(lineA):jointArea(lineA,lineB);
}`;
