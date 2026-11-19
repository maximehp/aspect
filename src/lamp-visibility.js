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
    return [a,b,-a*p[0]-b*p[1], (Math.abs(a)+Math.abs(b))*(1/size+pad)];
  });
  const cell=v=>Math.max(0, Math.min(size-1,Math.floor((v+1)*.5*size)));
  return {sides, minX:cell(Math.min(...hull.map(p=>p[0]))-pad), maxX:cell(Math.max(...hull.map(p=>p[0]))+pad),
    minY:cell(Math.min(...hull.map(p=>p[1]))-pad), maxY:cell(Math.max(...hull.map(p=>p[1]))+pad),
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
      list.sort((a,b)=>a.tile.near-b.tile.near);
      for (const {id,tile} of list) {
        if (tile.near>cutoff[cell]+pad*(1+cutoff[cell])) break;
        data[2*ptr++]=2*id+Number(Boolean(tile.full));
      }
      data[2*cell]=start;data[2*cell+1]=ptr;
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
bool seesLamp(vec3 world, int self) {
  vec3 d=world-lampPosition, magnitude=abs(d);
  int axis=magnitude.x>=magnitude.y && magnitude.x>=magnitude.z?0:(magnitude.y>=magnitude.z?1:2);
  vec3 p=axis==0?d.yzx:(axis==1?d.zxy:d);
  int face=2*axis+int(p.z<0.);
  ivec2 tile=clamp(ivec2((p.xy/abs(p.z)+1.)*float(${size})*.5), ivec2(0), ivec2(${size-1}));
  uvec2 range=lampRead((face*${size}+tile.y)*${size}+tile.x);
  for(int i=int(range.x);i<int(range.y);++i) {
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
}`;
