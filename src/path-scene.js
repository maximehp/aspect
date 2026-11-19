import { roundedBoxGLSL } from './rounded-box.js';

const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const scale=(a,k)=>a.map(x=>x*k);

export function pathScene(scene){
  const objects=[];
  for(const rect of scene.rectangles){
    const n=cross(rect.axisU,rect.axisV),area=Math.hypot(...n),normal=scale(n,1/area);
    const extent=rect.center.map((_,i)=>Math.abs(rect.axisU[i])+Math.abs(rect.axisV[i]));
    objects.push({center:rect.center,lo:rect.center.map((x,i)=>x-extent[i]),hi:rect.center.map((x,i)=>x+extent[i]),
      data:[...rect.center,0,...normal,0,...scale(cross(rect.axisV,normal),1/area),0,...scale(cross(normal,rect.axisU),1/area),0,...rect.color,0]});
  }
  for(const b of scene.roundedBoxes||[])objects.push({center:b.center,lo:b.bounds.min,hi:b.bounds.max,
    data:[...b.center,b.radius,...b.axes[0],b.halfSize[0],...b.axes[1],b.halfSize[1],...b.axes[2],b.halfSize[2],...b.color,0]});
  const bounds=[];
  function split(indices){
    const lo=[Infinity,Infinity,Infinity],hi=lo.map(x=>-x);
    const cmin=lo.slice(),cmax=hi.slice();
    for(const i of indices)for(let k=0;k<3;k++){
      const o=objects[i];
      lo[k]=Math.min(lo[k],o.lo[k]);hi[k]=Math.max(hi[k],o.hi[k]);
      cmin[k]=Math.min(cmin[k],o.center[k]);cmax[k]=Math.max(cmax[k],o.center[k]);
    }
    const offset=bounds.length;
    bounds.push(...lo,0,...hi,indices.length===1?indices[0]:-1);
    if(indices.length>1){
      const span=cmax.map((v,i)=>v-cmin[i]),k=span.indexOf(Math.max(...span));
      indices.sort((a,b)=>objects[a].center[k]-objects[b].center[k]);
      const pivot=indices.length>>1;split(indices.slice(0,pivot));split(indices.slice(pivot));
    }
    bounds[offset+3]=bounds.length/8;
  }
  if(objects.length)split(objects.map((_,i)=>i));
  return {shapes:new Float32Array(objects.flatMap(o=>o.data)),bounds:new Float32Array(bounds),objectCount:objects.length,nodeCount:bounds.length/8};
}

export const pathSceneGLSL=roundedBoxGLSL+`
uniform highp sampler2D shapeTable,nodeTable;
uniform int nodeCount;
const float rayMin=.000035;
struct SurfaceHit {float t; vec3 n; vec3 color;};
bool findHit(vec3 o,vec3 d,float limit,bool anyHit,out SurfaceHit hit){
 hit.t=limit;
 bool found=false;
 vec3 inv=1./mix(d,vec3(1.e-29),lessThan(abs(d),vec3(1.e-29)));
 int node=0;
 while(node<nodeCount){
   vec4 lo=texelFetch(nodeTable,ivec2(0,node),0),hi=texelFetch(nodeTable,ivec2(1,node),0);
   vec3 a=(lo.xyz-vec3(.000002)-o)*inv,b=(hi.xyz+vec3(.000002)-o)*inv;
   vec3 nearT=min(a,b),farT=max(a,b);
   float begin=max(rayMin,max(nearT.x,max(nearT.y,nearT.z)));
   float end=min(hit.t,min(farT.x,min(farT.y,farT.z)));
   if(end<begin){node=int(lo.w);continue;}
   node++;
   int id=int(hi.w);
   if(id<0)continue;
   vec4 c=texelFetch(shapeTable,ivec2(0,id),0),u=texelFetch(shapeTable,ivec2(1,id),0),
        v=texelFetch(shapeTable,ivec2(2,id),0),w=texelFetch(shapeTable,ivec2(3,id),0);
   vec3 rel=o-c.xyz,n;
   float t;
   if(c.w==0.){
     float den=dot(u.xyz,d);
     if(abs(den)<1.e-12)continue;
     t=-dot(rel,u.xyz)/den;
     if(t<=rayMin || t>=hit.t)continue;
     vec3 p=rel+t*d;
     if(max(abs(dot(p,v.xyz)),abs(dot(p,w.xyz)))>1.)continue;
     n=u.xyz*(den<0.?1.:-1.);
   }else{
     mat3 frame=mat3(u.xyz,v.xyz,w.xyz);
     vec3 origin=rel*frame,dir=d*frame,h=vec3(u.w,v.w,w.w);
     if(!intersectRoundedBox(origin,dir,h,c.w,rayMin,hit.t,t))continue;
     if(anyHit)return true;
     n=frame*roundedBoxNormal(origin+t*dir,h,c.w);
     if(dot(n,d)>0.)n=-n;
   }
   if(anyHit)return true;
   hit.t=t;hit.n=n;hit.color=texelFetch(shapeTable,ivec2(4,id),0).rgb;found=true;
 }
 return found;
}
`;
