// Surface locations at which the continuous incident diffuse field is estimated.
// A common planar parameterization gives adjoining pieces identical values at
// their join. Reflectance belongs to the visible material and is applied later.
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const mul=(a,k)=>a.map(x=>x*k);
const add=(a,b,k=1)=>a.map((x,i)=>x+k*b[i]);
const unit=a=>mul(a,1/Math.hypot(...a));
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));

function sheetFrame(rect){
  const area=cross(rect.axisU,rect.axisV),jac=Math.hypot(...area),n=mul(area,1/jac);
  const corners=[[-1,-1],[1,-1],[1,1],[-1,1]].map(([u,v])=>add(add(rect.center,rect.axisU,u),rect.axisV,v));
  return {rect,n,corners,du:mul(cross(rect.axisV,n),1/jac),dv:mul(cross(n,rect.axisU),1/jac)};
}

function commonEdge(a,b,epsilon){
  for(let i=0;i<4;i++)for(let j=0;j<4;j++){
    const p=a.corners[i],q=b.corners[j],e=add(a.corners[(i+1)%4],p,-1),f=add(b.corners[(j+1)%4],q,-1);
    const length=Math.hypot(...e),direction=mul(e,1/length),offset=add(q,p,-1);
    if(Math.hypot(...cross(direction,unit(f)))>1e-10 || Math.hypot(...cross(offset,direction))>epsilon)continue;
    const inward=cross(a.n,direction);
    if(dot(add(a.rect.center,p,-1),inward)*dot(add(b.rect.center,p,-1),inward)>=0)continue;
    const t=dot(offset,direction),end=t+dot(f,direction);
    const lo=Math.max(0,Math.min(t,end)),hi=Math.min(length,Math.max(t,end));
    if(hi-lo>epsilon)return [add(p,direction,lo),add(p,direction,hi)];
  }
  return null;
}
function barrierAcross(edge,face,normal,epsilon){
  const a=add(edge[0],face.rect.center,-1),b=add(edge[1],face.rect.center,-1);
  if(Math.abs(dot(a,face.n))>epsilon || Math.abs(dot(b,face.n))>epsilon)return false;
  const height=-dot(a,normal),extent=Math.abs(dot(face.rect.axisU,normal))+Math.abs(dot(face.rect.axisV,normal));
  if(height+extent<=epsilon)return false;
  let lo=0,hi=1;
  for(const d of [face.du,face.dv]){
    const x=dot(a,d),delta=dot(b,d)-x;
    if(Math.abs(delta)<1e-12){if(Math.abs(x)>1+1e-12)return false;continue;}
    const u=(-1-x)/delta,v=(1-x)/delta;
    lo=Math.max(lo,Math.min(u,v));hi=Math.min(hi,Math.max(u,v));
  }
  return hi-lo>1e-10;
}
function components(members,normal,all){
  const parent=members.map((_,i)=>i);
  function root(i){while(parent[i]!==i)i=parent[i];return i;}
  for(let i=0;i<members.length;i++)for(let j=i+1;j<members.length;j++){
    const a=members[i].frame,b=members[j].frame;
    const epsilon=128*Number.EPSILON*Math.max(1,...a.rect.center.map(Math.abs),...b.rect.center.map(Math.abs),
      Math.hypot(...a.rect.axisU),Math.hypot(...a.rect.axisV),Math.hypot(...b.rect.axisU),Math.hypot(...b.rect.axisV));
    const edge=commonEdge(a,b,epsilon);
    if(!edge || all.some(f=>f!==a&&f!==b&&barrierAcross(edge,f,normal,epsilon)))continue;
    parent[root(i)]=root(j);
  }
  const groups=new Map();
  for(const [i,member] of members.entries()){
    const k=root(i);if(!groups.has(k))groups.set(k,[]);groups.get(k).push(member);
  }
  return groups.values();
}

export function fieldLayout(scene) {
  const slots=[], groups=new Map(), grids=[];
  const all=scene.rectangles.map(sheetFrame);
  for(const [id,frame] of all.entries()) {
    const {rect:r,n}=frame;
    let canonical=n;
    const major=n.findIndex(x=>Math.abs(x)>1e-8);
    if(n[major]<0) canonical=mul(n,-1);
    const plane=dot(canonical,r.center);
    const key=[...canonical,plane].map(x=>Math.round(x*1e8)).join('/');
    let g=groups.get(key);
    if(!g) {
      const u=unit(r.axisU),v=unit(cross(canonical,u));
      g={n:canonical,u,v,plane,members:[],lo:[Infinity,Infinity],hi:[-Infinity,-Infinity]};
      groups.set(key,g);
    }
    g.members.push({id,r,n,frame});slots[id]=[];
  }
  function grid(origin,u,v,n){
    const a=Math.hypot(...u),b=Math.hypot(...v);
    const du=mul(u,1/(a*a)),dv=mul(v,1/(b*b));
    const w=clamp(1+Math.ceil(a/1.4),2,10),h=clamp(1+Math.ceil(b/1.4),2,10);
    const g={origin,u,v,n,w,h,du:[...du,-dot(origin,du)],dv:[...dv,-dot(origin,dv)]};
    grids.push(g);return g;
  }
  for(const g of groups.values()){
    for(const sign of [1,-1]){
      const n=mul(g.n,sign);
      for(const group of components(g.members,n,all)){
        const lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
        for(const {r} of group)for(const [k,ax] of [g.u,g.v].entries()){
          const extent=Math.abs(dot(ax,r.axisU))+Math.abs(dot(ax,r.axisV)),c=dot(ax,r.center);
          lo[k]=Math.min(lo[k],c-extent);hi[k]=Math.max(hi[k],c+extent);
        }
        const o=add(add(mul(g.n,g.plane),g.u,lo[0]),g.v,lo[1]);
        const field=grid(o,mul(g.u,hi[0]-lo[0]),mul(g.v,hi[1]-lo[1]),n);
        for(const m of group)slots[m.id][dot(m.n,n)>0?0:1]=field;
      }
    }
  }
  for(const [i,box] of (scene.roundedBoxes||[]).entries()){
    const faces=[];
    for(let d=0;d<3;d++){
      const a=(d+1)%3,b=(d+2)%3;
      // Face samples live on the flat core section, just inside its fillet.
      const u=mul(box.axes[a],2*(box.halfSize[a]-box.radius*.999));
      const v=mul(box.axes[b],2*(box.halfSize[b]-box.radius*.999));
      for(const sign of [1,-1]){
        const n=mul(box.axes[d],sign),c=add(box.center,n,box.halfSize[d]);
        faces.push(grid(add(add(c,u,-.5),v,-.5),u,v,n));
      }
    }
    slots[scene.rectangles.length+i]=faces;
  }
  let x=0,y=0,rowHeight=0,count=0;
  const width=Math.max(16,Math.ceil(Math.sqrt(grids.reduce((s,g)=>s+g.w*g.h,0))));
  for(const g of grids){
    if(x+g.w>width){x=0;y+=rowHeight;rowHeight=0;}
    g.x=x;g.y=y;x+=g.w;rowHeight=Math.max(rowHeight,g.h);count+=g.w*g.h;
  }
  const height=Math.max(1,y+rowHeight);
  const points=new Float32Array(width*height*4),normals=new Float32Array(points.length);
  const descriptors=new Float32Array(Math.max(1,slots.length)*6*4*4);
  for(const g of grids){
    for(let j=0;j<g.h;j++)for(let i=0;i<g.w;i++){
      // Take the interior limit at a wall boundary. A point exactly in an
      // intersecting plane can otherwise launch paths straight through it.
      const p=add(add(g.origin,g.u,clamp(i/(g.w-1),.00015,.99985)),g.v,clamp(j/(g.h-1),.00015,.99985));
      const idx=((g.y+j)*width+g.x+i)*4;points.set([...p,1],idx);normals.set(g.n,idx);
    }
  }
  for(const [id,faces] of slots.entries())for(const [i,g] of faces.entries()){
    const k=(6*id+i)*16;
    descriptors.set([g.x,g.y,g.w,g.h],k);descriptors.set(g.du,k+4);descriptors.set(g.dv,k+8);
    descriptors.set([...g.n,faces.length===6?1:0],k+12);
  }
  return {width,height,points,normals,descriptors,count,objectCount:slots.length};
}

export const indirectGLSL=`
uniform highp sampler2D irradianceField, irradianceMap;
vec3 fieldFace(int row,vec3 p){
  vec4 grid=texelFetch(irradianceMap,ivec2(0,row),0);
  vec4 u=texelFetch(irradianceMap,ivec2(1,row),0),v=texelFetch(irradianceMap,ivec2(2,row),0);
  vec2 uv=clamp(vec2(dot(u.xyz,p)+u.w,dot(v.xyz,p)+v.w),0.,1.);
  vec2 address=grid.xy+vec2(.5)+uv*(grid.zw-1.);
  return texture(irradianceField,address/vec2(textureSize(irradianceField,0))).rgb;
}
vec3 indirectAt(int id,vec3 p,vec3 n){
 int row=6*id;
 vec4 f=texelFetch(irradianceMap,ivec2(3,row),0);
 if(f.w==0.)return fieldFace(row+int(dot(f.xyz,n)<0.),p);
 vec3 v=vec3(dot(n,f.xyz),dot(n,texelFetch(irradianceMap,ivec2(3,row+2),0).xyz),
  dot(n,texelFetch(irradianceMap,ivec2(3,row+4),0).xyz));
 return v.x*v.x*fieldFace(row+int(v.x<0.),p)+v.y*v.y*fieldFace(row+2+int(v.y<0.),p)
   +v.z*v.z*fieldFace(row+4+int(v.z<0.),p);
}
`;
