import { roundedBoxGLSL } from './rounded-box.js';
export const fragment =
  `#version 300 es
precision highp float;precision highp int;
uniform highp sampler2D objects,field;
uniform vec3 eye,firstRay,stepX,stepY,lightPos,lamp;
uniform float intensity;uniform int mode;
flat in int objectId;flat in int visible;out vec4 color;
` +
  roundedBoxGLSL +
  `
vec4 item(int k){return texelFetch(objects,ivec2(k,objectId),0);}
vec4 entry(int i){int width=textureSize(field,0).x;return texelFetch(field,ivec2(i%width,i/width),0);}
vec3 surfaceField(int index, vec2 uv){
 if(index<0)return vec3(0.);
 for(;;){
   vec4 node=entry(index);
   if(node.x<0.)break;
   index=int((node.x==0.?uv.x:uv.y)<node.y?node.z:node.w);
 }
 vec4 box=entry(index+1);ivec4 indices=ivec4(entry(index+2));
 vec2 f=clamp((uv-box.xz)/(box.yw-box.xz),0.,1.);
 return mix(mix(entry(indices.x).rgb,entry(indices.y).rgb,f.x),
            mix(entry(indices.z).rgb,entry(indices.w).rgb,f.x),f.y);
}
void main(){
 vec3 ray=firstRay+floor(gl_FragCoord.x)*stepX+floor(gl_FragCoord.y)*stepY;
 vec4 c=item(0),a=item(1),b=item(2),z=item(3);vec3 o=eye-c.xyz,p,n;int slot;vec2 uv;
 if(c.w==0.){
  p=eye-dot(o,z.xyz)/dot(ray,z.xyz)*ray;
  slot=int(dot(z.xyz,ray)>0.);n=slot==0?z.xyz:-z.xyz;
  uv=.5+.5*vec2(dot(p-c.xyz,a.xyz),dot(p-c.xyz,b.xyz));
 }else{
  mat3 frame=mat3(a.xyz,b.xyz,z.xyz);vec3 h=vec3(a.w,b.w,z.w),core=h-c.w;
  vec3 ro=o*frame,rd=ray*frame;float t;if(!intersectRoundedBox(ro,rd,h,c.w,0.,1.e25,t))discard;
  p=eye+t*ray;vec3 pos=ro+t*rd,normal=roundedBoxNormal(pos,h,c.w);n=frame*normal;
  bvec3 on=greaterThan(abs(normal),vec3(.00001));int m=int(on.x)+int(on.y)+int(on.z);
  if(m==1){int d=on.x?0:(on.y?1:2),i=(d+1)%3,j=(d+2)%3;slot=2*d+int(normal[d]<0.);
   uv=.5+.5*vec2(pos[i],pos[j])/max(vec2(core[i],core[j]),vec2(1e-9));
  }else if(m==2){int d=!on.x?0:(!on.y?1:2),i=(d+1)%3,j=(d+2)%3;
   slot=6+4*d+2*int(normal[i]<0.)+int(normal[j]<0.);
   uv=vec2(.5+.5*pos[d]/max(core[d],1e-9),atan(abs(normal[j]),abs(normal[i]))*.63661977237);
  }else{slot=18+int(normal.x<0.)+2*int(normal.y<0.)+4*int(normal.z<0.);
   uv=vec2(asin(min(1.,abs(normal.z))),atan(abs(normal.y),abs(normal.x)))*.63661977237;}
 }
 vec3 bounced=surfaceField(int(item(5+slot).x),uv);
 vec3 d=lightPos-p;float r2=dot(d,d);vec3 direct=lamp*(float(visible)*max(0.,dot(n,d))/(r2*sqrt(r2)));
 vec3 x=item(4).rgb*intensity*((mode==1?vec3(0):bounced)+(mode==2?vec3(0):direct));x=max(vec3(0),x)/(1.+max(vec3(0),x));
 vec3 rgb=mix(12.92*x,1.055*pow(x,vec3(1./2.4))-.055,greaterThan(x,vec3(.0031308)));
 ivec2 pixel=ivec2(gl_FragCoord.xy);int m=pixel.x^pixel.y,y=pixel.y;
 float q=(float(32*(m&1)+16*(y&1)+8*((m>>1)&1)+4*((y>>1)&1)+2*((m>>2)&1)+((y>>2)&1))+.5)/64.;
 color=vec4(floor(255.*rgb+q)/255.,1.);
}`;
export function displayField({
  illumination: {
    geometry: { charts, shapes, vertices, packet },
    indirect: incident,
  },
}) {
  const objects = new Float32Array(31 * 4 * shapes.length);
  for (const [i, s] of shapes.entries()) {
    const record = packet.shapes.subarray(19 * i, 19 * i + 19),
      base = 124 * i;
    objects.set([...s.center, s.radius || 0], base);
    for (let d = 0; d < 3; d++)
      objects.set([...record.slice(3 + 3 * d, 6 + 3 * d), record[12 + d]], base + 4 + 4 * d);
    objects.set(s.color, base + 16);
  }
  const table = [];
  for (let i = 0; i < vertices.length; i++)
    table.push(incident[3 * i], incident[3 * i + 1], incident[3 * i + 2], 0);
  function store(node) {
    const index = table.length / 4;
    if (node.children) {
      table.push(node.axis, node.cut, 0, 0);
      for (let j = 0; j < 2; j++) table[4 * index + 2 + j] = store(node.children[j]);
    } else table.push(-1, 0, 0, 0, ...node.bounds, ...node.ids);
    return index;
  }
  for (const g of charts) objects[124 * g.object + 20 + 4 * g.slot] = g.empty ? -1 : store(g.domain);
  const width = Math.min(1024, Math.max(1, table.length / 4)),
    height = Math.max(1, Math.ceil(table.length / (4 * width)));
  const tex = new Float32Array(4 * width * height);
  tex.set(table);
  return { objects, tex, width, height };
}
