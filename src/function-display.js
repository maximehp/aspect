import { roundedBoxGLSL } from './rounded-box.js';
import { shapeRecord } from './fillet-transport.js';
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
float gridCoordinate(float t,int start,int count){
 int a=0,b=count;while(b>a+1){int m=(a+b)/2;if(t>=texelFetch(field,ivec2(m,start),0).a)a=m;else b=m;}
 float lo=texelFetch(field,ivec2(a,start),0).a,hi=texelFetch(field,ivec2(b,start),0).a;return float(a)+clamp((t-lo)/(hi-lo),0.,1.);
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
 vec4 grid=item(5+slot);vec2 offset=c.w>0.?clamp(uv,0.,1.)*grid.yz:vec2(gridCoordinate(uv.x,int(grid.x),int(grid.y)),gridCoordinate(uv.y,int(grid.x)+1,int(grid.z)));ivec2 pos=ivec2(floor(offset));vec2 f=fract(offset);ivec2 next=min(pos+1,ivec2(grid.yz));pos.y+=int(grid.x);next.y+=int(grid.x);
 vec3 bounced=mix(mix(texelFetch(field,pos,0).rgb,texelFetch(field,ivec2(next.x,pos.y),0).rgb,f.x),
                    mix(texelFetch(field,ivec2(pos.x,next.y),0).rgb,texelFetch(field,next,0).rgb,f.x),f.y);
 vec3 d=lightPos-p;float r2=dot(d,d);vec3 direct=lamp*(float(visible)*max(0.,dot(n,d))/(r2*sqrt(r2)));
 vec3 x=item(4).rgb*intensity*((mode==1?vec3(0):bounced)+(mode==2?vec3(0):direct));x=max(vec3(0),x)/(1.+max(vec3(0),x));
 vec3 rgb=mix(12.92*x,1.055*pow(x,vec3(1./2.4))-.055,greaterThan(x,vec3(.0031308)));
 ivec2 pixel=ivec2(gl_FragCoord.xy);int m=pixel.x^pixel.y,y=pixel.y;
 float q=(float(32*(m&1)+16*(y&1)+8*((m>>1)&1)+4*((y>>1)&1)+2*((m>>2)&1)+((y>>2)&1))+.5)/64.;
 color=vec4(floor(255.*rgb+q)/255.,1.);
}`;
export function displayField(solution) {
  const {
    geometry: { charts, shapes },
    incident,
  } = solution;
  const width = Math.max(...charts.map((c) => Math.max(c.cols, c.rows) + 1)),
    height = charts.reduce((n, c) => n + c.rows + 1, 0);
  const tex = new Float32Array(width * height * 4),
    objects = new Float32Array(31 * 4 * shapes.length);
  for (const [i, s] of shapes.entries()) {
    const base = i * 124,
      r = shapeRecord(s);
    objects.set([...s.center, s.radius || 0], base);
    for (let d = 0; d < 3; d++) objects.set([...r.slice(3 + 3 * d, 6 + 3 * d), r[12 + d]], base + 4 + 4 * d);
    objects.set(s.color, base + 16);
  }
  let row = 0;
  for (const c of charts) {
    objects.set([row, c.cols, c.rows, 0], 124 * c.object + 20 + 4 * c.slot);
    for (let j = 0; j <= c.rows; j++)
      for (let i = 0; i <= c.cols; i++) {
        const n = c.ids[j * (c.cols + 1) + i];
        if (n >= 0) tex.set(incident.subarray(3 * n, 3 * n + 3), 4 * ((row + j) * width + i));
      }
    for (let d = 0; d < 2; d++)
      for (let i = 0; i < c.knots[d].length; i++) tex[4 * ((row + d) * width + i) + 3] = c.knots[d][i];
    row += c.rows + 1;
  }
  return { objects, tex, width, height };
}
