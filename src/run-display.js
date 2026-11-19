// The CPU has already resolved the object and lamp visibility for each run.
// The fragment stage evaluates that one authored surface at the pixel center.
import { roundedBoxGLSL } from './rounded-box.js';
import { shapeRecord } from './fillet-transport.js';
import { dot, sub, cross } from './cpu-geometry.js';

export const vertex = `#version 300 es
layout(location=0) in ivec4 run;
layout(location=1) in int state;
uniform vec2 size;
flat out int objectId; flat out int visible;
void main(){
 float x=float(gl_VertexID==0?run.x:run.y);
 gl_Position=vec4(2.*vec2(x,float(run.z)+.5)/size-1.,0,1);
 objectId=run.w;visible=state;
}`;
export const fragment =
  `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D field,objects,coeffs;
uniform vec3 eye,firstRay,stepX,stepY,lightPos,lamp;
uniform float intensity;uniform int mode;
flat in int objectId;flat in int visible;
out vec4 color;
` +
  roundedBoxGLSL +
  `
vec4 item(int i){return texelFetch(objects,ivec2(i,objectId),0);}
float knot(int start,int i){return item(start+i/4)[i%4];}
float coordinate(float t,int start,int n){
 if(n<2)return clamp(t,0.,1.);
 int i=0;while(i+2<n && t>knot(start,i+1))i++;
 float a=knot(start,i),b=knot(start,i+1);
 return clamp((float(i)+(t-a)/(b-a))/float(n-1),0.,1.);
}
vec3 incident(vec3 p,int face){
 vec4 group=item(20+face);
 vec3 sum=vec3(0);float total=0.;
 for(int i=0;i<int(group.y);i++){
  int row=int(group.x)+i;
  vec4 shape=texelFetch(coeffs,ivec2(0,row),0), axis=texelFetch(coeffs,ivec2(1,row),0);
  vec3 d=p-shape.xyz; float z=dot(d,d)+.1*shape.w,w=shape.w/(z*z);
  sum+=w*(texelFetch(coeffs,ivec2(2,row),0).rgb+clamp(dot(d,axis.xyz),-1.,1.)*texelFetch(coeffs,ivec2(3,row),0).rgb);
  total+=w;
 }
 return sum/max(total,1.e-20);
}
void main(){
 vec3 ray=firstRay+floor(gl_FragCoord.x)*stepX+floor(gl_FragCoord.y)*stepY;
 vec4 center=item(0),a=item(1),b=item(2),c=item(3);
 vec3 rel=eye-center.xyz,n,p,ind=vec3(0);
 if(center.w==0.){
   float t=-dot(rel,c.xyz)/dot(ray,c.xyz);
   p=eye+t*ray;n=dot(c.xyz,ray)<0.?c.xyz:-c.xyz;
   vec4 du=item(5),dv=item(6),meta=item(7);
   vec2 uv=vec2(coordinate(dot(du.xyz,p)+du.w,8,int(meta.x)),coordinate(dot(dv.xyz,p)+dv.w,14,int(meta.y)));
   ivec2 dim=textureSize(field,0);
   ind=texture(field,(vec2(.5,meta.z+.5)+uv*vec2(float(dim.x-1),meta.w-1.))/vec2(dim)).rgb;
 }else{
   mat3 frame=mat3(a.xyz,b.xyz,c.xyz);vec3 h=vec3(a.w,b.w,c.w),o=rel*frame,d=ray*frame;float t;
   if(!intersectRoundedBox(o,d,h,center.w,0.,1.e25,t))discard;
   p=eye+t*ray;vec3 normal=roundedBoxNormal(o+t*d,h,center.w);n=frame*normal;
   for(int i=0;i<3;i++)if(abs(normal[i])>.00001)ind+=normal[i]*normal[i]*incident(p,2*i+int(normal[i]<0.));
 }
 vec3 L=lightPos-p;float z=dot(L,L);
 vec3 direct=lamp*(float(visible)*max(0.,dot(n,L))/(z*sqrt(z)));
 vec3 x=max(vec3(0),item(4).rgb*intensity*((mode==1?vec3(0):ind)+(mode==2?vec3(0):direct)));
 x=x/(1.+x);
 vec3 srgb=mix(12.92*x,1.055*pow(x,vec3(1./2.4))-.055,greaterThan(x,vec3(.0031308)));
 ivec2 pixel=ivec2(gl_FragCoord.xy);int m=pixel.x^pixel.y,y=pixel.y;
 float threshold=(float(32*(m&1)+16*(y&1)+8*((m>>1)&1)+4*((y>>1)&1)+2*((m>>2)&1)+((y>>2)&1))+.5)/64.;
 color=vec4(floor(255.*srgb+threshold)/255.,1);
}`;

export function drawTables(view, camera, runs) {
  const { scene, solution } = view,
    shapes = [...scene.rectangles, ...scene.roundedBoxes];
  const objects = new Float32Array(shapes.length * 26 * 4),
    coeffs = [];
  const visible = new Uint8Array(shapes.length);
  for (let i = 3; i < runs.length; i += 5) visible[runs[i]] = 1;
  shapes.forEach((s, i) => {
    if (!visible[i]) return;
    const at = 104 * i,
      r = shapeRecord(s);
    objects.set([...s.center, s.radius || 0], at);
    for (let j = 0; j < 3; j++) objects.set([...r.slice(3 + 3 * j, 6 + 3 * j), r[12 + j]], at + 4 + 4 * j);
    objects.set(s.color, at + 16);
    if (!s.axes) {
      const side = Number(dot(sub(camera.position, s.center), cross(s.axisU, s.axisV)) < 0),
        g = view.faces[i][side];
      view.prepareChart(g);
      objects.set(g.du, at + 20);
      objects.set(g.dv, at + 24);
      objects.set(
        [g.coords?.[0].length || 0, g.coords?.[1].length || 0, g.displayRow, view.atlas.rows],
        at + 28,
      );
      if (g.coords) {
        for (let axis = 0; axis < 2; axis++) {
          if (g.coords[axis].length > 24) throw Error('Surface coordinate table exceeds its chart capacity.');
          objects.set(g.coords[axis], at + 32 + 24 * axis);
        }
      }
    } else {
      const charts = solution.proxy[i - scene.rectangles.length];
      charts.forEach((chart, j) => {
        const sources = view.modeSources(chart, 0);
        objects.set([coeffs.length / 16, sources.length], at + 80 + 4 * j);
        for (const { r, rgb, slope } of sources)
          coeffs.push(...r.center, r.area, ...r.momentAxis, 0, ...rgb, 0, ...slope, 0);
      });
    }
  });
  return { objects, coeffs: new Float32Array(coeffs.length ? coeffs : 16) };
}
