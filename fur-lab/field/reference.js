'use strict';
// Independent forward geometry reference. This is the only path that creates
// strand instances. Warp actual rings forward, connect them, and depth-test.
function makeStrandReference(gl,program){
  const vert=`#version 300 es
  precision highp float;
  uniform ivec4 limits;uniform vec4 axes,cam,stroke;uniform vec4 brush;
  uniform vec2 viewdir; uniform vec3 eye; uniform vec2 mesh;
  out vec3 rgb;
  const float H=3.8, PI=3.141592653589793;
  vec2 guide(int side,float t){
    float a=side==1?.47:.69,p=side==1?2.1:.35,w=side==1?-1.62:1.03;
    float angle=p+w*2.*PI*t;
    return a*(vec2(cos(angle),sin(angle))-vec2(cos(p),sin(p)))+vec2(side==1?.28:-.34,.18)*t;
  }
  vec2 warp(vec2 p,float t){
    if(brush.y<.5)return p;
    vec2 v=p-stroke.xy, n=vec2(-stroke.w,stroke.z);
    float s=abs(dot(v,stroke.zw)),q=dot(v,n),d=abs(q);
    float g=max(0.,min(d/.8,(5.-d)/4.2))*sign(q);
    float f=clamp((brush.x+2.-s)*.5,0.,1.);
    return p+n*(2.65*t*t*g*f);
  }
  void main(){
    int i=limits.x+gl_InstanceID%limits.z,j=limits.y+gl_InstanceID/limits.z,side=i<0?0:1;
    int ring=gl_VertexID/(int(mesh.x)+1),sector=gl_VertexID%(int(mesh.x)+1);
    float t=min(1.,float(ring)/mesh.y),angle=2.*PI*float(sector)/mesh.x;
    float r=ring>int(mesh.y)?0.:.168*(1.-.57*t);
    vec2 root=vec2(float(i)+.5+((j&1)==1?.16:-.16),float(j)+.5);
    vec2 point=root+r*vec2(cos(angle),sin(angle)),p=warp(point+guide(side,t),t);
    vec2 tangentXY=(warp(point+guide(side,t+.001),t+.001)-warp(point+guide(side,t-.001),t-.001))*500.;
    vec3 T=normalize(vec3(tangentXY,H)),L=vec3(-.5,-.4,sqrt(.59));
    float h=dot(T,normalize(eye+L));
    float lum=(.34+.40*sqrt(max(0.,1.-pow(abs(dot(T,L)),2.)))+.43*pow(max(0.,1.-h*h),8.))*(.73+.27*t);
    rgb=(side==0?vec3(177.,107.,61.):vec3(149.,171.,184.))*lum/255.;
    vec2 v=p-t*viewdir-cam.xy;
    gl_Position=vec4(dot(v,axes.xy),dot(v,axes.zw),.999-1.998*t,1.);
  }`;
  const prog=program(vert,`#version 300 es
  precision highp float;in vec3 rgb;out vec4 color;void main(){color=vec4(rgb,1.);}`);
  const loc=Object.fromEntries(['limits','axes','cam','stroke','brush','viewdir','eye','mesh'].map(k=>[k,gl.getUniformLocation(prog,k)]));
  const vao=gl.createVertexArray(),ebo=gl.createBuffer();let prev='';
  return function draw(m,b,levels=40,sectors=12){
    gl.useProgram(prog);gl.bindVertexArray(vao);
    if(prev!==levels+'/'+sectors){
      let indices=[];for(let k=0;k<=levels;k++)for(let s=0;s<sectors;s++){
        let a=k*(sectors+1)+s,c=a+sectors+1;indices.push(a,a+1,c,c,a+1,c+1);
      }
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ebo);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,new Uint16Array(indices),gl.STATIC_DRAW);prev=levels+'/'+sectors;
    }
    const ex=Math.abs(m.rx)+Math.abs(m.ux),ey=Math.abs(m.ry)+Math.abs(m.uy);
    // Includes the entire view prism, maximum brush offset, curl, radius and row stagger.
    const x0=Math.max(-5000,Math.floor(m.ox-ex+Math.min(0,m.vx)-5)),x1=Math.min(4999,Math.ceil(m.ox+ex+Math.max(0,m.vx)+5));
    const y0=Math.max(-5000,Math.floor(m.oy-ey+Math.min(0,m.vy)-5)),y1=Math.min(4999,Math.ceil(m.oy+ey+Math.max(0,m.vy)+5));
    const nx=x1-x0+1,ny=y1-y0+1;
    const iz=1/(m.rx*m.rx+m.ry*m.ry),iu=1/(m.ux*m.ux+m.uy*m.uy);
    gl.uniform4i(loc.limits,x0,y0,nx,ny);gl.uniform4f(loc.axes,m.rx*iz,m.ry*iz,m.ux*iu,m.uy*iu);
    gl.uniform4f(loc.cam,m.ox,m.oy,0,0);gl.uniform2f(loc.viewdir,m.vx,m.vy);gl.uniform3f(loc.eye,Math.cos(m.az)*Math.cos(m.el),Math.sin(m.az)*Math.cos(m.el),Math.sin(m.el));
    gl.uniform4f(loc.stroke,b?.x||0,b?.y||0,b?.sx||0,b?.sy||0);gl.uniform4f(loc.brush,b?.L||0,b?1:0,0,0);gl.uniform2f(loc.mesh,sectors,levels);
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.clear(gl.DEPTH_BUFFER_BIT);
    if(nx>0&&ny>0)gl.drawElementsInstanced(gl.TRIANGLES,6*(levels+1)*sectors,gl.UNSIGNED_SHORT,0,nx*ny);
    gl.disable(gl.DEPTH_TEST);gl.bindVertexArray(null);return Math.max(0,nx)*Math.max(0,ny);
  };
}
