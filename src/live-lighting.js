import { fieldLayout, indirectGLSL } from './irradiance-layout.js';
import { pathScene, pathSceneGLSL } from './path-scene.js';
export { indirectGLSL };

const batchSize=16, sampleBudget=1024, lightCount=512;
const vertex=`#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0.,1.);}`;
const header=`#version 300 es
precision highp float;
precision highp int;
const float pi=3.141592653589793;
uniform vec3 lamp;
uniform int batchIndex;
uint mixBits(uint x){x^=x>>16;x*=0x7feb352du;x^=x>>15;x*=0x846ca68bu;return x^(x>>16);}
float unitFloat(uint x){return (float(x>>9)+.5)/8388608.;}
float random(inout uint state){state=state*747796405u+2891336453u;return unitFloat(mixBits(state));}
float encodeNormal(vec3 n){
 n/=dot(abs(n),vec3(1.)); vec2 uv=n.xy;
 if(n.z<0.)uv=(1.-abs(uv.yx))*mix(vec2(-1.),vec2(1.),greaterThanEqual(uv,vec2(0.)));
 ivec2 p=ivec2(round((uv*.5+.5)*4095.));return float(p.x+p.y*4096);
}
vec3 decodeNormal(float f){int p=int(f);vec2 u=vec2(p%4096,p/4096)/4095.*2.-1.;
 vec3 n=vec3(u,1.-abs(u.x)-abs(u.y));n.xy+=clamp(-n.z,0.,1.)*mix(vec2(1.),vec2(-1.),greaterThanEqual(n.xy,vec2(0.)));
 return normalize(n);
}`;
const lightSource=header+pathSceneGLSL+`
layout(location=0)out vec4 lightPosition;
layout(location=1)out vec4 lightPower;
void main(){
 int i=int(gl_FragCoord.x);uint batch=uint(batchIndex);
 vec2 shift=batchIndex==0?vec2(0.):vec2(unitFloat(mixBits(batch)),unitFloat(mixBits(batch^5411u)));
 float z=2.*fract((float(i)+.5)/${lightCount}.+shift.x)-1.;
 float phi=2.*pi*fract(float(i)*.61803398875+shift.y);
 vec3 d=vec3(sqrt(max(0.,1.-z*z))*vec2(cos(phi),sin(phi)),z);
 SurfaceHit hit;
 if(!findHit(lamp,d,1.e20,false,hit)){lightPosition=vec4(0.,0.,0.,-1.);lightPower=vec4(0.);return;}
 float g=max(0.,-dot(hit.n,d))/(hit.t*hit.t);
 lightPosition=vec4(lamp+d*hit.t,encodeNormal(hit.n));lightPower=vec4(hit.color*g,g/(4.*pi));
}`;
const pathSource=header+pathSceneGLSL+`
uniform highp sampler2D sites,siteNormals,lightVertices,lightPowers;
out vec4 sampleResult;
vec3 launch(vec3 n,vec2 u){
 vec3 helper=abs(n.z)>.95?vec3(0,1,0):vec3(0,0,1);
 vec3 a=normalize(cross(n,helper)), b=cross(n,a);
 float phi=2.*pi*u.y, r=sqrt(u.x);
 return r*(cos(phi)*a+sin(phi)*b)+sqrt(max(0.,1.-u.x))*n;
}
bool clearSegment(vec3 origin,vec3 d,float t){
 SurfaceHit unused;return !findHit(origin,d,t-2.*rayMin,true,unused);
}
void main(){
 ivec2 pixel=ivec2(gl_FragCoord.xy),uv=ivec2(pixel.x/${batchSize},pixel.y);
 vec4 site=texelFetch(sites,uv,0);
 if(site.w==0.){sampleResult=vec4(0.);return;}
 vec3 x=site.xyz, n=texelFetch(siteNormals,uv,0).xyz;
 int sampleId=batchIndex*${batchSize} + pixel.x%${batchSize};
 uint spatial=mixBits(uint(uv.x+uv.y*textureSize(sites,0).x));
 uint state=mixBits(uint(sampleId)+spatial);
 vec3 weight=vec3(1.),sum=vec3(0.);
 for(int depth=0;;depth++){
   float u=random(state),v=random(state);
   if(depth==0){u=fract((float(pixel.x%${batchSize})+.5)/${batchSize}.+unitFloat(mixBits(spatial^uint(batchIndex))));
    v=fract(float(pixel.x%${batchSize})*.61803398875+unitFloat(mixBits(spatial^uint(batchIndex)^711u)));}
   vec3 direction=launch(n,vec2(u,v)),origin=x+n*rayMin;
   int li=int(random(state)*${lightCount}.);
   vec4 z=texelFetch(lightVertices,ivec2(li,0),0);
   if(z.w>=0.){
      vec3 delta=z.xyz-x;float d2=dot(delta,delta);
      if(d2>rayMin*rayMin){
       float dist=sqrt(d2);vec3 dir=delta/dist;
       float pB=max(dot(n,dir),0.)*max(-dot(decodeNormal(z.w),dir),0.)/(pi*d2);
       if(pB>0. && clearSegment(origin,dir,dist)){
        vec4 light=texelFetch(lightPowers,ivec2(li,0),0);sum+=weight*light.rgb*(pB/(pB+light.w));
       }
      }
   }
   SurfaceHit y;
   if(!findHit(origin,direction,1.e20,false,y))break;
   vec3 next=origin+direction*y.t,delta=lamp-next;float dist=length(delta);
   vec3 toLamp=delta/dist;
   float g=max(0.,dot(y.n,toLamp))/(dist*dist);
   if(g>0. && clearSegment(next+y.n*rayMin,toLamp,dist)){
    float pB=max(dot(n,direction),0.)*max(-dot(y.n,direction),0.)/(pi*y.t*y.t);
    sum+=weight*y.color*g*(pB/(pB+g/(4.*pi)));
   }
   weight*=y.color;
   float survival=clamp(max(max(weight.r,weight.g),weight.b),.08,.82);
   if(random(state)>=survival)break;
   weight/=survival;n=y.n;x=next;
 }
 sampleResult=vec4(sum,1.);
}`;
const reduceSource=header+`
uniform highp sampler2D paths,history;
uniform vec3 lampRGB;
out vec4 average;
void main(){ivec2 p=ivec2(gl_FragCoord.xy);vec3 sum=vec3(0.);
 for(int i=0;i<${batchSize};i++)sum+=texelFetch(paths,ivec2(p.x*${batchSize}+i,p.y),0).rgb;
 vec3 previous=batchIndex==0?vec3(0.):texelFetch(history,p,0).rgb;
 average=vec4(mix(previous,lampRGB*(sum/${batchSize}.),1./float(1+batchIndex)),1.);
}`;

function program(gl,fragment){
 const p=gl.createProgram(), shaders=[];
 try{
  for(const [type,src] of [[gl.VERTEX_SHADER,vertex],[gl.FRAGMENT_SHADER,fragment]]){
   const s=gl.createShader(type);shaders.push(s);gl.shaderSource(s,src);gl.compileShader(s);
   if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));
   gl.attachShader(p,s);
  }
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
 }catch(e){gl.deleteProgram(p);throw e;}finally{for(const s of shaders)gl.deleteShader(s);}
 return p;
}

export function createLighting(gl){
 if(!gl.getExtension('EXT_color_buffer_float'))throw new Error('Diffuse transport needs floating point render targets.');
 const currentScene=program(gl,lightSource), trace=program(gl,pathSource), reduce=program(gl,reduceSource);
 const framebuffer=gl.createFramebuffer(),vao=gl.createVertexArray(),textures=[],locations=new WeakMap();
 const loc=(p,name)=>{if(!locations.has(p))locations.set(p,new Map());const cached=locations.get(p);
   if(!cached.has(name))cached.set(name,gl.getUniformLocation(p,name));return cached.get(name);};
 const tex=(linear=false)=>{const t=gl.createTexture();textures.push(t);gl.bindTexture(gl.TEXTURE_2D,t);
   const filter=linear?gl.LINEAR:gl.NEAREST;
   gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,filter);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,filter);
   gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return t;};
 const shapes=tex(),nodes=tex(),sites=tex(),normals=tex(),map=tex(),lightVertices=tex(),lightPowers=tex(),paths=tex(),mean=[tex(true),tex(true)];
 const storage=(t,w,h,data=null,half=false)=>{gl.bindTexture(gl.TEXTURE_2D,t);
   gl.texImage2D(gl.TEXTURE_2D,0,half?gl.RGBA16F:gl.RGBA32F,w,h,0,gl.RGBA,gl.FLOAT,data);};
 const bind=(p,name,unit,t)=>{gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t);gl.uniform1i(loc(p,name),unit);};
 const attach=(attachment,t)=>gl.framebufferTexture2D(gl.FRAMEBUFFER,attachment,gl.TEXTURE_2D,t,0);
 let scene=null,field=null,sceneData=null,batches=0,front=0;
 storage(lightVertices,lightCount,1);storage(lightPowers,lightCount,1);
 function draw(p,w,h){
   gl.useProgram(p);gl.uniform1i(loc(p,'batchIndex'),batches);gl.viewport(0,0,w,h);
   if(p!==reduce){
    gl.uniform3fv(loc(p,'lamp'),scene.lightPosition);gl.uniform1i(loc(p,'nodeCount'),sceneData.nodeCount);
    bind(p,'shapeTable',0,shapes);bind(p,'nodeTable',1,nodes);
   }
 }
 return {
  setScene(next){
   scene=next;field=fieldLayout(next);sceneData=pathScene(next);batches=0;front=0;
   gl.activeTexture(gl.TEXTURE0);
   storage(shapes,5,sceneData.objectCount,sceneData.shapes);storage(nodes,2,sceneData.nodeCount,sceneData.bounds);
   storage(sites,field.width,field.height,field.points);storage(normals,field.width,field.height,field.normals);
   storage(map,4,6*field.objectCount,field.descriptors);storage(paths,field.width*batchSize,field.height,null,true);
   for(const t of mean)storage(t,field.width,field.height,null,true);
  },
  step(){
   if(!field || batches*batchSize>=sampleBudget)return batches*batchSize;
   gl.bindFramebuffer(gl.FRAMEBUFFER,framebuffer);gl.bindVertexArray(vao);gl.disable(gl.DEPTH_TEST);gl.disable(gl.BLEND);gl.disable(gl.CULL_FACE);
   attach(gl.COLOR_ATTACHMENT0,lightVertices);attach(gl.COLOR_ATTACHMENT1,lightPowers);gl.drawBuffers([gl.COLOR_ATTACHMENT0,gl.COLOR_ATTACHMENT1]);
   draw(currentScene,lightCount,1);gl.drawArrays(gl.TRIANGLES,0,3);
   attach(gl.COLOR_ATTACHMENT0,paths);attach(gl.COLOR_ATTACHMENT1,null);gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
   draw(trace,field.width*batchSize,field.height);bind(trace,'sites',2,sites);bind(trace,'siteNormals',3,normals);
   bind(trace,'lightVertices',4,lightVertices);bind(trace,'lightPowers',5,lightPowers);gl.drawArrays(gl.TRIANGLES,0,3);
   attach(gl.COLOR_ATTACHMENT0,mean[1-front]);draw(reduce,field.width,field.height);
   bind(reduce,'paths',0,paths);bind(reduce,'history',1,mean[front]);gl.uniform3fv(loc(reduce,'lampRGB'),scene.lampRGB);
   gl.drawArrays(gl.TRIANGLES,0,3);front=1-front;batches++;
   gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.bindVertexArray(null);gl.enable(gl.DEPTH_TEST);
   return batches*batchSize;
  },
  bind(p,unit=4){bind(p,'irradianceField',unit,mean[front]);bind(p,'irradianceMap',unit+1,map);},
  get needsRefinement(){return !!field&&batches*batchSize<sampleBudget;},
  get shapeTexture(){return shapes;},
  get samples(){return batches*batchSize;},
  get stats(){return {supportCount:field?.count??0,samples:batches*batchSize};},
  dispose(){for(const t of textures)gl.deleteTexture(t);for(const p of [currentScene,trace,reduce])gl.deleteProgram(p);
   gl.deleteFramebuffer(framebuffer);gl.deleteVertexArray(vao);}
 };
}
