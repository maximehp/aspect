import { createLighting, indirectGLSL } from './live-lighting.js';
import { roundedBoxGLSL } from './rounded-box.js';
import { boxFaces, boxVertices, sheetVertices, LampVisibility, visibilityGLSL } from './lamp-visibility.js';

const vertexSource = `#version 300 es
layout(location=0) in vec4 vertex;
uniform mat4 viewProjection;
uniform highp sampler2D liveShapes;
flat out int id;
centroid out vec3 surfacePoint;
flat out vec3 sheetNormal, material;
void main() {
  id=int(vertex.w);
  surfacePoint=vertex.xyz;
  sheetNormal=texelFetch(liveShapes,ivec2(1,id),0).xyz;
  material=texelFetch(liveShapes,ivec2(4,id),0).xyz;
  gl_Position=viewProjection*vec4(surfacePoint,1.);
}`;

function fragmentSource(rounded) {
  return `#version 300 es
precision highp float;
precision highp int;
centroid in vec3 surfacePoint;
flat in vec3 sheetNormal, material;
flat in int id;
uniform mat4 viewProjection;
uniform vec3 eye, lampPosition, lampRGB;
uniform float dimmer;
uniform int mode;
uniform highp sampler2D liveShapes;
out vec4 color;
vec4 shapeRow(int objectId,int column){return texelFetch(liveShapes,ivec2(column,objectId),0);}
${roundedBoxGLSL}
${visibilityGLSL}
${indirectGLSL}
void main() {
  vec3 p=surfacePoint,n;
  ${rounded ? `
  vec4 c=shapeRow(id,0),x=shapeRow(id,1),y=shapeRow(id,2),z=shapeRow(id,3);
  mat3 axes=mat3(x.xyz,y.xyz,z.xyz);
  vec3 d=normalize(surfacePoint-eye), o=(eye-c.xyz)*axes, dir=d*axes, h=vec3(x.w,y.w,z.w);
  float t;
  if(!intersectRoundedBox(o,dir,h,c.w,0.,1.e25,t))discard;
  p=eye+t*d;
  n=axes*roundedBoxNormal(o+t*dir,h,c.w);
  if(dot(n,d)>0.)n=-n;
  vec4 clip=viewProjection*vec4(p,1.);
  gl_FragDepth=clip.z/clip.w*.5+.5;
  ` : 'n=gl_FrontFacing?sheetNormal:-sheetNormal;'}
  vec3 light=vec3(0);
  if(dimmer>0.) {
    if(mode!=1) light=indirectAt(id,p,n);
    if(mode!=2) {
      vec3 toLamp=lampPosition-p;
      float d2=dot(toLamp,toLamp), height=dot(n,toLamp);
      if(height>0.&&d2>0.&&seesLamp(p,id)) light+=lampRGB*(height*inversesqrt(d2)/d2);
    }
  }
  vec3 linear=light*material*dimmer;
  vec3 mapped=linear/(1.+linear);
  vec3 srgb=mix(12.92*mapped,1.055*pow(mapped,vec3(1./2.4))-.055,greaterThan(mapped,vec3(.0031308)));
  color=vec4(srgb,1);
}`;
}

function program(gl, fragment) {
  const p=gl.createProgram(), shaders=[];
  try {
    for (const [type,source] of [[gl.VERTEX_SHADER,vertexSource],[gl.FRAGMENT_SHADER,fragment]]) {
      const shader=gl.createShader(type);shaders.push(shader);
      gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(p,shader);
    }
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
    const locations={};
    for(const name of ['viewProjection','eye','lampPosition','lampRGB','dimmer','mode','liveShapes','lampTiles'])
      locations[name]=gl.getUniformLocation(p,name);
    return {p,locations};
  } catch(e) {gl.deleteProgram(p);throw e;}
  finally {for(const shader of shaders)gl.deleteShader(shader);}
}

function cameraMatrix({position,yaw,pitch,fov},aspect) {
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const vertical=1/Math.tan(fov/2),horizontal=vertical/aspect;
  const near=.035,far=500,z=(far+near)/(far-near),offset=-2*far*near/(far-near);
  const [x,y,w]=position, right=cy*x+sy*w,up=-sy*sp*x+cp*y+cy*sp*w,forward=sy*cp*x+sp*y-cy*cp*w;
  return new Float32Array([cy*horizontal,-sy*sp*vertical,sy*cp*z,sy*cp,0,cp*vertical,sp*z,sp,
    sy*horizontal,cy*sp*vertical,-cy*cp*z,-cy*cp,-right*horizontal,-up*vertical,-forward*z+offset,-forward]);
}

export class LiveRenderer {
  constructor(canvas) {
    const gl=canvas.getContext('webgl2',{alpha:false,depth:true,antialias:true});
    if(!gl)throw new Error('This room needs WebGL 2 with hardware acceleration.');
    this.gl=gl;
    this.maxTextureSize=gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.flat=program(gl,fragmentSource(false));
    this.solid=program(gl,fragmentSource(true));
    this.lighting=createLighting(gl);
    this.lampIndex=new LampVisibility();
    this.tiles=gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D,this.tiles);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    this.vao=gl.createVertexArray();this.buffer=gl.createBuffer();
    gl.bindVertexArray(this.vao);gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);
    gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,4,gl.FLOAT,false,16,0);
    this.pixel=new Uint8Array(4);
  }

  setScene(scene) {
    const gl=this.gl;
    this.scene=scene;
    this.lighting.setScene(scene);
    const shapes=[...scene.rectangles,...(scene.roundedBoxes||[])];
    const tiles=this.lampIndex.build(shapes,scene.lightPosition,this.maxTextureSize);
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.tiles);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RG32UI,tiles.width,tiles.height,0,gl.RG_INTEGER,gl.UNSIGNED_INT,tiles.pixels);
    const flatCount=scene.rectangles.length*6,solidCount=(shapes.length-scene.rectangles.length)*36;
    const vertices=new Float32Array((flatCount+solidCount)*4);
    let next=0;
    function quad(points,face,id) {
      for (const j of [face[0],face[1],face[2],face[0],face[2],face[3]]) {
        const p=points[j];vertices[next++]=p[0];vertices[next++]=p[1];vertices[next++]=p[2];vertices[next++]=id;
      }
    }
    scene.rectangles.forEach((r,id)=>quad(sheetVertices(r),[0,1,2,3],id));
    (scene.roundedBoxes||[]).forEach((b,k)=>{
      const corners=boxVertices(b),id=scene.rectangles.length+k;
      for (const face of boxFaces) quad(corners,face,id);
    });
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,vertices,gl.DYNAMIC_DRAW);
    this.flatCount=flatCount;this.solidCount=solidCount;
    this.directStats={shadowBytes:tiles.pixels.byteLength,maxShadowCandidates:tiles.maxCandidates};
  }

  get samples(){return this.lighting.samples;}
  get needsFrame(){return this.lighting.needsRefinement;}
  get stats(){return {...this.directStats,...this.lighting.stats};}

  draw(camera,intensity,size=null,mode='total') {
    const gl=this.gl,canvas=gl.canvas,ratio=globalThis.devicePixelRatio||1;
    const width=size?.[0]??Math.max(1,Math.round(canvas.clientWidth*ratio));
    const height=size?.[1]??Math.max(1,Math.round(canvas.clientHeight*ratio));
    if(canvas.width!==width)canvas.width=width;
    if(canvas.height!==height)canvas.height=height;
    if(intensity>0 && mode!=='direct' && this.needsFrame)this.lighting.step();
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    gl.viewport(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight);
    gl.disable(gl.CULL_FACE);gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.depthMask(true);
    gl.depthFunc(gl.LESS);gl.clearColor(0,0,0,1);gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
    if(intensity===0)return;
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.lighting.shapeTexture);
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.tiles);
    const matrix=cameraMatrix(camera,gl.drawingBufferWidth/gl.drawingBufferHeight);
    const setup=({p,locations:u})=>{
      gl.useProgram(p);
      gl.uniformMatrix4fv(u.viewProjection,false,matrix);gl.uniform3fv(u.eye,camera.position);
      gl.uniform3fv(u.lampPosition,this.scene.lightPosition);gl.uniform3fv(u.lampRGB,this.scene.lampRGB);
      gl.uniform1f(u.dimmer,intensity);gl.uniform1i(u.mode,mode==='direct'?1:mode==='indirect'?2:0);
      gl.uniform1i(u.liveShapes,0);gl.uniform1i(u.lampTiles,1);
      this.lighting.bind(p,4);
    };
    if(this.flatCount) {setup(this.flat);gl.drawArrays(gl.TRIANGLES,0,this.flatCount);}
    if(this.solidCount) {
      setup(this.solid);
      // Rasterize only exits from the conservative box. Each fragment solves
      // the curved surface hit and writes its true depth, including inside views.
      gl.enable(gl.CULL_FACE);gl.cullFace(gl.FRONT);
      gl.drawArrays(gl.TRIANGLES,this.flatCount,this.solidCount);
      gl.disable(gl.CULL_FACE);
    }
  }

  synchronize(){this.gl.readPixels(0,0,1,1,this.gl.RGBA,this.gl.UNSIGNED_BYTE,this.pixel);}
  dispose(){
    const gl=this.gl;
    this.lighting.dispose();gl.deleteTexture(this.tiles);gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.buffer);gl.deleteProgram(this.flat.p);gl.deleteProgram(this.solid.p);
  }
}
