import { CpuView } from './cpu-view.js';
import { Equilibrium } from './equilibrium.js';
import { createSweep } from './beam-exchange.js';
const sweep = await createSweep(
  await (await fetch(new URL('./beam-sweep.wasm', import.meta.url))).arrayBuffer(),
);
function cameraMatrix({ position, yaw, pitch, fov }, aspect) {
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw),
    cp = Math.cos(pitch),
    sp = Math.sin(pitch);
  const vertical = 1 / Math.tan(fov / 2),
    horizontal = vertical / aspect;
  const near = 0.035,
    far = 500,
    z = (far + near) / (far - near),
    offset = (-2 * far * near) / (far - near);
  const [x, y, w] = position,
    right = cy * x + sy * w,
    up = -sy * sp * x + cp * y + cy * sp * w,
    forward = sy * cp * x + sp * y - cy * cp * w;
  return new Float32Array([
    cy * horizontal,
    -sy * sp * vertical,
    sy * cp * z,
    sy * cp,
    0,
    cp * vertical,
    sp * z,
    sp,
    sy * horizontal,
    cy * sp * vertical,
    -cy * cp * z,
    -cy * cp,
    -right * horizontal,
    -up * vertical,
    -forward * z + offset,
    -forward,
  ]);
}

const vertex = `#version 300 es
layout(location=0)in vec3 position;layout(location=1)in vec2 address;layout(location=2)in vec3 reflectance;
layout(location=3)in vec4 supplied;layout(location=4)in float visible;
uniform mat4 projection;out vec2 uv;flat out vec3 material;out vec4 value;flat out float lit;
void main(){gl_Position=projection*vec4(position,1.);uv=address;material=reflectance;value=supplied;lit=visible;}`;
const fragment = `#version 300 es
precision highp float;
in vec2 uv;flat in vec3 material;in vec4 value;flat in float lit;
uniform sampler2D field;uniform vec3 lamp;uniform float intensity;uniform int mode;
out vec4 color;
void main(){vec4 f=uv.x<0.?value:texture(field,uv);
 vec3 x=material*intensity*((mode==1?vec3(0):f.rgb)+(mode==2?vec3(0):lit*f.a*lamp));
 x=x/(1.+x);color=vec4(mix(12.92*x,1.055*pow(x,vec3(1./2.4))-.055,greaterThan(x,vec3(.0031308))),1);}`;
export class LiveRenderer {
  constructor(canvas) {
    this.gl = canvas.getContext('webgl2', { alpha: false, antialias: true, depth: false });
    const gl = this.gl;
    if (!gl) throw Error('WebGL 2 required');
    const program = gl.createProgram();
    for (const [type, src] of [
      [gl.VERTEX_SHADER, vertex],
      [gl.FRAGMENT_SHADER, fragment],
    ]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s));
      gl.attachShader(program, s);
      gl.deleteShader(s);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
    gl.useProgram(program);
    this.program = program;
    this.uniforms = {};
    for (const name of ['projection', 'lamp', 'intensity', 'mode', 'field'])
      this.uniforms[name] = gl.getUniformLocation(program, name);
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    let off = 0;
    for (const [i, n] of [3, 2, 3, 4, 1].entries()) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, n, gl.FLOAT, false, 52, off);
      off += 4 * n;
    }
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    this.math = new Equilibrium(sweep);
    this.view = new CpuView();
    this.pixel = new Uint8Array(4);
  }
  setScene(scene) {
    const gl = this.gl,
      t = performance.now();
    this.scene = scene;
    this.eye = null;
    const result = this.math.update(scene);
    this.view.setScene(scene, result);
    this.iterations = result.iterations;
    this.cpuMs = performance.now() - t;
    const u = performance.now();
    const { width, height, tex } = this.view.atlas;
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.FLOAT, tex);
    this.uploadMs = performance.now() - u;
    this.stats = {
      bytes: result.bytes + tex.byteLength,
      iterations: result.iterations,
      residual: result.residual,
    };
  }
  draw(camera, intensity, size, mode = 'total') {
    const gl = this.gl,
      canvas = gl.canvas,
      ratio = devicePixelRatio || 1;
    const w = size?.[0] ?? Math.max(1, Math.round(canvas.clientWidth * ratio)),
      h = size?.[1] ?? Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== w) canvas.width = w;
    if (canvas.height !== h) canvas.height = h;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (intensity === 0) return;
    if (!this.eye || camera.position.some((x, i) => x !== this.eye[i])) {
      const start = performance.now(),
        vertices = this.view.drawList(camera.position);
      this.viewMs = performance.now() - start;
      const t = performance.now();
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
      this.uploadMs += performance.now() - t;
      this.vertexBytes = vertices.byteLength;
      this.count = vertices.length / 13;
      this.eye = camera.position.slice();
    }
    const u = this.uniforms;
    gl.uniformMatrix4fv(u.projection, false, cameraMatrix(camera, w / h));
    gl.uniform3fv(u.lamp, this.scene.lampRGB);
    gl.uniform1f(u.intensity, intensity);
    gl.uniform1i(u.mode, mode === 'direct' ? 1 : mode === 'indirect' ? 2 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, this.count);
  }
  synchronize() {
    this.gl.readPixels(0, 0, 1, 1, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.pixel);
  }
  dispose() {
    const gl = this.gl;
    gl.deleteTexture(this.texture);
    gl.deleteBuffer(this.buffer);
    gl.deleteProgram(this.program);
  }
}
