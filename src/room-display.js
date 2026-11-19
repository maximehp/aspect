import { BeamWorkers } from './beam-workers.js';
import { CpuView } from './cpu-view.js';
import { ContactVisibility } from './contact-visibility.js';
import { ResidentContacts } from './resident-contacts.js';
import { GpuContactIntegrals } from './gpu-contact.js';
import { Equilibrium } from './equilibrium.js';
import { instanceSweep } from './beam-exchange.js';
const [module, visibilityModule] = await Promise.all(
  ['./beam-sweep.wasm', './contact-visibility.wasm'].map((path) =>
    WebAssembly.compileStreaming(fetch(new URL(path, import.meta.url))),
  ),
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
uniform mat4 projection;uniform vec2 chartScale;out vec2 uv;flat out vec2 verticalLimits;flat out vec3 material;out vec4 value;flat out float lit;
void main(){gl_Position=projection*vec4(position,1.);uv=address;
 float bottom=floor(address.y/chartScale.x)*chartScale.x;
 verticalLimits=vec2(bottom+chartScale.y,bottom+chartScale.x-chartScale.y);material=reflectance;value=supplied;lit=visible;}`;
const fragment = `#version 300 es
precision highp float;
in vec2 uv;flat in vec2 verticalLimits;flat in vec3 material;in vec4 value;flat in float lit;
uniform sampler2D field;uniform vec3 lamp;uniform float intensity;uniform int mode;
out vec4 color;
// Ordered rounding keeps sub-code gradients visible in the 8-bit output.
// This threshold is fixed per framebuffer pixel across all scene revisions.
float roundingThreshold(ivec2 p){
  int x=p.x, y=p.y, m=x^y;
  int rank=32*(m&1)+16*(y&1)+8*((m>>1)&1)+4*((y>>1)&1)+2*((m>>2)&1)+((y>>2)&1);
  return (float(rank)+0.5)/64.;
}
void main(){vec4 f=uv.x<0.?value:texture(field,vec2(uv.x,clamp(uv.y,verticalLimits.x,verticalLimits.y)));
 vec3 x=material*intensity*((mode==1?vec3(0):f.rgb)+(mode==2?vec3(0):lit*f.a*lamp));
 x=x/(1.+x);
 vec3 srgb=mix(12.92*x,1.055*pow(x,vec3(1./2.4))-.055,greaterThan(x,vec3(.0031308)));
 color=vec4(floor(255.*srgb+roundingThreshold(ivec2(gl_FragCoord.xy)))/255.,1);}`;
export class LiveRenderer {
  constructor(canvas, { contacts = 'resident' } = {}) {
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
    for (const name of ['projection', 'chartScale', 'lamp', 'intensity', 'mode', 'field'])
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
    this.math = new Equilibrium(instanceSweep(new WebAssembly.Instance(module)));
    this.view = new CpuView();
    this.pixel = new Uint8Array(4);
    try {
      if (contacts === 'resident') {
        this.view.contactBackend = new ResidentContacts(this);
        this.view.contactBackend.visibility = new ContactVisibility(visibilityModule);
      } else if (contacts === 'gpu') this.view.contactBackend = new GpuContactIntegrals(this);
      else if (contacts !== 'cpu') throw Error('Unknown contact integrator: ' + contacts);
    } catch (error) {
      this.dispose();
      throw error;
    }
  }
  setScene(scene) {
    const t = performance.now();
    this.applyScene(scene, this.math.update(scene), t);
  }
  prepareScene(scene) {
    const calculate = async () => {
      // A new pose can spend tens of milliseconds on CPU geometry before the
      // first worker wait. Let the browser commit the preceding draw first.
      if (this.drawAwaitingCommit) {
        this.drawAwaitingCommit = false;
        if (globalThis.scheduler?.yield) await globalThis.scheduler.yield();
      }
      if (this.disposed) throw Error('Renderer disposed.');
      if (typeof Worker === 'undefined' || this.serial) return this.math.update(scene);
      try {
        this.workers ??= new BeamWorkers();
        let preparedContacts,
          contactMs = 0;
        const result = await this.math.update(scene, this.workers, (geometry) => {
          const start = performance.now();
          preparedContacts = this.view.prepareContactGeometry(geometry);
          contactMs = performance.now() - start;
        });
        result.preparedContacts = preparedContacts;
        result.timings.contactMs = contactMs;
        return result;
      } catch (error) {
        if (this.disposed || (this.workers && !this.workers.failed)) throw error;
        this.workers?.dispose();
        this.serial = true;
        return this.math.update(scene);
      }
    };
    const result = (this.preparations || Promise.resolve()).then(calculate);
    this.preparations = result.then(
      () => {},
      () => {},
    );
    return result;
  }
  applyScene(scene, result, t) {
    this.scene = scene;
    this.eye = null;
    this.viewMs = 0;
    const appearanceStart = performance.now();
    this.view.setScene(scene, result, true);
    this.view.contactBackend?.retainCharts?.(this.view.contacts);
    this.timings = { ...result.timings, appearance: performance.now() - appearanceStart };
    this.iterations = result.iterations;
    this.cpuMs = performance.now() - t;
    this.uploadMs = 0;
    const { tex } = this.view.atlas;
    this.stats = {
      bytes: result.bytes + tex.byteLength,
      workerBytes: result.workerBytes,
      links: result.links,
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
    this.drawAwaitingCommit = true;
    if (intensity === 0) return;
    const projection = cameraMatrix(camera, w / h);
    const maskChanged = this.view.updateCameraMask(projection);
    if (maskChanged || !this.eye || camera.position.some((x, i) => x !== this.eye[i])) {
      const start = performance.now(),
        vertices = this.view.drawList(camera.position, this.view.cameraMask);
      this.viewMs = performance.now() - start;
      const t = performance.now();
      if (this.view.atlasChanged || this.uploadedAtlas !== this.view.atlas) {
        const atlas = this.view.atlas;
        if (atlas !== this.uploadedAtlas) {
          gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA16F,
            atlas.width,
            atlas.height,
            0,
            gl.RGBA,
            gl.FLOAT,
            atlas.tex,
          );
          this.uploadedAtlas = atlas;
        } else {
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, atlas.width, atlas.height, gl.RGBA, gl.FLOAT, atlas.tex);
        }
        if (this.view.contactDraws?.size) this.view.contactBackend.paint(this.view);
        this.view.atlasChanged = false;
      }
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
      this.uploadMs += performance.now() - t;
      this.vertexBytes = vertices.byteLength;
      this.count = vertices.length / 13;
      this.eye = camera.position.slice();
    }
    const u = this.uniforms;
    const { rows, height } = this.view.atlas;
    gl.uniform2f(u.chartScale, rows / height, 0.5 / height);
    gl.uniformMatrix4fv(u.projection, false, projection);
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
    this.disposed = true;
    this.workers?.dispose();
    this.view.contactBackend?.dispose();
    gl.deleteTexture(this.texture);
    gl.deleteBuffer(this.buffer);
    gl.deleteProgram(this.program);
  }
}
