import { AngularMeasure } from './angular-measure.js';
import { Scanlines } from './fillet-transport.js';
import { FunctionLighting } from './function-lighting.js';
import { vertex } from './run-display.js';
import { fragment, displayField } from './function-display.js';
const filletModule = await WebAssembly.compileStreaming(
  fetch(new URL('./fillet-kernel.wasm', import.meta.url)),
);
export class LiveRenderer {
  constructor(canvas, options) {
    this.gl = canvas.getContext('webgl2', { alpha: false, antialias: false, depth: false });
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
    for (const name of [
      'eye',
      'firstRay',
      'stepX',
      'stepY',
      'lightPos',
      'size',
      'lamp',
      'intensity',
      'mode',
      'field',
      'objects',
    ])
      this.uniforms[name] = gl.getUniformLocation(program, name);
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    this.runVao = gl.createVertexArray();
    gl.bindVertexArray(this.runVao);
    for (const [i, n, offset] of [
      [0, 4, 0],
      [1, 1, 16],
    ]) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribIPointer(i, n, gl.INT, 20, offset);
      gl.vertexAttribDivisor(i, 1);
    }
    gl.bindVertexArray(null);
    this.objects = gl.createTexture();
    this.scan = new Scanlines(filletModule);
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    this.angular = new AngularMeasure(gl, { indexed: true });
    this.math = new FunctionLighting(this.angular, options);
    this.pixel = new Uint8Array(4);
  }
  outputSize(size) {
    const canvas = this.gl.canvas,
      ratio = globalThis.devicePixelRatio || 1;
    return (
      size || [
        Math.max(1, Math.round(canvas.clientWidth * ratio)),
        Math.max(1, Math.round(canvas.clientHeight * ratio)),
      ]
    );
  }
  prepareScene(scene, camera, size) {
    const calculate = async () => {
      // A new pose can spend tens of milliseconds on CPU geometry before the
      // first worker wait. Let the browser commit the preceding draw first.
      if (this.drawAwaitingCommit) {
        this.drawAwaitingCommit = false;
        if (globalThis.scheduler?.yield) await globalThis.scheduler.yield();
      }
      if (this.disposed) throw Error('Renderer disposed.');

      let scan;
      if (camera) {
        const dimensions = this.outputSize(size),
          key = [...camera.position, camera.yaw, camera.pitch, camera.fov, ...dimensions].join('/');
        this.scanWorker ??= new Worker(new URL('./scan-worker.js', import.meta.url), { type: 'module' });
        scan = new Promise((resolve) => {
          this.scanWorker.onmessage = ({ data }) => resolve(data.error ? null : { ...data, key });
          this.scanWorker.onerror = () => resolve(null);
          this.cancelScan = () => resolve(null);
          this.scanWorker.postMessage({ scene, camera, size: dimensions });
        });
      }
      const result = await this.math.update(scene);
      result.preparedScan = scan ? await scan : null;
      this.cancelScan = null;
      return result;
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
    this.view = displayField(result);
    this.preparedScan = result.preparedScan;
    this.scanScene = this.cameraKey = null;
    this.timings = { ...result.timings, appearance: performance.now() - appearanceStart };
    this.iterations = result.iterations;
    this.cpuMs = performance.now() - t;
    this.uploadMs = 0;
    const { tex } = this.view;
    this.stats = {
      bytes: result.bytes + tex.byteLength + this.view.objects.byteLength,
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
    const key = [...camera.position, camera.yaw, camera.pitch, camera.fov, w, h].join('/');
    if (this.cameraKey !== key) {
      const start = performance.now();
      const prepared = this.preparedScan?.key === key ? this.preparedScan : null;
      if (!prepared && this.scanScene !== this.scene) {
        this.scan.prepare(this.scene);
        this.scanScene = this.scene;
      }
      const { runs, stats, base, dx, dy } = prepared || this.scan.render(camera, w, h);
      this.preparedScan = null;
      this.viewMs = performance.now() - start;
      const upload = performance.now(),
        v = this.view;
      if (this.uploadedAtlas !== v) {
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.texture);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, v.width, v.height, 0, gl.RGBA, gl.FLOAT, v.tex);
        gl.activeTexture(gl.TEXTURE8);
        gl.bindTexture(gl.TEXTURE_2D, this.objects);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA32F,
          31,
          this.scene.rectangles.length + this.scene.roundedBoxes.length,
          0,
          gl.RGBA,
          gl.FLOAT,
          v.objects,
        );
        gl.activeTexture(gl.TEXTURE0);
        this.uploadedAtlas = v;
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(gl.ARRAY_BUFFER, runs, gl.DYNAMIC_DRAW);
      this.count = runs.length / 5;
      this.vertexBytes = runs.byteLength;
      this.frameStats = {
        runs: this.count,
        grouped: stats[0],
        individual: stats[1],
        covered: stats[2],
        lampTests: stats[3],
        bytes: prepared?.bytes || this.scan.wasm.memory.buffer.byteLength,
        scanMs: prepared?.ms,
      };
      this.rayFrame = [base, dx, dy];
      this.uploadMs += performance.now() - upload;
      this.cameraKey = key;
    }
    gl.useProgram(this.program);
    gl.bindVertexArray(this.runVao);
    const u = this.uniforms;
    gl.uniform1i(u.objects, 8);
    gl.uniform1i(u.field, 0);
    gl.uniform2f(u.size, w, h);
    gl.uniform3fv(u.eye, camera.position);
    for (const [i, name] of ['firstRay', 'stepX', 'stepY'].entries())
      gl.uniform3fv(u[name], this.rayFrame[i]);
    gl.uniform3fv(u.lightPos, this.scene.lightPosition);
    gl.uniform3fv(u.lamp, this.scene.lampRGB);
    gl.uniform1f(u.intensity, intensity);
    gl.uniform1i(u.mode, mode === 'direct' ? 1 : mode === 'indirect' ? 2 : 0);
    gl.drawArraysInstanced(gl.LINES, 0, 2, this.count);
    gl.bindVertexArray(null);
  }
  countedBytes() {
    // Named engine payloads, including the single RGBA8 output. Browser swap
    // buffers, driver allocations and garbage-collected JS objects are opaque.
    const a = this.angular;
    return (
      this.stats.bytes +
      this.stats.workerBytes +
      this.frameStats.bytes +
      this.scan.wasm.memory.buffer.byteLength +
      this.gl.drawingBufferWidth * this.gl.drawingBufferHeight * 4 +
      2 * this.vertexBytes +
      this.view.tex.byteLength +
      this.view.objects.byteLength +
      (a.packed?.byteLength || 0) +
      (a.values?.byteLength || 0) +
      (a.outputBytes || 0) +
      (a.inputBytes || 0) +
      (a.tableBytes || 0)
    );
  }
  synchronize() {
    this.gl.readPixels(0, 0, 1, 1, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.pixel);
  }
  dispose() {
    const gl = this.gl;
    this.disposed = true;
    this.math.dispose();
    this.cancelScan?.();
    this.scanWorker?.terminate();
    this.angular.dispose();
    gl.deleteTexture(this.texture);
    gl.deleteBuffer(this.buffer);
    gl.deleteVertexArray(this.runVao);
    gl.deleteTexture(this.objects);
    gl.deleteProgram(this.program);
  }
}
