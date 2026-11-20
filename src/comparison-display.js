import { BeamWorkers } from './beam-workers.js';
import { CpuView } from './cpu-view.js';
import { ContactVisibility } from './contact-visibility.js';
import { ResidentContacts } from './resident-contacts.js';
import { GpuContactIntegrals } from './gpu-contact.js';
import { Equilibrium } from './equilibrium.js';
import { instanceSweep } from './beam-exchange.js';
import { Scanlines, FilletTransport } from './fillet-transport.js';
import { vertex, fragment, drawTables } from './run-display.js';
const [module, visibilityModule, filletModule] = await Promise.all(
  ['./beam-sweep.wasm', './contact-visibility.wasm', './fillet-kernel.wasm'].map((path) =>
    WebAssembly.compileStreaming(fetch(new URL(path, import.meta.url))),
  ),
);
export class LiveRenderer {
  constructor(canvas, { contacts = 'resident' } = {}) {
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
      'coeffs',
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
    this.tables = [gl.createTexture(), gl.createTexture()];
    this.scan = new Scanlines(filletModule);
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.LINEAR);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T])
      gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    this.math = new Equilibrium(
      instanceSweep(new WebAssembly.Instance(module)),
      new FilletTransport(filletModule),
    );
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
      if (typeof Worker === 'undefined' || this.serial) return this.math.update(scene);
      try {
        this.workers ??= new BeamWorkers();
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
        let preparedContacts,
          contactMs = 0;
        const result = await this.math.update(scene, this.workers, (geometry) => {
          const start = performance.now();
          preparedContacts = this.view.prepareContactGeometry(geometry);
          this.view.contactBackend?.prepareGeometry?.(preparedContacts);
          contactMs = performance.now() - start;
        });
        result.preparedScan = scan ? await scan : null;
        this.cancelScan = null;
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
    this.preparedScan = result.preparedScan;
    this.scanScene = null;
    this.cameraKey = null;
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
      const { objects, coeffs } = drawTables(this.view, camera, runs);
      this.viewMs = performance.now() - start;
      const upload = performance.now();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
      const atlas = this.view.atlas;
      if (this.uploadedAtlas !== atlas) {
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
      } else if (this.view.atlasChanged)
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, atlas.width, atlas.height, gl.RGBA, gl.FLOAT, atlas.tex);
      if (this.view.atlasChanged && this.view.contactDraws?.size) this.view.contactBackend.paint(this.view);
      this.view.atlasChanged = false;
      for (const [j, data, width] of [
        [0, objects, 26],
        [1, coeffs, 4],
      ]) {
        gl.activeTexture(gl.TEXTURE0 + 8 + j);
        gl.bindTexture(gl.TEXTURE_2D, this.tables[j]);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA32F,
          width,
          data.length / (4 * width),
          0,
          gl.RGBA,
          gl.FLOAT,
          data,
        );
      }
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.texture);
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
    gl.uniform1i(u.coeffs, 9);
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
  synchronize() {
    this.gl.readPixels(0, 0, 1, 1, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.pixel);
  }
  dispose() {
    const gl = this.gl;
    this.disposed = true;
    this.workers?.dispose();
    this.cancelScan?.();
    this.scanWorker?.terminate();
    this.view.contactBackend?.dispose();
    gl.deleteTexture(this.texture);
    gl.deleteBuffer(this.buffer);
    gl.deleteVertexArray(this.runVao);
    this.tables.forEach((t) => gl.deleteTexture(t));
    gl.deleteProgram(this.program);
  }
}
