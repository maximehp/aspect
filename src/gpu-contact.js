// Optional motion-benchmark backend. Each GPU invocation clips one visible
// source polygon to an azimuth sector and integrates both elevation bands.
import { dot, point, sub } from './cpu-geometry.js';

const vertexLimit = 16,
  width = 256;
const vertex = `#version 300 es
precision highp float;
precision highp int;
layout(location = 0) in ivec2 range;
uniform highp sampler2D positions;
uniform vec3 lower[8], upper[8], elevation[8];
out vec4 integral;
const int limit = ${vertexLimit};
void cut(in vec4 a[limit], int n, vec3 axis, out vec4 b[limit], out int count) {
  count = 0;
  if (n == 0)
    return;
  vec4 prev = a[n - 1];
  float dp = dot(prev.xyz, axis);
  for (int i = 0; i < n; i++) {
    vec4 cur = a[i];
    float dc = dot(cur.xyz, axis);
    if (dp * dc < 0.)
      b[count++] = prev + (dp / (dp - dc)) * (cur - prev);
    if (dc >= 0.)
      b[count++] = cur;
    prev = cur;
    dp = dc;
  }
}
float edge(vec3 a, vec3 b) {
  vec3 c = cross(a, b);
  float s = length(c);
  return s > 1e-12 ? c.z * atan(s, dot(a, b)) / s : 0.;
}
vec2 integratePart(in vec4 b[limit], int n) {
  if (n < 3)
    return vec2(0);
  float sum = 0., moment = 0.;
  int hits = 0;
  vec3 first = vec3(0), prev = vec3(0);
  for (int j = 0; j < n; j++) {
    moment += b[j].w;
    if (dot(b[j].xyz, b[j].xyz) <= 1e-24)
      continue;
    vec3 dir = normalize(b[j].xyz);
    if (hits == 0)
      first = dir;
    else
      sum += edge(prev, dir);
    prev = dir;
    hits++;
  }
  if (hits < 3)
    return vec2(0);
  float w = abs(sum + edge(prev, first)) / 6.283185307179586;
  return w > 1e-10 ? vec2(w, w * moment / float(n)) : vec2(0);
}
void main() {
  gl_Position = vec4(0, 0, 0, 1);
  gl_PointSize = 1.;
  integral = vec4(0);
  vec4 a[limit], b[limit];
  int k = gl_VertexID, n, m;
  vec3 lo = lower[k], hi = upper[k];
  float maxLo = -1e30, maxHi = -1e30;
  for (int i = 0; i < range.y; i++) {
    int t = range.x + i;
    vec4 q = texelFetch(positions, ivec2(t % ${width}, t / ${width}), 0);
    a[i] = q;
    maxLo = max(maxLo, dot(q.xyz, lo));
    maxHi = max(maxHi, dot(q.xyz, hi));
  }
  if (min(maxLo, maxHi) < 0.)
    return;
  cut(a, range.y, lo, b, n);
  if (n < 3)
    return;
  cut(b, n, hi, a, m);
  if (m < 3)
    return;
  cut(a, m, -elevation[k], b, n);
  integral.xy = integratePart(b, n);
  cut(a, m, elevation[k], b, n);
  integral.zw = integratePart(b, n);
}`;
const fragment = `#version 300 es
precision highp float;out vec4 color;void main(){color=vec4(0);}`;

export class GpuContactIntegrals {
  constructor(renderer) {
    this.renderer = renderer;
    const gl = (this.gl = renderer.gl);
    this.maxHeight = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.jobs = [];
    this.stats = { polygons: 0, fallback: 0, batches: 0, packMs: 0, dispatchReadMs: 0, reduceMs: 0 };
    const program = (this.program = gl.createProgram());
    for (const [type, code] of [
      [gl.VERTEX_SHADER, vertex],
      [gl.FRAGMENT_SHADER, fragment],
    ]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, code);
      gl.compileShader(shader);
      gl.attachShader(program, shader);
      const error = gl.getShaderParameter(shader, gl.COMPILE_STATUS) ? null : gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      if (error) {
        this.dispose();
        throw Error(error);
      }
    }
    gl.transformFeedbackVaryings(program, ['integral'], gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const error = gl.getProgramInfoLog(program);
      this.dispose();
      throw Error(error);
    }
    gl.useProgram(program);
    gl.uniform1i(gl.getUniformLocation(program, 'positions'), 1);
    const lower = [],
      upper = [],
      elevation = [];
    for (let k = 0; k < 8; k++) {
      const mid = (k * Math.PI) / 4,
        a = mid - Math.PI / 8,
        b = mid + Math.PI / 8;
      lower.push(-Math.sin(a), Math.cos(a), 0);
      upper.push(Math.sin(b), -Math.cos(b), 0);
      elevation.push(-Math.cos(mid), -Math.sin(mid), 1);
    }
    for (const [name, data] of [
      ['lower', lower],
      ['upper', upper],
      ['elevation', elevation],
    ])
      gl.uniform3fv(gl.getUniformLocation(program, name + '[0]'), data);
    this.vao = gl.createVertexArray();
    this.ranges = gl.createBuffer();
    this.output = gl.createBuffer();
    this.feedback = gl.createTransformFeedback();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.ranges);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribIPointer(0, 2, gl.INT, 0, 0);
    gl.vertexAttribDivisor(0, 1);
    gl.bindVertexArray(null);
    this.texture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
      gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindBuffer(gl.ARRAY_BUFFER, renderer.buffer);
    gl.useProgram(renderer.program);
  }
  enqueue(mode, side, eye, polygon, feature, emit) {
    // A convex polygon gains at most one vertex per clipping plane.
    if (polygon.uv.length + 3 > vertexLimit) {
      this.stats.fallback++;
      return false;
    }
    this.jobs.push({ mode, side, eye, polygon, feature, emit });
    return true;
  }
  flush() {
    if (!this.jobs.length) return;
    const gl = this.gl,
      jobs = this.jobs;
    let begin = 0;
    try {
      while (begin < jobs.length) {
        let end = begin,
          count = 0;
        while (end < jobs.length && count + jobs[end].polygon.uv.length <= width * this.maxHeight)
          count += jobs[end++].polygon.uv.length;
        const start = performance.now(),
          n = end - begin,
          h = Math.ceil(count / width);
        if (!this.packed || this.packed.length < h * width * 4) this.packed = new Float32Array(h * width * 4);
        if (!this.indices || this.indices.length < n * 2) {
          this.indices = new Int32Array(n * 2);
          this.values = new Float32Array(n * 32);
        }
        let t = 0;
        for (let i = begin; i < end; i++) {
          const { mode, eye, polygon, feature } = jobs[i];
          this.indices[(i - begin) * 2] = t;
          this.indices[(i - begin) * 2 + 1] = polygon.uv.length;
          for (const [u, v] of polygon.uv) {
            const q = point(mode, u, v),
              d = sub(q, eye),
              off = t++ * 4;
            this.packed[off] = dot(d, feature.u);
            this.packed[off + 1] = dot(d, feature.v);
            this.packed[off + 2] = dot(d, feature.n);
            this.packed[off + 3] = dot(sub(q, mode.center), mode.momentAxis);
          }
        }
        const packedAt = performance.now();
        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.ranges);
        gl.bufferData(gl.ARRAY_BUFFER, this.indices, gl.STREAM_DRAW, 0, n * 2);
        gl.activeTexture(gl.TEXTURE1);
        if (!this.textureHeight || this.textureHeight < h) {
          this.textureHeight = h;
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, h, 0, gl.RGBA, gl.FLOAT, null);
        }
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, h, gl.RGBA, gl.FLOAT, this.packed);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.feedback);
        gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, this.output);
        if (!this.outputBytes || this.outputBytes < n * 128) {
          this.outputBytes = n * 128;
          gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, this.outputBytes, gl.STREAM_READ);
        }
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.output);
        gl.enable(gl.RASTERIZER_DISCARD);
        gl.beginTransformFeedback(gl.POINTS);
        gl.drawArraysInstanced(gl.POINTS, 0, 8, n);
        gl.endTransformFeedback();
        gl.disable(gl.RASTERIZER_DISCARD);
        gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
        gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);
        gl.bindBuffer(gl.COPY_READ_BUFFER, this.output);
        gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, this.values, 0, n * 32);
        gl.bindBuffer(gl.COPY_READ_BUFFER, null);
        const readAt = performance.now();
        for (let i = begin; i < end; i++)
          for (let k = 0; k < 8; k++)
            for (let bin = k; bin < 16; bin += 8) {
              const j = (i - begin) * 32 + k * 4 + (bin < 8 ? 0 : 2),
                w = this.values[j];
              if (w > 1e-10) jobs[i].emit(jobs[i].side, w, this.values[j + 1], bin);
            }
        this.stats.batches++;
        this.stats.polygons += n;
        this.stats.packMs += packedAt - start;
        this.stats.dispatchReadMs += readAt - packedAt;
        this.stats.reduceMs += performance.now() - readAt;
        begin = end;
      }
    } finally {
      jobs.length = 0;
      gl.bindVertexArray(null);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.renderer.buffer);
      gl.useProgram(this.renderer.program);
    }
  }
  dispose() {
    const gl = this.gl;
    gl.deleteProgram(this.program);
    if (this.vao) {
      gl.deleteVertexArray(this.vao);
      gl.deleteBuffer(this.ranges);
      gl.deleteBuffer(this.output);
      gl.deleteTexture(this.texture);
      gl.deleteTransformFeedback(this.feedback);
    }
  }
}
