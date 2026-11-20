// The entire clear source rectangle lies above the receiver horizon. The
// projected solid angle is half the receiver-normal component of the contour
// integral of the unit directions. Each great-circle edge contributes once.
const preamble = `#version 300 es
precision highp float;
layout(location=0) in vec3 relativeCenter;
layout(location=1) in vec3 axisU;
layout(location=2) in vec3 axisV;
layout(location=3) in vec3 normal;
out float measure;
float edge(vec3 a, vec3 b) {
  vec3 axis = cross(a,b);
  float sine = length(axis);
  return sine == 0. ? 0. : dot(normal,axis) * atan(sine,dot(a,b)) / sine;
}
`;
const vertex =
  preamble +
  `void main() {
  vec3 a=normalize(relativeCenter-axisU-axisV), b=normalize(relativeCenter+axisU-axisV);
  vec3 c=normalize(relativeCenter+axisU+axisV), d=normalize(relativeCenter-axisU+axisV);
  measure=abs(edge(a,b)+edge(b,c)+edge(c,d)+edge(d,a))*(0.5/3.141592653589793);
  gl_Position=vec4(0,0,0,1); gl_PointSize=1.;
}`;

// A clipped aperture can have any number of directed boundary edges.
// Integrate the signed projected area. Adjacent cell edges cancel.
const boundaryVertex =
  preamble +
  `void main() {
  measure=edge(normalize(relativeCenter),normalize(axisU))*(0.5/3.141592653589793);
  gl_Position=vec4(0,0,0,1); gl_PointSize=1.;
}`;

export class AngularMeasure {
  constructor(gl, { boundary = false } = {}) {
    this.gl = gl;
    this.stats = { polygons: 0, batches: 0 };
    this.program = gl.createProgram();
    for (const [type, source] of [
      [gl.VERTEX_SHADER, boundary ? boundaryVertex : vertex],
      [
        gl.FRAGMENT_SHADER,
        '#version 300 es\nprecision highp float; out vec4 color; void main(){color=vec4(0);}',
      ],
    ]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
      gl.attachShader(this.program, shader);
      gl.deleteShader(shader);
    }
    gl.transformFeedbackVaryings(this.program, ['measure'], gl.INTERLEAVED_ATTRIBS);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
      throw Error(gl.getProgramInfoLog(this.program));
    this.input = gl.createBuffer();
    this.output = gl.createBuffer();
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.input);
    for (let i = 0; i < 4; i++) {
      // The boundary program has no third input.
      if (boundary && i === 2) continue;
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 48, 12 * i);
    }
    gl.bindVertexArray(null);
  }
  reserve(count) {
    if (count && (!this.values || this.values.length < count)) {
      this.values = new Float32Array(count);
      this.packed = new Float32Array(count * 12);
    }
  }
  put(index, tile, receiver) {
    const a = this.packed,
      k = index * 12;
    for (let j = 0; j < 3; j++) {
      a[k + j] = tile.c[j] - receiver.p[j];
      a[k + 3 + j] = tile.ax[j] * tile.width;
      a[k + 6 + j] = tile.v[j] * tile.height;
      a[k + 9 + j] = receiver.n[j];
    }
  }
  putEdge(index, from, to, normal, orientation) {
    const k = 12 * index;
    this.packed.set(from, k);
    this.packed.set(to, k + 3);
    for (let d = 0; d < 3; d++) this.packed[k + 9 + d] = orientation * normal[d];
  }
  integrate(count) {
    if (!count) return;
    const gl = this.gl;
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.input);
    gl.bufferData(gl.ARRAY_BUFFER, this.packed, gl.STREAM_DRAW, 0, count * 12);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, this.output);
    if ((this.outputBytes || 0) < 4 * count) {
      this.outputBytes = 4 * count;
      gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, this.outputBytes, gl.STREAM_READ);
    }
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.output);
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindBuffer(gl.COPY_READ_BUFFER, this.output);
    gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, this.values, 0, count);
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    gl.bindVertexArray(null);
    this.inputBytes = 48 * count;
    this.stats.polygons += count;
    this.stats.batches++;
  }
  dispose() {
    const gl = this.gl;
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.input);
    gl.deleteBuffer(this.output);
    gl.deleteProgram(this.program);
  }
}
