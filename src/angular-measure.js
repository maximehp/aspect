import { rectangleAperture } from './rectangle-aperture.js';
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

// Coordinates are shared across the whole batch. Jobs reference two authored
// geometry records and never upload four endpoint vectors per interaction.
const indexedVertex = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
layout(location=0) in ivec2 job;
uniform sampler2D charts;
uniform sampler2D receivers;
out vec4 measure;
vec4 fetchAt(sampler2D s,int i) {return texelFetch(s,ivec2(i&1023,i>>10),0);}
${rectangleAperture}
float edge(vec3 a,vec3 b,vec3 n){
  vec3 v=cross(a,b);float s=length(v);
  return s==0.?0.:dot(n,v)*atan(s,dot(a,b))/s;
}
void main() {
  int src=job.x<0?~job.x:job.x/8, i=4*src,j=3*job.y;
  vec4 c=fetchAt(charts,i),u=fetchAt(charts,i+1),v=fetchAt(charts,i+2);
  vec3 center=(c.xyz-fetchAt(receivers,j).xyz)+(fetchAt(charts,i+3).xyz-fetchAt(receivers,j+2).xyz),normal=fetchAt(receivers,j+1).xyz;
  if(job.x<0) {
    vec3 U=u.xyz*c.w,V=v.xyz*u.w, a=normalize(center-U-V),b=normalize(center+U-V),
      d=normalize(center-U+V),e=normalize(center+U+V);
    measure=vec4(abs(edge(a,b,normal)+edge(b,e,normal)+edge(e,d,normal)+edge(d,a,normal))*(0.5/3.14159265359),0,0,0);
  } else measure=bilinear(center,u.xyz,v.xyz,normal,c.w,u.w,job.x&7);
  gl_Position=vec4(0,0,0,1);gl_PointSize=1.;
}`;

export class AngularMeasure {
  constructor(gl, { boundary = false, indexed = false } = {}) {
    this.gl = gl;
    this.indexed = indexed;
    this.channels = indexed ? 4 : 1;
    this.stride = indexed ? 2 : 12;
    this.stats = { polygons: 0, batches: 0 };
    this.program = gl.createProgram();
    for (const [type, source] of [
      [gl.VERTEX_SHADER, indexed ? indexedVertex : boundary ? boundaryVertex : vertex],
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
    if (indexed) {
      gl.enableVertexAttribArray(0);
      gl.vertexAttribIPointer(0, 2, gl.INT, 8, 0);
      this.tables = [gl.createTexture(), gl.createTexture()];
      gl.useProgram(this.program);
      for (const [i, name] of ['charts', 'receivers'].entries())
        gl.uniform1i(gl.getUniformLocation(this.program, name), 4 + i);
    } else
      for (let i = 0; i < 4; i++) {
        // The boundary program has no third input.
        if (boundary && i === 2) continue;
        gl.enableVertexAttribArray(i);
        gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 48, 12 * i);
      }
    gl.bindVertexArray(null);
  }
  setGeometry(geometry) {
    const gl = this.gl,
      sizes = [4 * geometry.tiles.length, 3 * geometry.receivers.length];
    this.tableBytes = 0;
    for (let k = 0; k < 2; k++) {
      const height = Math.max(1, Math.ceil(sizes[k] / 1024)),
        data = new Float32Array(height * 4096);
      if (k === 0)
        geometry.tiles.forEach((t, i) => {
          data.set(t.c, 16 * i);
          data[16 * i + 3] = t.width;
          data.set(t.ax, 16 * i + 4);
          data[16 * i + 7] = t.height;
          data.set(t.v, 16 * i + 8);
          for (let a = 0; a < 3; a++) data[16 * i + 12 + a] = t.c[a] - data[16 * i + a];
        });
      else
        geometry.receivers.forEach((r, i) => {
          data.set(r.p, 12 * i);
          data.set(r.n, 12 * i + 4);
          for (let a = 0; a < 3; a++) data[12 * i + 8 + a] = r.p[a] - data[12 * i + a];
        });
      gl.activeTexture(gl.TEXTURE4 + k);
      gl.bindTexture(gl.TEXTURE_2D, this.tables[k]);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1024, height, 0, gl.RGBA, gl.FLOAT, data);
      this.tableBytes += data.byteLength;
    }
    gl.activeTexture(gl.TEXTURE0);
  }
  reserve(count) {
    if (count && (!this.values || this.values.length < this.channels * count)) {
      this.values = new Float32Array(this.channels * count);
      this.packed = this.indexed ? new Int32Array(count * 2) : new Float32Array(count * 12);
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
    gl.bufferData(gl.ARRAY_BUFFER, this.packed, gl.STREAM_DRAW, 0, count * this.stride);
    gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, this.output);
    if ((this.outputBytes || 0) < 4 * this.channels * count) {
      this.outputBytes = 4 * this.channels * count;
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
    gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, this.values, 0, this.channels * count);
    gl.bindBuffer(gl.COPY_READ_BUFFER, null);
    gl.bindVertexArray(null);
    this.inputBytes = this.stride * 4 * count;
    this.stats.polygons += count;
    this.stats.batches++;
  }
  dispose() {
    const gl = this.gl;
    for (const t of this.tables || []) gl.deleteTexture(t);
    gl.deleteVertexArray(this.vao);
    gl.deleteBuffer(this.input);
    gl.deleteBuffer(this.output);
    gl.deleteProgram(this.program);
  }
}
