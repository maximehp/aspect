import { buildReconstruction } from "./reconstruction.js";
import { DEFAULT_LAMP_RGB } from "./lamp.js";

// The visibility compiler supplies convex, uniformly lit or shadowed cells.
// Their display triangles lie on the authored planes at every resolution.
const vertexShader = `#version 300 es
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
layout(location=3) in float lampVisible;
layout(location=4) in vec2 faceUV;
layout(location=5) in vec4 gridLayout;
uniform mat4 viewProjection;
out vec3 worldPosition;
flat out vec3 surfaceNormal;
flat out vec3 reflectance;
flat out float directVisibility;
out vec2 surfaceUV;
flat out ivec4 sampleGrid;
void main() {
  worldPosition = position;
  surfaceNormal = normal;
  reflectance = color;
  directVisibility = lampVisible;
  surfaceUV = faceUV;
  sampleGrid = ivec4(gridLayout);
  gl_Position = viewProjection * vec4(position, 1.0);
}`;

const fragmentShader = `#version 300 es
precision highp float;
precision highp int;
in vec3 worldPosition;
flat in vec3 surfaceNormal;
flat in vec3 reflectance;
flat in float directVisibility;
in vec2 surfaceUV;
flat in ivec4 sampleGrid; // start texel, side stride, nu, nv
uniform vec3 lampPosition;
uniform vec3 scaledLamp;
uniform float lampIntensity;
uniform int lightMode; // 0 total, 1 direct, 2 indirect
uniform highp sampler2D bounceTexture;
out vec4 outputColor;

vec4 sampleAt(int address, int width) {
  return texelFetch(bounceTexture, ivec2(address % width, address / width), 0);
}

vec3 edgeValue(int start, int edge, int count, float t, int width) {
  ivec2 header = ivec2(sampleAt(start + count + edge, width).xy);
  int offset = start + header.x, lo = 0, hi = header.y - 1;
  while (hi - lo > 1) {
    int mid = (hi + lo) / 2;
    if (sampleAt(offset + mid, width).w < t) lo = mid;
    else hi = mid;
  }
  vec4 a = sampleAt(offset + lo, width), b = sampleAt(offset + hi, width);
  return mix(a.rgb, b.rgb, clamp((t - a.w) / max(b.w - a.w, 1.e-30), 0., 1.));
}

vec3 indirectRadiance() {
  ivec2 size = sampleGrid.zw;
  if (size.x == 0) return vec3(0);
  int width = textureSize(bounceTexture, 0).x;
  int start = sampleGrid.x + (gl_FrontFacing ? 0 : sampleGrid.y);
  vec2 uv = clamp(0.5 * surfaceUV + 0.5, 0., 1.);
  vec2 raw = uv * vec2(size) - 0.5;
  vec2 center = clamp(raw, vec2(0), vec2(size - 1));
  ivec2 lo = ivec2(floor(center)), hi = min(lo + 1, size - 1);
  vec2 f = fract(center);
  int row0 = start + lo.y * size.x, row1 = start + hi.y * size.x;
  vec3 interior = mix(mix(sampleAt(row0 + lo.x, width).rgb, sampleAt(row0 + hi.x, width).rgb, f.x),
                      mix(sampleAt(row1 + lo.x, width).rgb, sampleAt(row1 + hi.x, width).rgb, f.x), f.y);
  vec2 w = clamp(2. * abs(raw - center), 0., 1.);
  if (w.x == 0. && w.y == 0.) return interior;
  // Use the common edge curves exactly at a seam. The nonnegative blend
  // extends each curve to the first row of independent interior samples.
  int count = size.x * size.y;
  vec3 xEdge = vec3(0), yEdge = vec3(0);
  if (w.x > 0.) xEdge = edgeValue(start, raw.x < 0. ? 0 : 1, count, uv.y, width);
  if (w.y > 0.) yEdge = edgeValue(start, raw.y < 0. ? 2 : 3, count, uv.x, width);
  if (w.x == 1. && w.y == 1.) return xEdge;
  return (interior * (1.-w.x) * (1.-w.y) + xEdge * w.x * (1.-w.y) +
          yEdge * w.y * (1.-w.x)) / (1.-w.x*w.y);
}
void main() {
  vec3 radiance = lightMode == 1 ? vec3(0) : lampIntensity * indirectRadiance();
  if (lightMode != 2 && directVisibility > 0.5) {
    vec3 toLamp = lampPosition - worldPosition;
    float distanceSquared = dot(toLamp, toLamp);
    if (distanceSquared > 0.0) {
      // An opaque sheet can reflect light from either face, but only toward
      // observers on the same side as the incident light.
      vec3 facingNormal = gl_FrontFacing ? surfaceNormal : -surfaceNormal;
      float cosine = max(dot(facingNormal, toLamp), 0.0) * inversesqrt(distanceSquared);
      radiance += reflectance * scaledLamp * (cosine / distanceSquared);
    }
  }
  vec3 mapped = radiance / (vec3(1.0) + radiance);
  vec3 srgb = mix(12.92 * mapped, 1.055 * pow(mapped, vec3(1.0 / 2.4)) - 0.055,
                  greaterThan(mapped, vec3(0.0031308)));
  outputColor = vec4(srgb, 1.0);
}`;

function buildProgram(gl) {
  const program = gl.createProgram();
  const shaders = [];
  try {
    for (const [type, source] of [[gl.VERTEX_SHADER, vertexShader], [gl.FRAGMENT_SHADER, fragmentShader]]) {
      const shader = gl.createShader(type);
      shaders.push(shader);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      gl.attachShader(program, shader);
    }
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    return program;
  } catch (error) {
    gl.deleteProgram(program);
    throw error;
  } finally {
    for (const shader of shaders) gl.deleteShader(shader);
  }
}

function cameraMatrix({ position, yaw, pitch, fov }, aspect) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const verticalScale = 1 / Math.tan(fov / 2);
  const horizontalScale = verticalScale / aspect;
  const near = 0.035, far = 500;
  const zScale = (far + near) / (far - near);
  const zOffset = -2 * far * near / (far - near);
  const [x, y, z] = position;
  const right = cy * x + sy * z;
  const up = -sy * sp * x + cp * y + cy * sp * z;
  const forward = sy * cp * x + sp * y - cy * cp * z;
  return new Float32Array([
    cy * horizontalScale, -sy * sp * verticalScale, sy * cp * zScale, sy * cp,
    0, cp * verticalScale, sp * zScale, sp,
    sy * horizontalScale, cy * sp * verticalScale, -cy * cp * zScale, -cy * cp,
    -right * horizontalScale, -up * verticalScale, -forward * zScale + zOffset, -forward,
  ]);
}

const floatsPerVertex = 16;

export class RoomRenderer {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { alpha: false, depth: true, antialias: true });
    if (!gl) throw new Error("This room needs WebGL 2. Check that hardware acceleration is enabled in your browser.");
    this.gl = gl;
    this.program = buildProgram(gl);
    this.viewProjection = gl.getUniformLocation(this.program, "viewProjection");
    this.lampPosition = gl.getUniformLocation(this.program, "lampPosition");
    this.lampIntensity = gl.getUniformLocation(this.program, "lampIntensity");
    this.scaledLamp = gl.getUniformLocation(this.program, "scaledLamp");
    this.lightMode = gl.getUniformLocation(this.program, "lightMode");
    this.bounceTexture = gl.createTexture();
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.bounceTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.buffer = gl.createBuffer();
    this.layout = gl.createVertexArray();
    gl.bindVertexArray(this.layout);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    for (const [index, size, offset] of [[0, 3, 0], [1, 3, 3], [2, 3, 6], [3, 1, 9], [4, 2, 10], [5, 4, 12]]) {
      gl.enableVertexAttribArray(index);
      gl.vertexAttribPointer(index, size, gl.FLOAT, false, floatsPerVertex * 4, offset * 4);
    }
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 1);
    this.vertexCount = 0;
  }

  setGeometry(surfaces, field) {
    const atlas = buildReconstruction(surfaces, field, this.maxTextureSize);
    this.uploadGeometry(surfaces, atlas);
  }

  uploadGeometry(surfaces, atlas) {
    let count = 0;
    for (const { cells } of surfaces) {
      for (const { uv } of cells) count += 3 * Math.max(0, uv.length - 2);
    }
    const vertices = new Float32Array(count * floatsPerVertex);
    let offset = 0;
    for (const [surfaceIndex, { rect, cells }] of surfaces.entries()) {
      const { center, axisU: a, axisV: b, color } = rect;
      const nx = a[1] * b[2] - a[2] * b[1];
      const ny = a[2] * b[0] - a[0] * b[2];
      const nz = a[0] * b[1] - a[1] * b[0];
      const norm = Math.hypot(nx, ny, nz);
      if (norm === 0) throw new Error("Degenerate surface: " + rect.id);
      const normal = [nx / norm, ny / norm, nz / norm];
      for (const { uv, lit } of cells) {
        for (let i = 1; i + 1 < uv.length; i++) {
          for (const index of [0, i, i + 1]) {
            const [u, v] = uv[index];
            for (let k = 0; k < 3; k++) vertices[offset++] = center[k] + u * a[k] + v * b[k];
            vertices.set(normal, offset);
            vertices.set(color, offset + 3);
            offset += 6;
            vertices[offset++] = Number(lit);
            vertices[offset++] = u;
            vertices[offset++] = v;
            vertices.set(atlas.layouts[surfaceIndex], offset);
            offset += 4;
          }
        }
      }
    }
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.bounceTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, atlas.width, atlas.height, 0, gl.RGBA, gl.FLOAT, atlas.pixels);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.DYNAMIC_DRAW);
    this.vertexCount = count;
  }

  draw(camera, lightPosition, intensity, outputSize = null, mode = "total", fullPowerRGB = DEFAULT_LAMP_RGB) {
    const gl = this.gl, canvas = gl.canvas;
    const ratio = globalThis.devicePixelRatio || 1;
    const width = outputSize?.[0] ?? Math.max(1, Math.round(canvas.clientWidth * ratio));
    const height = outputSize?.[1] ?? Math.max(1, Math.round(canvas.clientHeight * ratio));
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.layout);
    gl.uniformMatrix4fv(this.viewProjection, false, cameraMatrix(camera, gl.drawingBufferWidth / gl.drawingBufferHeight));
    gl.uniform3fv(this.lampPosition, lightPosition);
    gl.uniform1f(this.lampIntensity, intensity);
    gl.uniform3f(this.scaledLamp, fullPowerRGB[0] * intensity, fullPowerRGB[1] * intensity, fullPowerRGB[2] * intensity);
    gl.uniform1i(this.lightMode, mode === "indirect" ? 2 : mode === "direct" ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, this.vertexCount);
  }

  // Read immediately after draw to include GPU completion in an explicit timing sample.
  synchronize() {
    const gl = this.gl;
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
  }

  dispose() {
    this.gl.deleteTexture(this.bounceTexture);
    this.gl.deleteVertexArray(this.layout);
    this.gl.deleteBuffer(this.buffer);
    this.gl.deleteProgram(this.program);
  }
}
