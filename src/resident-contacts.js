// A moving polygon contributes to one receiver and eight azimuth bins. Accumulate
// both elevation bands and reflected RGB at their destinations on the device.
import { contactKernel, contactTextureWidth, contactVertexLimit, setContactPlanes } from './gpu-contact.js';
import { areaSteps, azimuthCount } from './contact-transfer.js';
import { dot, point, sub } from './cpu-geometry.js';

const header = `#version 300 es
precision highp float;
precision highp int;
`;
const integrate =
  header +
  `
layout(location=0) in vec4 polygon; // start, count, receiver, quadrature weight
layout(location=1) in vec2 source; // packed source index and receiver moment multiplier
uniform highp sampler2D radiance;
uniform float targets;
flat out vec4 opening;
flat out vec3 light;
${contactKernel}
void main() {
  int bin = gl_VertexID;
  // Two (area, source moment) pairs, one for each elevation band.
  vec4 integral = sectorIntegral(bin, ivec2(polygon.xy));
  opening = -integral.xzxz * polygon.w * vec4(1., 1., source.y, source.y);
  light = vec3(0);
  if (source.x >= 0.) {
    int side = int(source.x);
    light = (integral.x + integral.z) * texelFetch(radiance, ivec2(0, side), 0).rgb
          + (integral.y + integral.w) * texelFetch(radiance, ivec2(1, side), 0).rgb;
  }
  gl_Position = vec4(2. * (float(bin) + .5) / 8. - 1., 2. * (polygon.z + .5) / targets - 1., 0., 1.);
  gl_PointSize = 1.;
}`;
const accumulate =
  header +
  `
flat in vec4 opening;flat in vec3 light;
layout(location=0) out vec4 delta;layout(location=1) out vec4 energy;
void main(){delta=opening;energy=vec4(light,0.);}`;
const resolve =
  header +
  `
layout(location=0) in vec4 cells;
layout(location=1) in vec4 weights;
layout(location=2) in vec4 moments;
layout(location=3) in vec4 local; // cached source RGB and site number
layout(location=4) in vec4 background; // original RGB and contact strength
uniform highp sampler2D openings, energy, directional;
uniform ivec2 chart;
uniform int cellCount;
flat out vec3 rgb;
void main(){
  int site = int(local.w);
  int target = cellCount + site;
  vec3 sum = local.rgb;
  for (int bin = 0; bin < 8; bin++) {
    vec2 open = max(vec2(0), texelFetch(openings, ivec2(bin, target), 0).xy);
    vec2 denominator = vec2(0);
    vec3 low = vec3(0), high = vec3(0);
    for (int j = 0; j < 4; j++) {
      int cell = int(cells[j]);
      vec4 f = texelFetch(openings, ivec2(bin, cell), 0);
      f.xy = max(f.xy, vec2(0));
      f.zw = clamp(f.zw, -f.xy, f.xy);
      denominator += weights[j] * f.xy + moments[j] * f.zw;
      low += weights[j] * texelFetch(directional, ivec2(2 * bin, cell), 0).rgb
           + moments[j] * texelFetch(directional, ivec2(2 * bin + 1, cell), 0).rgb;
      high += weights[j] * texelFetch(directional, ivec2(2 * (bin + 8), cell), 0).rgb
            + moments[j] * texelFetch(directional, ivec2(2 * (bin + 8) + 1, cell), 0).rgb;
    }
    vec2 gain = open / max(denominator, vec2(1. / ${2 * azimuthCount * areaSteps * areaSteps}.));
    sum += texelFetch(energy, ivec2(bin, target), 0).rgb + gain.x * low + gain.y * high;
  }
  rgb = mix(background.rgb, sum, background.a);
  vec2 texel = vec2(site % chart.x, site / chart.x);
  gl_Position = vec4((texel + .5) * 2. / vec2(chart) - 1., 0., 1.);
  gl_PointSize = 1.;
}`;
const shade = header + `flat in vec3 rgb;out vec4 color;void main(){color=vec4(rgb,0.);}`;

function program(gl, vs, fs) {
  const p = gl.createProgram();
  try {
    for (const [type, text] of [
      [gl.VERTEX_SHADER, vs],
      [gl.FRAGMENT_SHADER, fs],
    ]) {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, text);
      gl.compileShader(shader);
      gl.attachShader(p, shader);
      gl.deleteShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(shader));
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(p));
    return p;
  } catch (e) {
    gl.deleteProgram(p);
    throw e;
  }
}

export class ResidentContacts {
  constructor(renderer) {
    const gl = (this.gl = renderer.gl);
    if (!gl.getExtension('EXT_color_buffer_float') || !gl.getExtension('EXT_float_blend'))
      throw Error('Resident contacts require renderable and blendable RGBA32F.');
    this.renderer = renderer;
    this.limit = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.stats = { polygons: 0, fallback: 0, fallbackCharts: 0, uploadBytes: 0, peakBytes: 0 };
    this.payloadBytes = 0;
    this.buffers = [];
    this.vaos = [];
    this.textures = [];
    this.programs = [];
    this.frames = [];
    try {
      this.acc = this.makeProgram(integrate, accumulate, { positions: 1, radiance: 5 });
      setContactPlanes(gl, this.acc);
      this.final = this.makeProgram(resolve, shade, { openings: 2, energy: 3, directional: 4 });
      this.poly = this.attributes([4, 2], 1);
      this.site = this.attributes([4, 4, 4, 4, 4], 0);
      this.scratch = gl.createFramebuffer();
      this.frames.push(this.scratch);
      this.atlasFbo = gl.createFramebuffer();
      this.frames.push(this.atlasFbo);
    } catch (e) {
      this.dispose();
      throw e;
    } finally {
      this.restore();
    }
  }
  makeProgram(vs, fs, samplers) {
    const gl = this.gl,
      p = program(gl, vs, fs);
    this.programs.push(p);
    gl.useProgram(p);
    for (const [name, unit] of Object.entries(samplers)) gl.uniform1i(gl.getUniformLocation(p, name), unit);
    return p;
  }
  attributes(sizes, divisor) {
    const gl = this.gl,
      vao = gl.createVertexArray(),
      buffer = gl.createBuffer();
    this.vaos.push(vao);
    this.buffers.push(buffer);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    const stride = 4 * sizes.reduce((a, b) => a + b);
    let offset = 0;
    sizes.forEach((size, i) => {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, size, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(i, divisor);
      offset += 4 * size;
    });
    return { vao, buffer };
  }
  texture(unit, w, h, data) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    let t = this.textures[unit];
    if (!t) {
      t = this.textures[unit] = { id: gl.createTexture() };
      gl.bindTexture(gl.TEXTURE_2D, t.id);
      for (const key of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
        gl.texParameteri(gl.TEXTURE_2D, key, gl.NEAREST);
    } else gl.bindTexture(gl.TEXTURE_2D, t.id);
    if (data) this.stats.uploadBytes += data.byteLength;
    if (t.w !== w || t.h !== h) {
      this.sizeAllocation(t, 16 * w * h);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data);
      Object.assign(t, { w, h });
    } else if (data) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, w, h, gl.RGBA, gl.FLOAT, data);
    return t.id;
  }
  beginChart(feature, width, rows) {
    const targets = feature.size + width * rows;
    if (targets > this.limit) {
      this.stats.fallbackCharts++;
      return null;
    }
    const sources = new Map(),
      positions = [],
      jobs = [],
      owner = this;
    return {
      enqueue(mode, side, site, poly) {
        const start = positions.length / 4,
          n = poly.uv.length;
        if (n + 3 > contactVertexLimit || start + n > contactTextureWidth * owner.limit) {
          owner.stats.fallback++;
          return false;
        }
        const cell = site.cell !== undefined;
        if (!cell && !sources.has(side)) {
          if (sources.size >= owner.limit) {
            owner.stats.fallback++;
            return false;
          }
          sources.set(side, sources.size);
        }
        jobs.push(
          start,
          n,
          cell ? site.cell : feature.size + site.texel,
          cell ? 1 / (areaSteps * areaSteps) : 1,
          cell ? -1 : sources.get(side),
          cell ? 3 * site.pos : 0,
        );
        for (const [u, v] of poly.uv) {
          const q = point(mode, u, v),
            d = sub(q, site.p);
          positions.push(
            dot(d, feature.u),
            dot(d, feature.v),
            dot(d, feature.n),
            dot(sub(q, mode.center), mode.momentAxis),
          );
        }
        return true;
      },
      finish(cellOpen, slopeOpen, sites, openings) {
        const initial = new Float32Array(targets * 8 * 4);
        for (let c = 0; c < feature.size; c++)
          for (let k = 0; k < 8; k++) {
            const src = c * 16 + k;
            initial.set(
              [cellOpen[src], cellOpen[src + 8], slopeOpen[src], slopeOpen[src + 8]],
              4 * (8 * c + k),
            );
          }
        sites.forEach((s, i) => {
          for (let k = 0; k < 8; k++) {
            const dest = 4 * (8 * (feature.size + s) + k);
            initial[dest] = openings[16 * i + k];
            initial[dest + 1] = openings[16 * i + k + 8];
          }
        });
        const rowLength = 4 * contactTextureWidth;
        const packed = new Float32Array(rowLength * Math.max(1, Math.ceil(positions.length / rowLength)));
        packed.set(positions);
        return {
          initial,
          packed,
          jobs: Float32Array.from(jobs),
          targets,
          sources: Array.from(sources.keys()),
        };
      },
    };
  }
  paint(view) {
    const gl = this.gl,
      { width, rows, tex } = view.atlas;
    try {
      for (const [grid, { contact, filter }] of view.contactDraws) {
        const { feature, sites, resident } = contact,
          { jobs, targets, initial, packed } = resident;
        const sol = view.solution,
          { field, slope } = sol.directional.get(feature);
        const dir = new Float32Array(feature.size * 32 * 4);
        for (let i = 0; i < field.length / 3; i++)
          for (let c = 0; c < 3; c++) {
            dir[8 * i + c] = field[3 * i + c];
            dir[8 * i + 4 + c] = slope[3 * i + c];
          }
        this.texture(4, 32, feature.size, dir);
        this.texture(1, contactTextureWidth, packed.length / (4 * contactTextureWidth), packed);
        const openTex = this.texture(2, 8, targets, initial),
          lightTex = this.texture(3, 8, targets, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.scratch);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, openTex, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, lightTex, 0);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
        gl.clearBufferfv(gl.COLOR, 1, new Float32Array(4));
        if (jobs.length) {
          const lighting = new Float32Array(8 * Math.max(1, resident.sources.length));
          resident.sources.forEach((side, i) => {
            for (let c = 0; c < 3; c++) {
              lighting[8 * i + c] = sol.outgoing[3 * side + c];
              lighting[8 * i + 4 + c] = sol.outSlope[3 * side + c];
            }
          });
          this.texture(5, 2, lighting.length / 8, lighting);
          gl.useProgram(this.acc);
          gl.uniform1f(gl.getUniformLocation(this.acc, 'targets'), targets);
          this.upload(this.poly, jobs);
          gl.viewport(0, 0, 8, targets);
          gl.enable(gl.BLEND);
          gl.blendFunc(gl.ONE, gl.ONE);
          gl.drawArraysInstanced(gl.POINTS, 0, 8, jobs.length / 6);
          gl.disable(gl.BLEND);
          this.stats.polygons += jobs.length / 6;
        }
        const data = new Float32Array(sites.length * 20);
        let link = 0;
        for (let i = 0; i < sites.length; i++) {
          const site = sites[i],
            a = 20 * i,
            o = 4 * site,
            b = 4 * grid.displayRow * width + o;
          for (let k = 0; k < 4; k++) {
            data[a + k] = filter.indices[o + k];
            data[a + 4 + k] = filter.weights[o + k];
            data[a + 8 + k] = filter.linearWeights[o + k];
          }
          const sum = [0, 0, 0];
          for (; link < contact.ends[i]; link++) {
            const j = 3 * contact.sides[link];
            for (let c = 0; c < 3; c++)
              sum[c] +=
                contact.localMean[link] * sol.outgoing[j + c] +
                contact.localSlope[link] * sol.outSlope[j + c];
          }
          data.set([...sum, site, tex[b], tex[b + 1], tex[b + 2], contact.strength[i]], a + 12);
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.atlasFbo);
        gl.framebufferTexture2D(
          gl.FRAMEBUFFER,
          gl.COLOR_ATTACHMENT0,
          gl.TEXTURE_2D,
          this.renderer.texture,
          0,
        );
        gl.useProgram(this.final);
        gl.uniform2i(gl.getUniformLocation(this.final, 'chart'), width, rows);
        gl.uniform1i(gl.getUniformLocation(this.final, 'cellCount'), feature.size);
        this.upload(this.site, data);
        gl.viewport(0, grid.displayRow, width, rows);
        gl.colorMask(true, true, true, false);
        gl.drawArrays(gl.POINTS, 0, sites.length);
        gl.colorMask(true, true, true, true);
      }
    } finally {
      this.restore();
    }
  }
  sizeAllocation(resource, bytes) {
    this.payloadBytes += bytes - (resource.bytes || 0);
    resource.bytes = bytes;
    this.stats.peakBytes = Math.max(this.stats.peakBytes, this.payloadBytes);
  }
  upload(input, data) {
    const { vao, buffer } = input;
    this.sizeAllocation(input, data.byteLength);
    this.stats.uploadBytes += data.byteLength;
    const gl = this.gl;
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);
  }
  restore() {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ZERO);
    gl.colorMask(true, true, true, true);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.bindVertexArray(null);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.renderer.buffer);
    gl.useProgram(this.renderer.program);
  }
  dispose() {
    const gl = this.gl;
    this.programs.forEach((p) => gl.deleteProgram(p));
    this.vaos.forEach((v) => gl.deleteVertexArray(v));
    this.buffers.forEach((b) => gl.deleteBuffer(b));
    this.textures.forEach((t) => gl.deleteTexture(t.id));
    this.frames.forEach((f) => gl.deleteFramebuffer(f));
  }
}
