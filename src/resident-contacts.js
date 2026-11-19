// A moving polygon contributes to one receiver and eight azimuth bins. Accumulate
// both elevation bands and reflected RGB at their destinations on the device.
import { contactKernel, contactTextureWidth, contactVertexLimit, setContactPlanes } from './gpu-contact.js';
import { areaSteps, azimuthCount, lowerFraction } from './contact-transfer.js';
import { dot, sub } from './cpu-geometry.js';

const header = `#version 300 es
precision highp float;
precision highp int;
`;
const attributes = `
layout(location=0) in vec4 polygon; // vertex range, receiver, quadrature weight
layout(location=1) in vec3 source; // radiance row, receiver moment multiplier, static coefficient row
uniform highp sampler2D radiance, cached;
uniform float targets;
flat out vec4 opening;
flat out vec3 light;`;
const deposit = `
  opening = -integral.xzxz * polygon.w * vec4(1., 1., source.y, source.y);
  light = vec3(0);
  if (source.x >= 0.) {
    int side = int(source.x);
    light = (integral.x + integral.z) * texelFetch(radiance, ivec2(0, side), 0).rgb
          + (integral.y + integral.w) * texelFetch(radiance, ivec2(1, side), 0).rgb;
  }
  gl_Position = vec4(2. * (float(bin) + .5) / 8. - 1., 2. * (polygon.z + .5) / targets - 1., 0., 1.);
  gl_PointSize = 1.;`;
function polygonShader(kernel, save) {
  if (!save)
    return (
      header +
      attributes +
      kernel +
      `
void main() {
  int bin = gl_VertexID;
  int index = int(source.z) * 8 + bin;
  int width = textureSize(cached, 0).x;
  vec4 whole = texelFetch(cached, ivec2(index % width, index / width), 0);
  // A moving occluder can only remove light from this stationary aperture.
  vec4 integral = whole.x + whole.z > 0. ? sectorIntegral(bin, ivec2(polygon.xy), int(source.z)) : vec4(0);
  ${deposit}
}`
    );
  return (
    header +
    kernel +
    `
layout(location=0) in vec4 polygon;
uniform ivec2 table;
flat out vec4 coefficients;
void main() {
  int bin = gl_VertexID, index = 8 * int(polygon.z) + bin;
  vec2 texel = vec2(index % table.x, index / table.x);
  gl_Position = vec4((texel + .5) * 2. / vec2(table) - 1., 0., 1.);
  gl_PointSize = 1.;
  coefficients = sectorIntegral(bin, ivec2(polygon.xy), int(polygon.z));
}`
  );
}
const saveCoefficients =
  header + `flat in vec4 coefficients;out vec4 result;void main(){result=coefficients;}`;
const reuse =
  header +
  attributes +
  `
void main() {
  int bin = gl_VertexID, index = 8 * int(polygon.x) + bin;
  int width = textureSize(cached, 0).x;
  vec4 integral = texelFetch(cached, ivec2(index % width, index / width), 0);
  ${deposit}
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
layout(location=3) in float siteNumber;
layout(location=4) in vec4 background; // original RGB and contact strength
uniform highp sampler2D openings, energy, directional;
uniform ivec2 chart;
uniform int cellCount;
flat out vec3 rgb;
void main(){
  int site = int(siteNumber);
  int target = cellCount + site;
  vec3 sum = vec3(0.);
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
const emptyOpening = new Float32Array([
  lowerFraction / azimuthCount,
  (1 - lowerFraction) / azimuthCount,
  0,
  0,
]);
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

// A row may contain several convex pieces of the same stationary aperture.
// The static pass adds all of their area and first-moment coefficients.
function polygonBatch() {
  return { positions: [], jobs: [], maxVertices: 0 };
}
function packPolygon(batch, polygon, target, weight, source, moment, staticRow) {
  const count = polygon.uv.length;
  batch.maxVertices = Math.max(batch.maxVertices, count);
  batch.jobs.push(batch.positions.length / 2, count, target, weight, source, moment, staticRow);
  for (const [u, v] of polygon.uv) batch.positions.push(u, v);
}
function sealBatch(batch) {
  const stride = 4 * contactTextureWidth;
  const positions = new Float32Array(stride * Math.max(1, Math.ceil(batch.positions.length / stride)));
  positions.set(batch.positions);
  return { positions, jobs: Float32Array.from(batch.jobs), maxVertices: batch.maxVertices };
}
export class ResidentContacts {
  constructor(renderer) {
    const gl = (this.gl = renderer.gl);
    if (!gl.getExtension('EXT_color_buffer_float') || !gl.getExtension('EXT_float_blend'))
      throw Error('Resident contacts require renderable and blendable RGBA32F.');
    this.renderer = renderer;
    this.limit = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.stats = { polygons: 0, staticPolygons: 0, uploadBytes: 0, peakBytes: 0 };
    this.payloadBytes = 0;
    this.tables = new Map();
    this.staticRecords = new WeakMap();
    this.kernels = new Map();
    this.buffers = [];
    this.vaos = [];
    this.textures = [];
    this.programs = [];
    this.frames = [];
    try {
      this.reuse = this.makeProgram(reuse, accumulate, { cached: 6, radiance: 5 });
      this.final = this.makeProgram(resolve, shade, { openings: 2, energy: 3, directional: 4 });
      this.poly = this.attributes([4, 3], 1);
      this.site = this.attributes([4, 4, 4, 1, 4], 0);
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
    if (w > this.limit || h > this.limit) throw Error('Contact field exceeds GPU texture capacity.');
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
  kernel(vertices, save) {
    const capacity = Math.max(contactVertexLimit, 2 ** Math.ceil(Math.log2(vertices + 3)));
    const key = capacity + '/' + save;
    let p = this.kernels.get(key);
    if (!p) {
      const kernel = contactKernel(capacity, true);
      p = this.makeProgram(
        polygonShader(kernel, save),
        save ? saveCoefficients : accumulate,
        save ? { positions: 1, frames: 7 } : { positions: 1, radiance: 5, cached: 6, frames: 7 },
      );
      setContactPlanes(this.gl, p);
      this.kernels.set(key, p);
    }
    return p;
  }
  beginChart(feature, width, rows, stationary) {
    const targets = feature.size + width * rows;
    if (targets > this.limit) throw Error('Contact chart exceeds GPU texture height.');
    let cache = this.staticRecords.get(stationary);
    if (!cache) {
      cache = { count: 0, build: polygonBatch() };
      this.staticRecords.set(stationary, cache);
    }
    const sources = new Map(),
      changed = polygonBatch(),
      cachedJobs = [];
    function sourceRow(side, site) {
      if (site.cell !== undefined) return -1;
      if (!sources.has(side)) sources.set(side, sources.size);
      return sources.get(side);
    }
    return {
      register(sample) {
        sample.coefficientRow = cache.count++;
        for (const poly of sample.lit) packPolygon(cache.build, poly, sample.coefficientRow, 0, 0, 0, 0);
      },
      reuse(sample, site) {
        const cell = site.cell !== undefined;
        cachedJobs.push(
          sample.coefficientRow,
          0,
          cell ? site.cell : feature.size + site.texel,
          cell ? 1 / (areaSteps * areaSteps) : 1,
          sourceRow(sample.side, site),
          cell ? 3 * site.pos : 0,
          sample.coefficientRow,
        );
      },
      enqueue(sample, site, poly, sign = 1) {
        const cell = site.cell !== undefined;
        packPolygon(
          changed,
          poly,
          cell ? site.cell : feature.size + site.texel,
          cell ? sign / (areaSteps * areaSteps) : 1,
          sourceRow(sample.side, site),
          cell ? 3 * site.pos : 0,
          sample.coefficientRow,
        );
      },
      finish() {
        if (cache.build) {
          cache.batch = sealBatch(cache.build);
          delete cache.build;
        }
        return {
          cache,
          cachedJobs: Float32Array.from(cachedJobs),
          changed: sealBatch(changed),
          targets,
          sources: Array.from(sources.keys()),
        };
      },
    };
  }
  drawPolygons(batch, save, width, height) {
    if (!batch.jobs.length) return;
    const gl = this.gl;
    if (batch.positions.length > contactTextureWidth * 4 * this.limit)
      throw Error('Contact polygon batch exceeds GPU vertex texture capacity.');
    this.texture(1, contactTextureWidth, batch.positions.length / (4 * contactTextureWidth), batch.positions);
    const p = this.kernel(batch.maxVertices, save);
    gl.useProgram(p);
    if (save) gl.uniform2i(gl.getUniformLocation(p, 'table'), width, height);
    else gl.uniform1f(gl.getUniformLocation(p, 'targets'), height);
    this.upload(this.poly, batch.jobs);
    gl.drawArraysInstanced(gl.POINTS, 0, 8, batch.jobs.length / 7);
    if (save) this.stats.staticPolygons += batch.jobs.length / 7;
    else this.stats.polygons += batch.jobs.length / 7;
  }
  staticTable(grid, contact) {
    const cache = contact.resident.cache;
    const gl = this.gl;
    let t = this.tables.get(grid);
    if (t?.cache === cache) return t;
    if (t) this.releaseTable(t);
    this.tables.delete(grid);
    const width = Math.min(contactTextureWidth, this.limit);
    const height = Math.max(1, Math.ceil((cache.count * 8) / width));
    if (height > this.limit) throw Error('Contact coefficient table exceeds GPU texture capacity.');
    t = { id: gl.createTexture(), frames: gl.createTexture(), cache };
    try {
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, t.id);
      for (const param of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
        gl.texParameteri(gl.TEXTURE_2D, param, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      const frameHeight = Math.max(1, Math.ceil((cache.count * 3) / width));
      const frames = new Float32Array(4 * width * frameHeight);
      const { u, v, n } = contact.feature;
      for (const site of contact.stationary)
        for (const sample of site.samples) {
          const m = sample.m,
            delta = sub(m.center, site.p);
          const offset = sample.coefficientRow * 12;
          frames.set(
            [
              dot(delta, u),
              dot(delta, v),
              dot(delta, n),
              0,
              dot(m.axisU, u),
              dot(m.axisU, v),
              dot(m.axisU, n),
              dot(m.axisU, m.momentAxis),
              dot(m.axisV, u),
              dot(m.axisV, v),
              dot(m.axisV, n),
              dot(m.axisV, m.momentAxis),
            ],
            offset,
          );
        }
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, t.frames);
      for (const param of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
        gl.texParameteri(gl.TEXTURE_2D, param, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, frameHeight, 0, gl.RGBA, gl.FLOAT, frames);
      this.stats.uploadBytes += frames.byteLength;
      this.sizeAllocation(t, width * height * 16 + frames.byteLength);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.scratch);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.id, 0);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, null, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
      gl.viewport(0, 0, width, height);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      // A rebuilt table can recover the input from retained scene geometry.
      // Release the packed upload once the device owns its coefficients.
      if (!cache.batch) {
        const batch = polygonBatch();
        for (const site of contact.stationary)
          for (const sample of site.samples)
            for (const poly of sample.lit) packPolygon(batch, poly, sample.coefficientRow, 0, 0, 0, 0);
        cache.batch = sealBatch(batch);
      }
      this.drawPolygons(cache.batch, true, width, height);
      delete cache.batch;
      this.tables.set(grid, t);
    } catch (error) {
      this.releaseTable(t);
      throw error;
    }
    gl.disable(gl.BLEND);
    return t;
  }
  releaseTable(t) {
    this.gl.deleteTexture(t.id);
    this.gl.deleteTexture(t.frames);
    this.sizeAllocation(t, 0);
  }
  retainCharts(charts) {
    for (const [grid, t] of this.tables)
      if (!charts.has(grid)) {
        this.releaseTable(t);
        this.tables.delete(grid);
      }
  }
  paint(view) {
    const gl = this.gl,
      { width, rows, tex } = view.atlas;
    try {
      for (const [grid, { contact, filter }] of view.contactDraws) {
        const { feature, sites, resident } = contact,
          { changed, cachedJobs, targets } = resident;
        const table = this.staticTable(grid, contact);
        gl.activeTexture(gl.TEXTURE7);
        gl.bindTexture(gl.TEXTURE_2D, table.frames);
        const sol = view.solution,
          { field, slope } = sol.directional.get(feature);
        const dir = new Float32Array(feature.size * 32 * 4);
        for (let i = 0; i < field.length / 3; i++)
          for (let c = 0; c < 3; c++) {
            dir[8 * i + c] = field[3 * i + c];
            dir[8 * i + 4 + c] = slope[3 * i + c];
          }
        this.texture(4, 32, feature.size, dir);
        const openTex = this.texture(2, 8, targets, null),
          lightTex = this.texture(3, 8, targets, null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.scratch);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, openTex, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, lightTex, 0);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
        gl.clearBufferfv(gl.COLOR, 0, emptyOpening);
        gl.clearBufferfv(gl.COLOR, 1, new Float32Array(4));
        if (changed.jobs.length || cachedJobs.length) {
          const lighting = new Float32Array(8 * Math.max(1, resident.sources.length));
          resident.sources.forEach((side, i) => {
            for (let c = 0; c < 3; c++) {
              lighting[8 * i + c] = sol.outgoing[3 * side + c];
              lighting[8 * i + 4 + c] = sol.outSlope[3 * side + c];
            }
          });
          this.texture(5, 2, lighting.length / 8, lighting);
          gl.viewport(0, 0, 8, targets);
          gl.enable(gl.BLEND);
          gl.blendFunc(gl.ONE, gl.ONE);
          gl.activeTexture(gl.TEXTURE6);
          gl.bindTexture(gl.TEXTURE_2D, table.id);
          if (cachedJobs.length) {
            gl.useProgram(this.reuse);
            gl.uniform1f(gl.getUniformLocation(this.reuse, 'targets'), targets);
            this.upload(this.poly, cachedJobs);
            gl.drawArraysInstanced(gl.POINTS, 0, 8, cachedJobs.length / 7);
          }
          this.drawPolygons(changed, false, 8, targets);
          gl.disable(gl.BLEND);
        }
        const data = new Float32Array(sites.length * 17);
        for (let i = 0; i < sites.length; i++) {
          const site = sites[i],
            a = 17 * i,
            o = 4 * site,
            b = 4 * grid.displayRow * width + o;
          for (let k = 0; k < 4; k++) {
            data[a + k] = filter.indices[o + k];
            data[a + 4 + k] = filter.weights[o + k];
            data[a + 8 + k] = filter.linearWeights[o + k];
          }
          data.set([site, tex[b], tex[b + 1], tex[b + 2], contact.strength[i]], a + 12);
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
    for (const t of this.tables.values()) this.releaseTable(t);
    this.tables.clear();
    this.programs.forEach((p) => gl.deleteProgram(p));
    this.vaos.forEach((v) => gl.deleteVertexArray(v));
    this.buffers.forEach((b) => gl.deleteBuffer(b));
    this.textures.forEach((t) => gl.deleteTexture(t.id));
    this.frames.forEach((f) => gl.deleteFramebuffer(f));
  }
}
