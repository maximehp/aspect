// Integrate visible apertures as soon as geometry is ready. The GPU retains
// their angular openings and source moments until the current RGB is available.
import { contactKernel, contactTextureWidth, contactVertexLimit, setContactPlanes } from './gpu-contact.js';
import { areaSteps, azimuthCount, lowerFraction } from './contact-transfer.js';
import { dot, sub } from './cpu-geometry.js';

const header = `#version 300 es
precision highp float;
precision highp int;
`;
// Each source/site pair has eight consecutive RGBA coefficients. The two
// elevation bands each store projected area and its source-gradient moment.
const tablePosition = `
  int index = row * 8 + bin;
  ivec2 address = ivec2(index % table.x, index / table.x);
  gl_Position = vec4((vec2(address) + .5) * 2. / vec2(table) - 1., 0., 1.);
  gl_PointSize = 1.;`;
function polygonShader(kernel, stationary) {
  return (
    header +
    kernel +
    `
layout(location=0) in vec4 polygon; // first vertex, count, coefficient row, sign
uniform ivec2 table;
uniform highp sampler2D cached;
flat out vec4 coefficients;
void main() {
  int row = int(polygon.z), bin = gl_VertexID;
  ${tablePosition}
  ${
    stationary
      ? ''
      : `vec4 whole = texelFetch(cached, address, 0);
  if (whole.x + whole.z <= 0.) { coefficients = vec4(0); gl_Position.x = 2.; return; }`
  }
  coefficients = polygon.w * sectorIntegral(bin, ivec2(polygon.xy), row);
}`
  );
}
const saveCoefficients =
  header + `flat in vec4 coefficients;out vec4 result;void main(){result=coefficients;}`;
const reuse =
  header +
  `
layout(location=0) in float sourceRow;
uniform ivec2 table;
uniform highp sampler2D cached;
flat out vec4 coefficients;
void main() {
  int row = int(sourceRow), bin = gl_VertexID;
  ${tablePosition}
  coefficients = texelFetch(cached, address, 0);
  if (all(equal(coefficients, vec4(0)))) gl_Position.x = 2.;
}`;
// The background needs a directional opening. Diffuse source RGB is common
// to every incident direction, so only its total area and first moment survive.
const responseAddress = `
uniform highp sampler2D visible, frames;
uniform ivec2 responseSize;
vec4 atIndex(highp sampler2D t,int i) {
  int w = textureSize(t,0).x;
  return texelFetch(t, ivec2(i % w, i / w), 0);
}
void place(int i) {
  ivec2 p = ivec2(i % responseSize.x, i / responseSize.x);
  gl_Position = vec4(2. * (vec2(p) + .5) / vec2(responseSize) - 1., 0., 1.);
  gl_PointSize = 1.;
}
flat out vec4 coefficients;`;
const opening =
  header +
  responseAddress +
  `
void main() {
  int index = gl_VertexID;
  vec4 w = atIndex(visible, index);
  if (w.x == 0. && w.z == 0.) {
    gl_Position = vec4(2, 0, 0, 1); gl_PointSize = 1.; coefficients = vec4(0); return;
  }
  vec4 dest = atIndex(frames, 4 * (index / 8) + 3);
  float moment = dest.z < 0. ? dest.w : 0.;
  coefficients = -w.xzxz * dest.y * vec4(1., 1., moment, moment);
  place(8 * int(dest.x) + index % 8);
}`;
const sourceTotal =
  header +
  responseAddress +
  `
uniform int sourceOffset;
void main() {
  int row = gl_VertexID;
  vec4 dest = atIndex(frames, 4 * row + 3);
  if (dest.z < 0.) {
    gl_Position = vec4(2, 0, 0, 1); gl_PointSize = 1.; coefficients = vec4(0); return;
  }
  vec2 sum = vec2(0);
  for (int bin = 0; bin < 8; bin++) {
    vec4 w = atIndex(visible, 8 * row + bin);
    sum += w.xy + w.zw;
  }
  coefficients = vec4(sum, dest.z, 0.);
  place(sourceOffset + int(dest.w));
}`;
const resolve =
  header +
  `
layout(location=0) in vec4 cells;
layout(location=1) in vec4 weights;
layout(location=2) in vec4 moments;
layout(location=3) in vec4 siteData; // texel, contact strength, source offset, source count
layout(location=4) in vec3 background;
uniform highp sampler2D response, radiance, directional;
uniform int sourceOffset;
vec4 responseAt(int i) {
  int w = textureSize(response,0).x;
  return texelFetch(response, ivec2(i % w, i / w), 0);
}
uniform ivec2 chart;
uniform int cellCount;
flat out vec3 rgb;
void main(){
  int site = int(siteData.x);
  int target = cellCount + site;
  vec3 sum = vec3(0.);
  for (int bin = 0; bin < 8; bin++) {
    vec2 open = max(vec2(0), responseAt(8 * target + bin).xy);
    vec2 denominator = vec2(0);
    vec3 low = vec3(0), high = vec3(0);
    for (int j = 0; j < 4; j++) {
      int cell = int(cells[j]);
      vec4 f = responseAt(8 * cell + bin);
      f.xy = max(f.xy, vec2(0));
      f.zw = clamp(f.zw, -f.xy, f.xy);
      denominator += weights[j] * f.xy + moments[j] * f.zw;
      low += weights[j] * texelFetch(directional, ivec2(2 * bin, cell), 0).rgb
           + moments[j] * texelFetch(directional, ivec2(2 * bin + 1, cell), 0).rgb;
      high += weights[j] * texelFetch(directional, ivec2(2 * (bin + 8), cell), 0).rgb
            + moments[j] * texelFetch(directional, ivec2(2 * (bin + 8) + 1, cell), 0).rgb;
    }
    vec2 gain = open / max(denominator, vec2(1. / ${2 * azimuthCount * areaSteps * areaSteps}.));
    sum += gain.x * low + gain.y * high;
  }
  for (int j = 0; j < int(siteData.w); j++) {
    vec4 source = responseAt(sourceOffset + int(siteData.z) + j);
    int side = int(source.z);
    sum += source.x * texelFetch(radiance, ivec2(0, side), 0).rgb
         + source.y * texelFetch(radiance, ivec2(1, side), 0).rgb;
  }
  rgb = mix(background, sum, siteData.y);
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
function packPolygon(batch, polygon, row, sign = 1) {
  const count = polygon.uv.length;
  batch.maxVertices = Math.max(batch.maxVertices, count);
  batch.jobs.push(batch.positions.length / 2, count, row, sign);
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
      this.reuse = this.makeProgram(reuse, saveCoefficients, { cached: 6 });
      this.opening = this.makeProgram(opening, saveCoefficients, { visible: 6, frames: 7 });
      this.sourceTotal = this.makeProgram(sourceTotal, saveCoefficients, { visible: 6, frames: 7 });
      this.final = this.makeProgram(resolve, shade, { response: 2, radiance: 5, directional: 4 });
      this.poly = this.attributes([4], 1);
      this.sampleRows = gl.createVertexArray();
      this.vaos.push(this.sampleRows);
      this.cachedRows = this.attributes([1], 1);
      this.background = { buffer: gl.createBuffer() };
      this.buffers.push(this.background.buffer);
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
        saveCoefficients,
        save ? { positions: 1, frames: 7 } : { positions: 1, cached: 6, frames: 7 },
      );
      setContactPlanes(this.gl, p);
      this.kernels.set(key, p);
    }
    return p;
  }
  beginChart(feature, width, rows, stationary) {
    const targets = feature.size + width * rows;
    if (feature.size > this.limit || targets * 8 > Math.min(contactTextureWidth, this.limit) * this.limit)
      throw Error('Contact chart exceeds GPU texture capacity.');
    let cache = this.staticRecords.get(stationary);
    if (!cache) {
      cache = { count: 0, build: polygonBatch() };
      this.staticRecords.set(stationary, cache);
    }
    const changed = polygonBatch(),
      cachedJobs = [];
    return {
      register(sample) {
        sample.coefficientRow = cache.count++;
        for (const poly of sample.lit) packPolygon(cache.build, poly, sample.coefficientRow);
      },
      reuse(sample) {
        cachedJobs.push(sample.coefficientRow);
      },
      enqueue(sample, site, poly, sign = 1) {
        packPolygon(changed, poly, sample.coefficientRow, sign);
      },
      finish(packets) {
        if (cache.build) {
          cache.batch = sealBatch(cache.build);
          delete cache.build;
        }
        if (packets) return { ...packets, cache, targets };
        return {
          cache,
          cachedJobs: Float32Array.from(cachedJobs),
          changed: sealBatch(changed),
          targets,
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
    gl.uniform2i(gl.getUniformLocation(p, 'table'), width, height);
    this.upload(this.poly, batch.jobs);
    gl.drawArraysInstanced(gl.POINTS, 0, 8, batch.jobs.length / 4);
    if (save) this.stats.staticPolygons += batch.jobs.length / 4;
    else this.stats.polygons += batch.jobs.length / 4;
  }
  target(texture, width, height) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.scratch);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.viewport(0, 0, width, height);
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
    t = { id: gl.createTexture(), frames: gl.createTexture(), cache, width, height };
    try {
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, t.id);
      for (const param of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
        gl.texParameteri(gl.TEXTURE_2D, param, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
      const frameHeight = Math.max(1, Math.ceil((cache.count * 4) / width));
      const sources = new Map();
      const frames = new Float32Array(4 * width * frameHeight);
      const { u, v, n } = contact.feature;
      t.ranges = new Int32Array(2 * (contact.resident.targets - contact.feature.size));
      let slots = 0;
      for (const site of contact.stationary) {
        if (site.texel !== undefined) t.ranges.set([slots, site.samples.length], 2 * site.texel);
        for (const sample of site.samples) {
          const m = sample.m,
            delta = sub(m.center, site.p);
          const offset = sample.coefficientRow * 16;
          const cell = site.cell !== undefined;
          if (!cell && !sources.has(sample.side)) sources.set(sample.side, sources.size);
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
              cell ? site.cell : contact.feature.size + site.texel,
              cell ? 1 / (areaSteps * areaSteps) : 1,
              cell ? -1 : sources.get(sample.side),
              cell ? 3 * site.pos : slots++,
            ],
            offset,
          );
        }
      }
      t.sourceOffset = 8 * contact.resident.targets;
      t.responseHeight = Math.max(1, Math.ceil((t.sourceOffset + slots) / width));
      if (t.responseHeight > this.limit) throw Error('Contact response exceeds GPU texture capacity.');
      t.sources = Array.from(sources.keys());
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, t.frames);
      for (const param of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
        gl.texParameteri(gl.TEXTURE_2D, param, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, frameHeight, 0, gl.RGBA, gl.FLOAT, frames);
      this.stats.uploadBytes += frames.byteLength;
      this.sizeAllocation(t, width * height * 16 + frames.byteLength);
      this.target(t.id, width, height);
      gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      // A rebuilt table can recover the input from retained scene geometry.
      // Release the packed upload once the device owns its coefficients.
      if (!cache.batch) {
        const batch = polygonBatch();
        for (const site of contact.stationary)
          for (const sample of site.samples)
            for (const poly of sample.lit) packPolygon(batch, poly, sample.coefficientRow);
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
  responseTable(grid, contact) {
    const table = this.staticTable(grid, contact);
    const { resident } = contact,
      { changed, cachedJobs } = resident;
    if (table.geometry === resident) return table;
    const gl = this.gl,
      { width, height } = table;
    let visible = table.id;
    if (changed.jobs.length || cachedJobs.length !== table.cache.count) {
      visible = this.texture(3, width, height, null);
      this.target(visible, width, height);
      gl.clearBufferfv(gl.COLOR, 0, new Float32Array(4));
      gl.activeTexture(gl.TEXTURE6);
      gl.bindTexture(gl.TEXTURE_2D, table.id);
      gl.activeTexture(gl.TEXTURE7);
      gl.bindTexture(gl.TEXTURE_2D, table.frames);
      if (cachedJobs.length) {
        gl.disable(gl.BLEND);
        gl.useProgram(this.reuse);
        gl.uniform2i(gl.getUniformLocation(this.reuse, 'table'), width, height);
        this.upload(this.cachedRows, cachedJobs);
        gl.drawArraysInstanced(gl.POINTS, 0, 8, cachedJobs.length);
      }
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      this.drawPolygons(changed, false, width, height);
      gl.disable(gl.BLEND);
    }
    if (!table.response) {
      table.response = { id: gl.createTexture() };
      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, table.response.id);
      for (const key of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
        gl.texParameteri(gl.TEXTURE_2D, key, gl.NEAREST);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, table.responseHeight, 0, gl.RGBA, gl.FLOAT, null);
      this.sizeAllocation(table.response, 16 * width * table.responseHeight);
    }
    this.target(table.response.id, width, table.responseHeight);
    gl.clearBufferfv(gl.COLOR, 0, emptyOpening);
    gl.activeTexture(gl.TEXTURE6);
    gl.bindTexture(gl.TEXTURE_2D, visible);
    gl.activeTexture(gl.TEXTURE7);
    gl.bindTexture(gl.TEXTURE_2D, table.frames);
    gl.bindVertexArray(this.sampleRows);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.opening);
    gl.uniform2i(gl.getUniformLocation(this.opening, 'responseSize'), width, table.responseHeight);
    gl.drawArrays(gl.POINTS, 0, 8 * table.cache.count);
    gl.disable(gl.BLEND);
    gl.useProgram(this.sourceTotal);
    gl.uniform2i(gl.getUniformLocation(this.sourceTotal, 'responseSize'), width, table.responseHeight);
    gl.uniform1i(gl.getUniformLocation(this.sourceTotal, 'sourceOffset'), table.sourceOffset);
    gl.drawArrays(gl.POINTS, 0, table.cache.count);
    table.geometry = resident;
    return table;
  }
  prepareGeometry(contacts) {
    if (!contacts?.size) return;
    try {
      for (const [grid, contact] of contacts) this.responseTable(grid, contact);
      this.gl.flush();
    } finally {
      this.restore();
    }
  }
  releaseTable(t) {
    if (t.response) {
      this.gl.deleteTexture(t.response.id);
      this.sizeAllocation(t.response, 0);
    }
    if (t.siteData) {
      this.gl.deleteVertexArray(t.siteData.vao);
      this.gl.deleteBuffer(t.siteData.buffer);
      this.sizeAllocation(t.siteData, 0);
    }
    this.gl.deleteTexture(t.id);
    this.gl.deleteTexture(t.frames);
    this.sizeAllocation(t, 0);
  }
  retainCharts(charts) {
    this.visibility?.retain(charts);
    for (const [grid, t] of this.tables)
      if (!charts.has(grid)) {
        this.releaseTable(t);
        this.tables.delete(grid);
      }
  }
  bindSites(table, contact, filter) {
    const gl = this.gl;
    const input = (table.siteData ??= { vao: gl.createVertexArray(), buffer: gl.createBuffer() });
    gl.bindVertexArray(input.vao);
    if (input.filter === filter) return;
    const data = new Float32Array(contact.sites.length * 16);
    for (let i = 0; i < contact.sites.length; i++) {
      const site = contact.sites[i],
        o = 4 * site,
        j = 16 * i;
      for (let k = 0; k < 4; k++) {
        data[j + k] = filter.indices[o + k];
        data[j + 4 + k] = filter.weights[o + k];
        data[j + 8 + k] = filter.linearWeights[o + k];
      }
      data.set([site, contact.strength[i], table.ranges[2 * site], table.ranges[2 * site + 1]], j + 12);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, input.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
    this.stats.uploadBytes += data.byteLength;
    this.sizeAllocation(input, data.byteLength);
    for (let k = 0; k < 4; k++) {
      gl.enableVertexAttribArray(k);
      gl.vertexAttribPointer(k, 4, gl.FLOAT, false, 64, 16 * k);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.background.buffer);
    gl.enableVertexAttribArray(4);
    gl.vertexAttribPointer(4, 3, gl.FLOAT, false, 0, 0);
    input.filter = filter;
  }
  paint(view) {
    const gl = this.gl,
      { width, rows, tex } = view.atlas;
    try {
      for (const [grid, { contact, filter }] of view.contactDraws) {
        const { feature, sites } = contact;
        const table = this.responseTable(grid, contact);
        const sol = view.solution,
          { field, slope } = sol.directional.get(feature);
        const dir = new Float32Array(feature.size * 32 * 4);
        for (let i = 0; i < field.length / 3; i++)
          for (let c = 0; c < 3; c++) {
            dir[8 * i + c] = field[3 * i + c];
            dir[8 * i + 4 + c] = slope[3 * i + c];
          }
        this.texture(4, 32, feature.size, dir);
        if (table.sources.length) {
          const lighting = new Float32Array(8 * Math.max(1, table.sources.length));
          table.sources.forEach((side, i) => {
            for (let c = 0; c < 3; c++) {
              lighting[8 * i + c] = sol.outgoing[3 * side + c];
              lighting[8 * i + 4 + c] = sol.outSlope[3 * side + c];
            }
          });
          this.texture(5, 2, lighting.length / 8, lighting);
        } else this.texture(5, 2, 1, new Float32Array(8));
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, table.response.id);
        this.bindSites(table, contact, filter);
        const data = new Float32Array(sites.length * 3);
        for (let i = 0; i < sites.length; i++) {
          const b = 4 * (grid.displayRow * width + sites[i]);
          data[3 * i] = tex[b];
          data[3 * i + 1] = tex[b + 1];
          data[3 * i + 2] = tex[b + 2];
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.background.buffer);
        gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);
        this.stats.uploadBytes += data.byteLength;
        this.sizeAllocation(this.background, data.byteLength);
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
        gl.uniform1i(gl.getUniformLocation(this.final, 'sourceOffset'), table.sourceOffset);
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
    this.visibility = null;
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
