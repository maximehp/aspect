// Paired-path control variate. For a nearby pose, the first m indexed paths
// in its estimate have counterparts at a previously refined pose. Correct
// their sampling error by the difference between the anchor's long and short
// estimates. The correction disappears as the current estimate accumulates.
const prefixes = [1, 2, 4, 8];
const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const distance = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));

function poseConfidence(a, b) {
  const boxesA = a?.roundedBoxes ?? [],
    boxesB = b.roundedBoxes ?? [];
  if (
    !a ||
    a.rectangles.length !== b.rectangles.length ||
    boxesA.length !== boxesB.length ||
    !same(a.lightPosition, b.lightPosition) ||
    !same(a.lampRGB, b.lampRGB)
  )
    return 0;
  for (let i = 0; i < a.rectangles.length; i++) {
    const r = a.rectangles[i],
      s = b.rectangles[i];
    if (
      r !== s &&
      (r.id !== s.id ||
        !same(r.center, s.center) ||
        !same(r.axisU, s.axisU) ||
        !same(r.axisV, s.axisV) ||
        !same(r.color, s.color))
    )
      return 0;
  }
  let movement = 0;
  for (let i = 0; i < boxesA.length; i++) {
    const r = boxesA[i],
      s = boxesB[i];
    if (r === s) continue;
    if (r.id !== s.id || !same(r.halfSize, s.halfSize) || r.radius !== s.radius || !same(r.color, s.color))
      return 0;
    const translation = distance(r.center, s.center) / Math.min(...r.halfSize);
    movement = Math.max(movement, translation, ...r.axes.map((axis, j) => distance(axis, s.axes[j])));
  }
  // Nearby orientations share more of their light paths. A smoothly tapered
  // coefficient prevents poorly correlated poses increasing the variance.
  return Math.exp(-0.5 * (movement / 0.27) ** 2);
}

function sameSites(a, b) {
  if (!a || a.width !== b.width || a.height !== b.height || a.objectCount !== b.objectCount) return false;
  // Only the addressing must agree. Points on a moving solid follow its frame.
  const x = a.descriptors,
    y = b.descriptors;
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i += 16) {
    for (let c = 0; c < 4; c++) if (x[i + c] !== y[i + c]) return false;
  }
  return true;
}

const fragment = `#version 300 es
precision highp float;
uniform highp sampler2D currentField, anchorLong, anchorShort;
uniform float weight;
out vec4 estimate;
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  vec3 now=texelFetch(currentField,p,0).rgb;
  vec3 correction=texelFetch(anchorLong,p,0).rgb-texelFetch(anchorShort,p,0).rgb;
  estimate=vec4(max(vec3(0.),now+weight*correction),1.);
}`;

export function createPairedHistory(gl, compileFullscreen) {
  const prog = compileFullscreen(fragment);
  const inputs = ['currentField', 'anchorLong', 'anchorShort'].map((name) =>
    gl.getUniformLocation(prog, name),
  );
  const weight = gl.getUniformLocation(prog, 'weight'),
    framebuffer = gl.createFramebuffer(),
    vao = gl.createVertexArray();
  const texture = () => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  };
  const bank = () => ({ parts: prefixes.map(texture), scene: null, layout: null });
  let anchor = bank(),
    pending = bank();
  const high = texture(),
    corrected = texture();
  const textures = [high, corrected, ...anchor.parts, ...pending.parts];
  let width = 0,
    height = 0,
    latest = null,
    batchCount = 0,
    confidence = 0,
    similarity = 0;
  function copy(source, dest) {
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, source, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, dest);
    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, width, height);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
  }
  return {
    setScene(previous, next, layout) {
      const resized = layout.width !== width || layout.height !== height;
      if (resized) {
        width = layout.width;
        height = layout.height;
        gl.activeTexture(gl.TEXTURE0);
        for (const t of textures) {
          gl.bindTexture(gl.TEXTURE_2D, t);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null);
        }
        anchor.scene = null;
      } else if (batchCount >= 16 && latest && previous) {
        copy(latest, high);
        [anchor, pending] = [pending, anchor];
        anchor.scene = previous;
      }
      similarity = sameSites(anchor.layout, layout) ? poseConfidence(anchor.scene, next) : 0;
      if (similarity < 0.02) similarity = 0;
      pending.layout = layout;
      batchCount = 0;
      latest = null;
      confidence = 0;
    },
    update(raw, batches) {
      latest = raw;
      batchCount = batches;
      const slot = prefixes.indexOf(batches);
      if (slot >= 0) copy(raw, pending.parts[slot]);
      if (!similarity || batches >= 32) {
        confidence = 0;
        return raw;
      }
      let i = prefixes.length - 1;
      while (prefixes[i] > batches) i--;
      const proportion = Math.min(1, (32 - batches) / 16);
      confidence = ((similarity * prefixes[i]) / batches) * proportion;
      gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, corrected, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.viewport(0, 0, width, height);
      gl.bindVertexArray(vao);
      gl.useProgram(prog);
      for (const [unit, t] of [raw, high, anchor.parts[i]].entries()) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.uniform1i(inputs[unit], unit);
      }
      gl.uniform1f(weight, confidence);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      return corrected;
    },
    get confidence() {
      return confidence;
    },
    dispose() {
      for (const t of textures) gl.deleteTexture(t);
      gl.deleteProgram(prog);
      gl.deleteVertexArray(vao);
      gl.deleteFramebuffer(framebuffer);
    },
  };
}
