import { VisibilityCompiler } from './visibility.js';
import { DiffuseTransport } from './radiosity.js';
import { buildReconstruction } from './reconstruction.js';

const visibility = new VisibilityCompiler();
const transport = new DiffuseTransport();

self.onmessage = ({data: {scene, maxTextureSize}}) => {
  try {
    const start = performance.now();
    const shadows = visibility.update(scene.rectangles, scene.lightPosition);
    const afterShadows = performance.now();
    const { surfaces, field } = transport.update(shadows.surfaces, scene.lightPosition, scene.lampRGB);
    const atlas = buildReconstruction(surfaces, field, maxTextureSize);
    self.postMessage({
      cells: surfaces.map(surface => surface.cells), atlas,
      receivers: shadows.stats.receiversRebuilt, shadowMs: afterShadows - start,
      transportMs: performance.now() - afterShadows,
    }, [atlas.pixels.buffer]);
  } catch (error) {
    self.postMessage({ error: error.message });
  }
};
