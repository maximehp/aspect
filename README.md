# 3D vector engine

A walkable room with one point lamp and independently tiltable blinds. Run `npm start` and open [the room](http://127.0.0.1:8736/). The server uses Python 3. The browser needs WebAssembly and WebGL 2. There are no package dependencies.

Drag to look and use WASD to walk. The room opens with one slat. Change its fillet radius at a fixed angle, then enable **Animate slats** or add a second independently tilted slat. **Full blind** restores the nine-slat bank. Move the lamp slider to zero to darken the room. Choose **Reflected only** and switch the floor between red and cobalt to see the color of the floor on the plaster. The furniture layout is editable.

## Authoring the room

The room uses planar surfaces, rounded boxes and linear RGB material reflectances. Lengths are in metres. A rectangle has a center and two half-axes. A rounded box has a center, three orthonormal axes, three outer half-extents and a fillet radius. The single-slat control authors a 3.5 m × 340 mm × 90 mm solid with an adjustable 1–45 mm radius. The full blind uses 4 m × 226 mm × 24 mm slats with an initial 8 mm radius. The renderer derives its draw, shadow and transport representations from that solid. Shapes are immutable. Submit a replacement shape to move it or change its material. [Room definition](src/room.js) · [Rounded box](src/rounded-box.js)

## How a frame gets its light

The CPU derives visibility and a diffuse lighting operator from the current pose of the complete scene. The lamp's shadow clips the actual planes, cylindrical faces and spherical corners. On suitable wall-to-floor pairs, six moments of the lit wall give the receiving light as a function of floor position. Other pairs use deterministic clipped integration. The GPU evaluates clear angular apertures and draws horizontal pixel runs. Dimming scales the solved single-lamp response. The path tracer remains available for independent comparison. [Mathematics, authoring and limits](TRANSPORT.md) · [Room solver](src/function-lighting.js) · [Open acceptance and measurements](OPEN_WORK.md)

## Measuring it

The [continuous motion benchmark](http://127.0.0.1:8736/motion.html) is the primary speed measurement. Choose one slat, two independent slats or two full blinds. A camera and its shadow-casting body travel together while the selected slats rotate and a separate solid moves near the bench. In the single-slat and two-slat replays, the radii also vary. Each of 120 successive 60 Hz simulation ticks goes through the normal scene API, equilibrium solve, camera update, full 4K draw and GPU readback. The page reports sustained finished frames per second, frame-time percentiles, missed work budgets and cold start. Expand **Mean moving-frame profile** for the additive breakdown through readback and the overlapping CPU work inside the exchange stage. A second pass advances game time at 60 Hz regardless of rendering speed and reports how many poses were shown or skipped. The two-second trajectory is identical on all backends. Choose four or eight additional moving solids to stress overlapping silhouettes and local contact shadows.

The target is **1 ms for a complete moving frame at 4K**. The function-based room starts each pose from fresh lighting, transfer and visibility. The point-receiver and historical contact implementations are available on the benchmark page. [Motion scene](src/motion-benchmark.js)

To measure beyond readback, run the local server and launch a separate windowed Chrome with a fresh profile and remote debugging enabled on port 9222. With Python packages `websocket-client` and `Pillow` installed, run `python3 scripts/measure-presentation.py function 4`. Select `function` for production, `pointwise` for the fresh individual-receiver comparison, or a historical contact backend, and 0, 4 or 8 extra solids. Block the optional optics module with `--exclude-optional-optics`. Use `--digest-frames` for a separate whole-image comparison. Add `--slats 1` or `--slats 2` to measure the individual slats. The script opens a benchmark tab, marks each finished 4K frame, matches captures to simulation ticks and reports start-to-swap median/p95 plus any missing pose IDs. Browser capture and marker drawing add overhead. The endpoint is Chrome’s frame-swap timestamp. Physical panel scanout requires a separate measurement. [Swap probe](scripts/measure-presentation.py) · [Chrome frame timestamp](https://chromedevtools.github.io/devtools-protocol/tot/Page/#type-ScreencastFrameMetadata)

Select **3840 × 2160** and enable **Time GPU completion**. **Scene update** measures wall time from preparation start through visible-field reconstruction, camera visibility, upload and draw submission. **Finished frame** includes synchronous GPU readback. **Last edit to frame** starts at the input handler. Browser compositing and physical refresh are outside those timings. Expand **Last scene update profile** for the CPU breakdown. The worker sweep overlaps lamp visibility and lamp integration, so the displayed stages cannot simply be added. Scanline visibility can overlap the lighting solve. The work list records prior-revision timings, independent light probes, whole-image comparisons and remaining failures. [Room benchmark results](OPEN_WORK.md#propagated-spatial-variation-and-sparse-exchange)

For an independent estimate of indirect radiance, run `node scripts/probe-light.mjs` for the bench underside or `node scripts/probe-light.mjs floor 2 2 back 24 262144` for a floor region. The arguments are surface id, two zero-based area indices, side, blind angle in degrees and path count. The probe reports linear RGB and standard errors using the mathematical fillet. It runs independently of the renderer. A final JSON argument accepts ordinary room edits. Reproduce the two-slat floor probe with `node scripts/probe-light.mjs floor 2 2 back 17.188733853924695 131072 '{"slats":1,"radius":0.044,"secondBlind":true,"secondRadius":0.033,"secondAngle":-0.4}'`. Authored angles in JSON are radians.

Rebuild the modules with Zig from the repository root.

```sh
zig cc -target wasm32-freestanding -O3 -ffreestanding -c src/function-transport.c -o /tmp/aspect-function.o
zig wasm-ld --strip-debug --no-entry --export=transport --export=buildRegions --export=__heap_base --export-memory -z stack-size=1048576 /tmp/aspect-function.o -o src/function-transport.wasm
zig cc -target wasm32-freestanding -O3 -c src/beam-sweep.c -o /tmp/room-beam.o
zig wasm-ld --strip-debug --no-entry --export=sweep --export=mergeLinks --export=openOperator --export=installExact --export=prepareRows --export=solveLinks --export=__heap_base --export-memory /tmp/room-beam.o -o src/beam-sweep.wasm
zig cc -target wasm32-freestanding -O3 -c src/contact-visibility.c -o /tmp/aspect-visibility.o
zig wasm-ld --strip-debug --no-entry --export=coneBytes --export=contactPackets --export=__heap_base --export-memory /tmp/aspect-visibility.o -o src/contact-visibility.wasm
zig cc -target wasm32-freestanding -O3 -c src/fillet-kernel.c -o /tmp/aspect-fillet.o
zig wasm-ld --strip-debug --no-entry --export=renderRuns --export=surfaceExchange --export=directSites --export=recoverFlux --export=__heap_base --export-memory -z stack-size=262144 /tmp/aspect-fillet.o -o src/fillet-kernel.wasm
```

## Two-tuft fur study

With the local server running, open [the fur study](http://127.0.0.1:8736/fur-lab/). **Brush center part** reproduces the measured groom. Drag in **Brush** mode to make your own part. Orbit and zoom, then switch between the CPU appearance, native-pixel individual hairs, and their difference. **Measure at 4K** takes the current groom on a repeatable twelve-view walk and exports per-view costs and image errors. The GPU displays CPU-generated images. The study lives in `fur-lab/` and can also run directly by opening its `index.html` in a WebGL 2 browser.
