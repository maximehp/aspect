# 3D vector engine

A walkable room with one point lamp and independently tiltable blinds. Run `npm start` and open [the room](http://127.0.0.1:8736/). The server uses Python 3. The browser needs WebAssembly and WebGL 2. There are no package dependencies.

Drag to look and use WASD to walk. Try rotating the blind slowly, then move the lamp slider to zero. Choose **Reflected only** and switch the floor between red and cobalt to see the color of the floor on the plaster. You can also add a second blind, tilt it independently, or rearrange the furniture.

## Authoring the room

The room uses planar surfaces, rounded boxes and linear RGB material reflectances. Lengths are in metres. A rectangle has a center and two half-axes. A rounded box has a center, three orthonormal axes, three outer half-extents and a fillet radius. Each slat is one solid 4 m long, 226 mm wide and 24 mm thick with 8 mm fillets. The renderer derives its draw, shadow and transport representations from that solid. Shapes are immutable. Submit a replacement shape to move it or change its material. [Room definition](src/room.js) · [Rounded box](src/rounded-box.js)

## How a frame gets its light

The CPU computes the lamp-visible regions, diffuse equilibrium and camera visibility. The GPU draws the resulting ordered regions, interpolates the supplied values, applies the dimmer and display curve, and writes pixels. After an edit the first displayed frame already contains the solved field. The preceding path tracer remains available as a quality reference. [CPU renderer](src/room-display.js) · [Display geometry](src/cpu-view.js) · [Reference](src/live-renderer.js)

For fixed geometry, let D be the direct incident light, H the indirect incident light, ρ the surface's linear RGB reflectance, and K the diffuse transfer operator. An opaque diffuse scene obeys **H = K ρ(D + H)**. This accounts for light that reaches a surface after any number of reflections. The room approximates the spatial equation and solves for a mean and a slope on each side of each surface region. It stops when both coefficients satisfy a local update tolerance of 10⁻⁷ lamp units. The positivity limiter is homogeneous, so scaling the single lamp scales the unit-power result before display quantization. [Equilibrium](src/equilibrium.js)

- **Direct shadows.** The CPU projects casters onto receivers and clips their shadow footprints. It integrates the solid angle of the lit polygons for the mean and uses a boundary integral for the first spatial moment. The display uses the lit polygons themselves for the hard shadow. Rounded boxes use tangent support planes for shadow silhouettes, filleted shells for drawing and six broad faces for diffuse exchange. [Visibility](src/visibility.js)
- **Variation in every reflection.** Each region carries RGB coefficients for a field of the form `a + bξ`, where ξ runs from −1 to 1 along one local tangent. The choice of tangent uses the solid face dimensions, lamp geometry and nearby sheet edges. For each visible pair, integrating 1, ξ at each endpoint, and their product gives four coefficients. These transfer a source gradient into both the mean and the gradient of the receiver. Positive endpoint bounds keep the reconstructed light nonnegative. This basis still loses variation across its second tangent and fine occlusion boundaries within an indirectly lit region. [CPU area sweep and solve](src/beam-sweep.c)
- **Near a parallel surface.** An empty-prism check enables the analytic mean form factor between aligned rectangles. Deterministic integration of that formula supplies the higher moments. For the last transfer the CPU can evaluate a clear rectangle's contribution at the receiver position. Planar surfaces share a continuous display field, with the same 48 × 24 samples per chart concentrated near geometric contacts. [Parallel-sheet transfer](src/parallel-transfer.js) · [Surface coordinates](src/irradiance-layout.js)
- **Where work disappears.** The CPU integrates visible area for 96 directions and their opposites using 192 transverse strips. It stores sparse reciprocal links and reuses the stationary links. Moving a solid subtracts blocked stationary exchange and adds newly exposed interactions. A material-only edit reuses the geometric operator. Dimming uses one scalar, looking around updates the camera matrix, and walking reuses the lighting. Up to six CPU workers share a geometry edit and their partial results are merged in a fixed order. Under rapid input the room keeps the newest waiting pose and presents each accepted pose with its lighting solved. [Operator storage](src/beam-exchange.js) · [Scene updates](src/room-display.js)

Storage for exchange follows the number of recorded interactions. A highly connected scene can still produce quadratically many links. Hierarchical compression is a remaining research question for the street, where large groups might share a smooth visible field. Occlusion can split a proposed group even when its members are close together. [Hierarchical radiosity](https://graphics.stanford.edu/papers/rad/) · [Wavelet radiosity](https://graphics.stanford.edu/papers/wavrad/)

## Measuring it

Select **3840 × 2160** and enable **Time GPU completion**. **Scene update** measures wall time from preparation start through field upload. **Finished frame** also includes camera visibility, drawing and synchronous GPU readback. **Last edit to frame** starts at the input handler. Browser compositing and physical refresh are outside those timings. Expand **Last scene update profile** for the CPU breakdown. The worker sweep overlaps lamp visibility and lamp integration, so the displayed stages cannot simply be added. Draw-list construction includes preparing visible fillet faces. The work list records prior-revision timings, independent light probes, whole-image comparisons and remaining failures. [Room benchmark results](OPEN_WORK.md#propagated-spatial-variation-and-sparse-exchange)

For an independent estimate of indirect radiance, run `node scripts/probe-light.mjs` for the bench underside or `node scripts/probe-light.mjs floor 2 2 back 24 262144` for a floor region. The arguments are surface id, two zero-based area indices, side, blind angle in degrees and path count. The probe reports linear RGB and standard errors using the mathematical fillet. It runs independently of the renderer.

The checked-in WebAssembly module runs on the CPU. Rebuild it with Zig from the repository root.

```sh
zig cc -target wasm32-freestanding -O3 -c src/beam-sweep.c -o /tmp/room-beam.o
zig wasm-ld --strip-debug --no-entry --export=sweep --export=mergeLinks --export=openOperator --export=installExact --export=prepareRows --export=solveLinks --export=__heap_base --export-memory /tmp/room-beam.o -o src/beam-sweep.wasm
```

## Two-tuft fur study

With the local server running, open [the fur study](http://127.0.0.1:8736/fur-lab/). **Brush center part** reproduces the measured groom. Drag in **Brush** mode to make your own part. Orbit and zoom, then switch between the CPU appearance, native-pixel individual hairs, and their difference. **Measure at 4K** takes the current groom on a repeatable twelve-view walk and exports per-view costs and image errors. The GPU displays CPU-generated images. The study lives in `fur-lab/` and can also run directly by opening its `index.html` in a WebGL 2 browser.
