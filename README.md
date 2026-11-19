# 3D vector engine

A walkable room with one point lamp and independently tiltable blinds. Run `npm start` and open [the room](http://127.0.0.1:8736/). The server uses Python 3. The browser needs WebAssembly and WebGL 2. There are no package dependencies.

Drag to look and use WASD to walk. Try rotating the blind slowly, then move the lamp slider to zero. Choose **Reflected only** and switch the floor between red and cobalt to see the color of the floor on the plaster. You can also add a second blind, tilt it independently, or rearrange the furniture.

## Authoring the room

The room uses planar surfaces, rounded boxes and linear RGB material reflectances. Lengths are in metres. A rectangle has a center and two half-axes. A rounded box has a center, three orthonormal axes, three outer half-extents and a fillet radius. Each slat is one solid 4 m long, 226 mm wide and 24 mm thick with 8 mm fillets. The renderer derives its draw, shadow and transport representations from that solid. Shapes are immutable. Submit a replacement shape to move it or change its material. [Room definition](src/room.js) · [Rounded box](src/rounded-box.js)

## How a frame gets its light

The CPU computes the lamp-visible regions, diffuse equilibrium and camera visibility. The GPU draws the resulting ordered regions, interpolates the supplied values, applies the dimmer and display curve, and writes pixels. After an edit the first displayed frame already contains the solved field. The preceding path tracer remains available as a quality reference. [CPU renderer](src/room-display.js) · [Display geometry](src/cpu-view.js) · [Reference](src/live-renderer.js)

For fixed geometry, let D be the direct incident field, H the indirect incident field, ρ the surface's linear RGB reflectance, and K the diffuse transfer operator. The room solves **H = K ρ(D + H)**. Repeated substitution represents every order of diffuse reflection. A unit of light escapes through an opening or loses energy on reflection. Scaling the lamp therefore scales the entire linear solution exactly before display quantization. The CPU iterates the discrete equation until the largest change in an incident coefficient is at most 10⁻⁷ lamp units. That stopping criterion measures iteration error. The spatial basis and visibility integration also contribute error. [Equilibrium solver](src/equilibrium.js)

- **Lamp visibility.** Project the casters onto each receiver and clip their shadow footprints. Integrating the solid angle of the lit polygons gives each planar mode its mean direct input from the point lamp. The display retains the actual lit/dark polygons separately, so a shadow boundary can cut through a mode. A rounded box uses tangent support planes for its shadow silhouette, a filleted drawing shell for its appearance and six planar faces for diffuse exchange. All three derive from the authored solid.
- **Exchange over visible area.** Viewed along an orthographic direction, two adjacent depth layers exchange diffuse energy in proportion to their common projected area. The CPU integrates lengths and partial heights in transverse strips for 96 fixed directions and their opposites, using 192 strips over the stationary extent. Each neighboring pair contributes in both directions. A stationary operator is reusable. Adding or moving a solid removes stationary links it blocks and adds links to the newly adjacent layers in its projected footprint. Editing a wall can invalidate the stationary operator. Crossing depths, finite angular coverage, strip integration and limiting a row to unit energy are approximations. [CPU area sweep](src/beam-sweep.c)
- **Variation within a surface.** The solve uses mean outgoing RGB on finite surface regions. For rounded solids the same area integral also accumulates the first moment along each face's short axis. A linear function reconstructed from that moment represents the reflected-light gradient across a slat. Adjoining pieces of plaster use a shared continuous incident-light field across their joins. For closely spaced aligned sheets, an empty-prism check licenses an analytic rectangle-to-rectangle coefficient and a continuous point-to-rectangle field for the last transfer. An intervening object disables that calculation. The remaining directions use the ordinary area operator. The 48 × 24 display samples per planar chart bound the resolution of that reconstruction. [Parallel-sheet transfer](src/parallel-transfer.js) · [Shared planar charts](src/irradiance-layout.js)
- **Reuse.** A dimmer edit needs one scalar. Looking around changes a camera matrix. Walking can change visibility and the CPU draw list, while reusing equilibrium. A blind edit rebuilds the affected transport and solves the coupled field before drawing. A developer can add a second blind, change the floor or move the furniture using the same scene API. The current implementation retains a dense stationary operator and working matrix, and compresses rows for iteration. Their storage grows quadratically with the number of sided modes.

The design builds on finite-area radiosity. A hierarchy could save work where surfaces see a common smooth field. A moving slat can split visibility across many interactions, so geometric proximity alone cannot certify a reusable or low-rank block. The room measures explicit visibility changes and analytic cases first. [Hierarchical radiosity](https://graphics.stanford.edu/papers/rad/) · [Wavelet radiosity](https://graphics.stanford.edu/papers/wavrad/)

## Measuring it

Choose **3840 × 2160** for a full-size framebuffer. **Scene update** measures CPU preparation and texture submission. **Finished frame** includes scene update, camera visibility, drawing and synchronous readback. **Last edit to frame** includes the wait from the input handler. Enable **Time GPU completion** for the latter two measurements. Browser compositing and physical refresh are outside those timings. The room's comparison against the settled path tracer, independent area probes and current limitations are recorded in the work list. [Measurements and acceptance](OPEN_WORK.md#deterministic-equilibrium-in-the-room)

For an independent estimate of indirect radiance, run `node scripts/probe-light.mjs` for the bench underside or `node scripts/probe-light.mjs floor 2 2 back 24 262144` for a floor region. The arguments are surface id, two zero-based area indices, side, blind angle in degrees and path count. The probe reports linear RGB and standard errors using the mathematical fillet. It runs independently of the renderer.

The checked-in WebAssembly module runs on the CPU. Rebuild it with Zig from the repository root.

```sh
zig cc -target wasm32-freestanding -O3 -c src/beam-sweep.c -o /tmp/room-beam.o
zig wasm-ld --strip-debug --no-entry --export=sweep --export=equilibrium --export=__heap_base --export-memory /tmp/room-beam.o -o src/beam-sweep.wasm
```
