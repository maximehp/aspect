# 3D vector engine

A walkable room with a point lamp, a window and independently tiltable blinds. Run `npm start` in this directory and open [the room](http://127.0.0.1:8736/). The server uses Python 3. The browser needs WebGL 2. There are no package dependencies.

Drag the view to look around. WASD moves the camera. Change the lamp intensity, rotate either blind, change the floor material or switch the furniture arrangement. Select 3840 × 2160 and enable **Time GPU completion** to see the cost of each action through framebuffer readback. Graphics work runs on demand.

## Changing the room

Edit [the scene](src/room.js) to position furniture and window slats. Each planar surface has a stable id, a center, two half-axes and a linear RGB reflectance. Treat submitted surfaces as immutable. Replace a surface when an object moves. The point source has a position and a full-power linear RGB intensity. The same source supplies the direct and reflected light.

[The visibility compiler](src/visibility.js) intersects the shadow volume of each opaque planar blocker with each receiver. It stores convex patches in receiver-local coordinates with a direct-light visibility flag. When geometry changes, it compares the new shadow footprints with the previous ones and partitions only the affected receivers. Walking and dimming can use the same patches. A first build checks up to a quadratic number of pairs.

## Where the light comes from

The point lamp is the only energy source. Choose **Reflected only** to see the light that has traveled from the lamp to at least one other surface before reaching the one you see. Moving the blind changes both the sharp stripes of direct light and the diffuse light those stripes send back into the room.

The first reflected contribution is a function of position. At a point on the bench, for example, the renderer asks which illuminated parts of the floor, slats, walls and other surfaces it can see. A sample contributes the local direct radiance times the geometric coupling and the bench reflectance. A stripe on the floor keeps its position, width and direct-light variation throughout that calculation. Each RGB channel is filtered by every material on the path. To see this in the room, switch between **Red tile** and **Cobalt tile** and watch the plaster above the bench.

The field evaluator combines directions over the receiving hemisphere with samples of the lit polygons drawn in proportion to their solid angle at the lamp. This gives more samples to the brightest portions of a large reflector. It can evaluate arbitrary positions on a face. The solver integrates that field over the existing patches to obtain the first reflected contribution `G`. A sparse operator `K` describes subsequent diffuse exchange. It accounts for visible area, absorption at each material and light escaping the room. It uses deterministic cosine-weighted samples and reciprocal surface-to-surface weights. The equilibrium of the **indirect** radiance `I` satisfies

```
I = G + K I
```

Iterations stop at a residual tolerance for this finite system. The direct lamp term is added separately at the visible point. The renderer evaluates the first reflected field at display sample positions, adds the interpolated higher-order contribution, and reconstructs the result on the visible face. Matching coplanar pieces of the same material share samples along unobstructed joins. Apertures, dividers and material changes can create genuine lighting or color boundaries. Finite visibility samples, spatial averaging of later reflections and interpolation still affect the image. The numerical stopping rule covers the discrete solve only.

The solution lives in world space at full lamp power. Walking reuses it. The dimmer scales all radiance before tone mapping, down to black at zero. Geometry or material edits update lighting in a worker. The browser presents completed geometry and lighting together and keeps the camera interactive while a new edit is being computed. The current model covers opaque matte surfaces.

[First reflected light](src/gather.js), [surface exchange](src/exchange.js), [the equilibrium solver](src/radiosity.js) and [surface reconstruction](src/reconstruction.js) implement these steps. [The renderer](src/renderer.js) handles projection, depth and display.

**Shadow update** and **Reflected light update** measure worker time for the result published on that frame. **Finished frame** measures the rendering callback through a synchronous one-pixel GPU readback when timing is enabled. **Last edit to frame** measures elapsed time from the scene request through that readback, including any wait for the worker. All timings exclude browser compositing and display refresh. [Open work and run results](OPEN_WORK.md) records the accuracy, cost and remaining comparisons.

## Checking a lighting estimate

Run `node scripts/probe-light.mjs` for the left half of the bench underside. To choose another area, pass a surface id, zero-based patch coordinates, side, blind angle in degrees and path count. For example, `node scripts/probe-light.mjs floor 2 2 back 24 262144` checks a floor region. The front normal is the cross product of the two half-axes. The probe reports mean linear RGB from the solver and from independently traced paths, the sampling standard errors, and the first and later reflected contributions.
