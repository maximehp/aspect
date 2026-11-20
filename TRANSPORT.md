# Transport in Aspect

The camera observes a field. Authors supply geometry, material response, media, emitters and an observer. The moving room uses one point emitter, opaque Lambertian surfaces and vacuum. Those properties remove outgoing direction from the material response and remove the volume integral entirely. The optical phenomena list is the long-term coverage target. Its observer effects need a measurement or perception model, and its unexplained entries need a known mechanism before they can become physical acceptance tests. [Coverage inventory](https://en.wikipedia.org/wiki/Optical_phenomenon) · [Open acceptance](OPEN_WORK.md)

## The current pose

For an outward-facing source point $x$, receiver $y$, $d=y-x$, and binary visibility $V$, transfer of outgoing radiance to irradiance is

$$T[L](y)=\int_S V(x,y)L(x,\hat d)\,[n_x\cdot d]_+[-n_y\cdot d]_+\,|d|^{-4}\,dA_x.$$

The room solves $H=E+T[\rho H/\pi]$ in linear RGB. It stores the indirect term $B=H-E$. The first right-hand side is the directly illuminated source integrated over its actual lit portion. Subsequent terms apply the same geometric operator to $\rho B/\pi$. This separation keeps the point lamp’s sharp shadow in the first reflection. The final shader evaluates the direct term at the visible point and interpolates the same indirect field used during reflection. Its established exposure is proportional to $\pi$ times outgoing radiance.

Each new pose rebuilds the geometric operator and starts the equilibrium solve from its current right-hand side. Device allocations and compiled functions survive between poses. The rotating slats, player, bench and artwork use the same geometry rules. Camera visibility runs concurrently on a separate CPU worker. The GPU integrates clear near-field planar apertures using the four great-circle edges of their projected contour, reads one scalar back per aperture, and draws the scene with horizontal lines. [Room](src/room-display.js) · [GPU measure](src/angular-measure.js)

## Eliminating a receiving coordinate from the integration

Consider a planar part of the wall, $x=c+s$, $s=A u+B v$, with $u,v\in[-1,1]$. Keep the floor position $y$ free. Write $D=y-c$, $q=|D|^{-2}$, $h=-n_y\cdot D$, $j=n_y\cdot s$, and $a=q D\cdot s$. With the whole pair facing and mutually visible, the diffuse geometric kernel including $1/\pi$ is

$$G(s,y)=\frac{(n_x\cdot D)q^2}{\pi}\,(h+j)(1-2a+q|s|^2)^{-2}.$$

Its quadratic expansion in $s$ contains six source modes, $1,u,v,u^2,uv,v^2$. The bracket through quadratic order is $h+j+4ha+4ja+h(12a^2-2q|s|^2)$. Each coefficient is a function of floor position and normal. Integrating the lamp-lit wall against these six modes gives six numbers independent of the floor. They include its curved shadow edges. For later bounces the moments of a bilinear source field follow directly from its four coefficients. For example, a corner basis with signs $(i,j)$ has normalized moments $(1,i/3,j/3,1/3,ij/9,1/3)$ before its area Jacobian. The floor calculation contracts the current source moments with the six coefficient functions. Sampling another floor position in this domain requires no new lamp or wall-area integral.

Selection applies to an entire receiver domain. Projected bounds test both facing factors and separate blockers from the convex hull joining the source and receiver. For a separating rectangular blocker, the crossings of its plane lie inside the convex hull of the sixteen corner crossings. This can prove a whole shadow or a whole opening. For the expansion, bound $z=-2a+q|s|^2$ over both domains. If $|z|\le b<1$, the tail of $(1+z)^{-2}$ after the quadratic term in $z$ is bounded by $b^3(4-3b)/(1-b)^2$. The implementation also bounds the cubic and quartic terms discarded when expanding in $s$, the facing factors, area and maximum direct lamp irradiance. The accepted per-interaction absolute budget is $5\times10^{-5}$ for the normalized source modes or direct lamp term. This assumes exact arithmetic in the bound. The numerical source moments have their own integration error. [Receiver-domain proofs and moments](src/plane-transfer.h)

If the facing sign varies, a blocker cannot be separated, or the distance makes that bound too large, the engine uses the clipped integral at individual receiver positions. Curved receivers also use it unless they belong to a planar face at the point in question. The remaining integrals supply construction coefficients on continuous bilinear charts. A narrow indirect stripe between chart knots can be lost. A curved chart uses seam-sharing local functions. A planar chart changes basis on both ends of transfer before equilibrium.

## What the next reflection reads

On a sheet, the outgoing field is a sum of products $B(u,v)=\sum_{ij} c_{ij}\psi_i(u)\psi_j(v)$ in each color channel. The coordinates $u,v$ remain free. A coarse pair spans constant and linear functions on each axis. An interior knot contributes a detail relative to the linear interpolant of its two neighbors. For its fine hat $h$ and neighboring coarse hats $l,r$, choose coefficients $a,b$ so that $h-al-br$ has zero area and zero first moment. For example, if their areas are $m_h,m_l,m_r$ and their centroids are $x_h,x_l,x_r$, then $b=m_h(x_h-x_l)/(m_r(x_r-x_l))$ and $a=(m_h-bm_r)/m_l$. This uses actual nonuniform knot locations. A locally affine kernel integrates such a detail to zero. Repeating the construction gives detail functions at different spatial scales in both coordinates. A query can evaluate the sum at a new floor position without touching wall visibility.

Let $W$ analyze nodal values into those coefficients and $K$ denote the geometric transfer assembled at the current pose, including the projected GPU aperture measures and $1/\pi$. In coefficient coordinates the transfer is $C=W K W^{-1}$. Geometry is compiled once per pose. Where a rounded chart shares edge and pole degrees of freedom, the change of basis there is the identity. With per-object RGB reflectance $R$, which commutes with the per-surface transform, the equation is $c=W T[\rho E/\pi]+C R c$. When the floor emits, the solver reads its coarse and detail coefficients directly from $c$. It never expands them for a Krylov application. One reconstruction supplies the observer and one independent residual check against the original construction coefficients. Construction still materializes columns and rows at individual chart knots. This is a measured cost of the current method.

The retained rank depends on the pose. For each source coefficient use a scale $s_j=\max(0.005,\|d_j\|_\infty)$ from the transformed first reflection $d$. Drop a set of weights per receiving mode with $\sum_j |C_{ij}| s_j\le 10^{-4}$. Quantization of retained weights is included separately in the computed bound. Uniform scale alone retained more links than the nodal representation. The current-pose forcing and two applications of the compressed map seed the active modes. When a full modal residual exceeds $1.5\times10^{-4}$ on an omitted mode, promote it and solve again before display. A final physical-space residual measures the combined effect of pruning, truncation and rounding against the expanded operator. It cannot detect a bright stripe absent from both representations. [Surface basis](src/surface-modes.js) · [Geometric coefficient map](src/modal-operator.c) · [Equilibrium](src/modal-equilibrium.js) · [Room path trace](scripts/trace-room.js)

## A point lamp on a cylindrical fillet

At fixed angle $\theta$, write the cylinder generator as $p(x)=p_0+x e$, $|e|=1$. The normal is independent of $x$. On a visible, lit and facing interval, the source cosine numerator for each of the lamp and receiver is constant. The receiving cosine numerator is $c+m x$. Let $l,y_e$ be the projected axial coordinates of lamp and receiver, and $L,Y$ their respective **squared** distances from the generator line. Apart from a constant factor including radius, the integrand is

$$ (c+m x)Q_l(x)^{-3/2}Q_y(x)^{-2},\qquad Q_l=(x-l)^2+L,\quad Q_y=(x-y_e)^2+Y. $$

A local analytic primitive avoids sampling along $x$. About interval midpoint $x_0$, use $z=(x-x_0)/h$, with $h$ its half-length, $U=Q_l/Q_l(x_0)$ and $W=Q_y/Q_y(x_0)$. Set $g=U^{-3/2}W^{-2}$. Its Taylor coefficients follow from $g(0)=1$ and $(UW)g'=(-\tfrac32 U'W-2UW')g$. With $g_{-1}=0$, a primitive is

$$F(x)=\frac{h}{Q_l(x_0)^{3/2}Q_y(x_0)^2}\sum_{k\ge0}
\frac{(c+m x_0)g_k+mh g_{k-1}}{k+1}z^{k+1}.$$

Evaluate $F(b)-F(a)$ at visible endpoints, sum over openings and integrate over $\theta$. The implementation splits a long generator until both complex pole distances exceed $3h$ and bounds the remaining series using Cauchy’s estimate on $|z|=2$. For bilinear outgoing light on a generator, only the moments of $1,x,x^2$ over $Q_y^2$ remain. Their endpoint formulas use rational functions and an angle difference. Spherical octants retain their curved meridians and area Jacobian. Their remaining integrals and the outer strip integral use deterministic embedded estimates. The per-pair absolute budget is $2\times10^{-4}$ and the outer relative target is 0.002. Refinement exhaustion is counted. These targets apply locally and do not certify every pixel or all visibility topology. [Length primitive and integration](src/function-transport.c)

## Arithmetic and geometry

Text such as `"0.1"` and `"1/10"` describes the same rational length. Numeric inputs retain the binary value actually passed. Slat radii, centers and half-extents preserve their authored definitions. Rotation axes calculated with sine and cosine carry the precision of the pose computation.

For a fillet sphere, write $Q$ for the light position relative to its center and $d$ for the travel vector. The tangent equation is $(Q\cdot d)^2-(Q\cdot Q-r^2)(d\cdot d)=0$. For a cylindrical edge, first project both vectors perpendicular to its axis. The axial extent, the sphere octant or cylinder quadrant, and the finite light segment select the physically relevant roots. Substituting a position on a wall gives an implicit conic across that wall. Along a generator its root is identified by $Ax^2+2Bx+C=0$ and a branch. Its decimal expansion is unnecessary for a side test. Within a pose, a lamp-lit interval union belongs to the source. Identical generator queries can use it on multiple receivers. Unions with the receiving visibility are intersected before integration.

The same feature equation can be restricted to a whole receiver plane as $Q(u,v)=a u^2+2buv+c v^2+2du+2ev+f$. For a cylinder, projecting onto its cross section annihilates the axial direction. If a planar generator family translates exactly along that axis, $b=c=e=0$ in those coordinates. The same tangent roots in $u$ serve every transverse position $v$. The native kernel recognizes the exact axis alignments it can prove in the room. A pose-local table keys those roots by source, observer and feature. Every strip still checks the clipped interval, finite axial extent, quadrant and segment ordering. The spherical caps can restore dependence on $v$. A separate bound on $n\cdot(eye-center)$ over a feature’s allowed unit normals rejects whole features that cannot be tangent. Separation of the source from the solid also eliminates endpoint quadratics. [Exact conic restriction](src/silhouette.js) · [Native clipping](src/function-transport.c)

The kernel uses floating point for ordinary decisions. Near a suspected tangent it can reconstruct the lamp conic from rational authored quantities and exact binary inputs. The camera’s uncertain curved intersections compare candidate rational endpoints with quadratic roots by sign and squaring. Several geometric clipping and depth comparisons still use absolute epsilons. The renderer does not provide a universal certified predicate, a guaranteed downstream physical error or a per-pixel rounding certificate. For its discrete positive equilibrium it reports the residual and, when the infinity-norm contraction bound $q<1$ holds, an error bound of residual divided by $1-q$. [Authored inputs](src/authored-scalar.js) · [Silhouette](src/silhouette.js) · [Filtered ray intersections](src/exact-hit.js)

## Direction, media and the observer

A general source may supply $L(x,\omega)$. The optional operations can apply the room's compiled geometric weights to a position-only source or evaluate a smooth directional source toward each receiver. A perfectly smooth interface instead maps a direction into reflected and transmitted branches using the authored normal, Snell’s law and unpolarized Fresnel energy fractions. For a future wave surface, a refracted map $f$ from incident surface area onto the pool floor gives irradiance density by summing transmitted incident flux divided by $|\det Df|$ over all preimages, in consistent area coordinates. Finite illumination and observer footprints matter at a fold.

A medium supplies density, absorption, scattering and a normalized directional phase response. Along a characteristic, a segment acts on incoming radiance as $L_{out}=T L_{in}+J$. For constant material and phase-weighted incident radiance $S$, $T=\exp(-\sigma_t d)$ and $J=(1-T)(\sigma_s/\sigma_t)S$ per channel. A supplied column-density integral also permits this reduction. The operations intersect authored rounded regions, combine overlapping coefficients, and compose segment responses in travel order. A varying field uses deterministic refinement. The resulting radiance can illuminate a diffuse receiver or enter another segment. A closed 3D solver for an arbitrary scattering cloud, wax body or ideal specular caustic remains open. The playable room never imports the optional optics module. An audit found no dedicated playable water or dust pass. The separate fur study still has a fiber-direction response and coverage model whose equivalence and cost have not been established. [Optical operations](src/optical-operations.js) · [Equation of transfer](https://www.pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer) · [Dielectric interface](https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission)

## Reciprocal transfer between clear parallel planes

For two aligned rectangles separated by distance $h$, the diffuse kernel including the Lambertian $1/\pi$ factor is $K=h^2/[\pi(h^2+x^2+y^2)^2]$. Here $x$ and $y$ are **differences** of tangential coordinates between the surfaces. For bilinear source and receiver basis functions $b_j$ and $b_i$, the shared coefficient is $W_{ij}=\iint b_i(r)K(r-s)b_j(s)\,ds\,dr$. A transpose gives the reverse exchange. Each coefficient has units of square metres. These formulas require clear mutual visibility throughout the two supports.

Integration by parts on each tangential coordinate leaves only values and slope changes at authored or basis boundaries. Values cancel at interior knots of a continuous function. Nine endpoint primitives suffice, with two to four integrations in each of the two relative coordinates. Every source or receiver basis using the same displacement shares its primitive. No direction samples or pointwise receiver integrations enter this block.

To evaluate the primitives, put $D=h^2+s^2+t^2$ and integrate the monomials $s^it^j/D^2$ for $0\leq i,j\leq3$. Integration by parts and the divergence of $(s,t)s^it^j/D$ reduce them to edge values and one integral of $1/D$. For positive endpoint differences let $X=\operatorname{asinh}(x/h)$ and $Y=\operatorname{asinh}(y/h)$. That integral is

$$
\tfrac{\pi}{2}\min(X,Y)-\operatorname{Ti}_2(e^{-|X-Y|})+\operatorname{Ti}_2(e^{-X-Y}),
\qquad \operatorname{Ti}_2(z)=\int_0^z \frac{\arctan t}{t}\,dt.
$$

The code accounts for endpoint orientation. Its positive series for $\operatorname{Ti}_2$ has remainder at most $z[z^2/(1+z^2)]^n$ after $n$ terms. Thin, distant basis supports can still lose digits when nearly equal endpoint primitives are subtracted. Their geometry gives a separate enclosure by multiplying the masses of the two basis functions by the smallest and largest possible kernel on their supports. The probe implementation recovers an out-of-range coefficient from this enclosure when its half-width meets the absolute recovery tolerance and reports the accumulated bound for recovered entries. A wide interval produces a failure. This check leaves arithmetic error inside wide intervals and downstream model error open.

Run `node scripts/probe-plane-exchange.mjs` for the paper back and north wall, or append `print-clay print-paper` for the next layer. The independent check evaluates the four source moments at receiver positions and integrates them over the receiver with a separate quadrature. The room experiment also applied these shared coefficients through diffuse equilibrium. Turning each integrated coefficient back into a nonnegative continuous radiance field is a separate approximation with measured accuracy and cost. [Bilinear operator](src/rectangle-exchange.js) · [Independent integral check](scripts/probe-plane-exchange.mjs) · [Room experiment](benchmarks/reciprocal-plane-room.json)

## Reproducing the room measurements

The measurement archive contains the three 120-pose replays, isolated light paths, edits, physical reference estimates and output comparisons. The 4K replay uses the normal room entry point and a fresh solve for every pose. In a browser with the local room loaded, import `traceRoom` from `/scripts/trace-room.js` and call it with a live renderer and a scene from `buildRoom`. It isolates the light reflected by the rounded bodies and the wall light passing around their shadows. [Moving-room evidence](benchmarks/moving-room-4k.json) · [Handoff audit](scripts/trace-room.js)

The independent reference can condition on a sequence of diffuse reflections. The example below starts at the ceiling center, samples cosine-weighted paths through the floor and north wall, and measures both the direct lamp contribution there and the contribution reflected by the filleted slat. The latter uses true area sampling on the flat, cylindrical and spherical parts to resolve a weak path. Output includes a standard error for each channel.

```sh
node scripts/probe-light.mjs ceiling 0 0 front 17.188733853924695 2097152 \
  '{"slats":1,"radius":0.044}' '{"via":["floor","north-wall"],"point":[0.5,0.5]}'
```

## A reflection over a finite receiving region

Let \(R\) be a region on the receiving floor and \(W\) an intermediate wall. Write \(K(w,y)\) for the reciprocal diffuse kernel, including geometric visibility and the Lambertian factor of \(1/\pi\). For incident wall irradiance \(H_W(w)\), the mean reflected contribution on \(R\), in the room's unit-lamp convention, is

$$
\frac{\rho_R\rho_W}{|R|}\int_W H_W(w) A_R(w)\,dw,
\qquad A_R(w)=\int_R K(w,y)\,dy .
$$

The point lamp and the illuminated rounded bodies produce separate functions \(H_W\). The first keeps the indicator of the filleted lamp shadow. The second integrates the directly illuminated planes, cylinder generators and spherical corners as seen from the wall. RGB reflectances multiply channel by channel. A slat or bench occluding the next segment enters \(A_R\) through the current geometry. A different receiving region changes this function.

For fixed \(w\), project sheet blockers onto \(R\), clip to the wall's forward hemisphere and take the union of their shadows. Each visible contour edge has endpoint directions \(a,b\) of unit length. Its signed contribution is the wall-normal component of \((a\times b)\operatorname{atan2}(|a\times b|,a\cdot b)/(2\pi |a\times b|)\), with orientation toward the wall. Contributions on shared internal edges cancel. The GPU evaluates these angular measures. Thus both coordinates of \(y\) disappear from numerical integration for this **region mean**. A curved obstruction that survives the geometric bounds requires the clipped native generator integral. The outer wall integral remains numerical because the incident function and the visible outline both vary with \(w\). Successive refinements estimate convergence. They do not bound missed features between evaluations.

There is a useful zero contribution for the playable equilibrium as well. In a sheet blocker's own coordinates let the receiver have depth \(-d\), \(d>0\), and horizontal coordinate \(u_r\). For a source with depth \(z>0\), a ray crosses inside the edge \(\sigma u\le1\) when \(1-\sigma u_s+z(1-\sigma u_r)/d\ge0\). The slope \((1-\sigma u_r)/d\) attains its minimum at a corner of a rectangular receiver because it is a ratio of affine functions with positive denominator. The four limiting edge planes and the depth plane therefore suffice for *all* receiver positions. Strict enclosure of the source support proves it sends zero across that entire pair. Ambiguous pairs continue to the existing classification. A single support test can also cover every chart of one rounded object. This proof and the lamp-shadow boundary functions are rebuilt from each pose.

In the room browser, import `comparePlanePath` from `/scripts/probe-wall-floor.js` and call it with a renderer and a scene from `buildRoom`. The default checks the north wall and a finite region of the floor. Supply `{panels:[4,8,16,32]}` to inspect the refinements or `{tolerance:{relative:0.005,absolute:1e-9}}` to request an estimated tolerance with a refinement limit. It also isolates these paths in the playable field and integrates that field over the same receiver region. To measure the paths independently:

```sh
node scripts/probe-light.mjs floor 2 2 back 17.188733853924695 2097152 \
  '{"slats":1,"radius":0.044}' '{"via":["north-wall"],"region":[0.5,0.75,0.4,0.6]}'
```

[Finite-region composition](src/plane-composition.js) · [Current-pose shadow functions](src/plane-shadow.js) · [Room measurements](benchmarks/wall-floor-composition.json)

## Spatial weights for a clear rectangular source

An orthogonal rectangle with clear visibility and zero direct lamp irradiance still sends the spatially varying light it received from other surfaces. Four coefficients transfer its bilinear outgoing field. They enter the same equilibrium operator on every bounce. In receiver-centred coordinates let the source lie at height $h>0$, with $D=h^2+x^2+y^2$. In the forward hemisphere the kernel is $h(n_z h+n_x x+n_y y)/(\pi D^2)$. The implementation retains receiver position in this expression. It currently evaluates the expression at the room's receiver nodes.

Three reductions apply according to geometry and conditioning.

* For a well-separated source, expand $D^{-2}$ around the centre of the source. With $z$ bounding the fractional change in $D$, the tail after the quadratic term is bounded by $z^3(4-3z)/(1-z)^2$ times the centre value. Each signed hat on $[-1,1]$ has normalized moments $1/(k+1)$ for even $k$ and $\pm1/(k+2)$ for odd $k$. Products of the two one-dimensional moments integrate the expansion exactly. This branch takes no samples inside the source.
* For a source of appreciable angular extent, integrate $1,x,y,xy$ against the kernel at its edges and corners. Integrating by parts reduces the required monomials over $D^2$ to elementary edge terms and $J=\int 1/D\,dx\,dy$. In the tilted case the four corners give $J$ through the inverse tangent integral described above. For parallel surfaces its coefficient vanishes. The endpoint formula can lose significance on a thin support or at grazing incidence.
* On a straight generator at fixed $y$, set $g=\sqrt{h^2+y^2}$ and $x=g\tan\phi$. Let $a$ be the angle between the endpoint rays, $N_0,N_1$ the receiver-normal cosines there, and $d_0,d_1$ the distances to the endpoints. The linear source hats supply sines of the angle from the opposite endpoint. Put $S=(a-\sin a\cos a)/(2\sin^2 a)$ and $C=(\sin a-a\cos a)/(2\sin^2 a)$. The two length weights, before the common $1/\pi$, are $h(SN_0+CN_1)/(g d_0)$ and $h(CN_0+SN_1)/(g d_1)$. Both are positive. At small angles the shader evaluates their series. The remaining transverse integral uses two, four or eight Gauss nodes. Geometry selects the transverse axis.

For the last reduction a bound selects the order. Over the transverse interval enclose $D$ between $L$ and $U$ for each longitudinal point. The largest relative spread occurs at the nearest longitudinal point. Write $s=\sqrt{LU}$, $a=(L+U)/2$ and $q=(\sqrt U-\sqrt L)/(\sqrt U+\sqrt L)$. The Chebyshev coefficients of $1/D^2$ for degrees $k>0$ are $2(-q)^k(a/s+k)/s^2$, obtained by differentiating the cosine expansion of the reciprocal with respect to $a$. An $N$-node rule integrates the terms through degree $N-2$ in $D$ exactly, including the affine numerator and transverse hat. For $m=N-1$ the remaining uniform error is at most $2q^m(a/s+m+q/(1-q))/(s^2(1-q))$. Positive quadrature weights bound the integral difference by twice this tail times the hat mass and the largest numerator. The per-coefficient approximation target is $10^{-10}+2\times10^{-5}$ times the kernel scale used in dispatch. The adaptive generator handles cases outside the selected domains. The tail bounds concern approximation in real arithmetic. GPU rounding and equilibrium representation have separate measured errors.

The CPU chooses the applicable expression. The GPU evaluates a batch of source/receiver index pairs using the current pose's geometry tables. Each result contributes to the four actual source nodes. The observer still draws analytic scanline runs. [Selection and bounds](src/rectangle-rule.h) · [GPU integrals](src/rectangle-aperture.js)

## Cylindrical silhouette generators

In a fillet's transverse plane let $q$ point from the centre of a corner circle of radius $r$ to the eye. A tangent normal obeys $q\cdot n=r$ and $|n|=1$, giving $n=(r q\pm\sqrt{|q|^2-r^2}\,q^\perp)/|q|^2$. Keep the normals lying in the authored quadrant. Along each box axis the projected rounded rectangle has at most two tangent support lines. Each lifts to a plane through the eye and a straight, finite generator on the cylinder. These planes describe the whole pose for that eye.

Clip a candidate source interval against these half-spaces. At a limiting plane, intersect the ray with the corresponding finite tangent generator and check segment depth. If both extremes reach the solid, convexity proves the entire interval is blocked. If only the infinite extension of a generator was reached, its spherical corner can supply the actual boundary. The solver retains those corner conics. Uncertain signs, coincident bounds and source endpoints that may lie in the solid use the general feature calculation. These rules use each rounded object's own axes, extents and radius. [Silhouette and interval proof](src/cylinder-outline.h) · [Measured room results](benchmarks/rectangle-and-fillet.json)
