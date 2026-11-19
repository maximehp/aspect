// CPU narrow phase for rounded occluders. A site shares one lazy perspective
// silhouette across its candidate source rectangles. Only visible 2D geometry
// crosses the boundary to the contact integrator.
typedef struct {
    double x, y, z;
} Vec;
typedef struct {
    double u, v;
} UV;
typedef struct {
    Vec center, u, v;
} Rect;
typedef struct {
    Vec eye;
    Rect source, region;
} Query;
typedef struct {
    Vec center, axis[3], half;
    double r;
} Shape;
typedef struct {
    int site, group, first, last, sourceRect, regionRect;
    int cell, row;
} Address;
typedef struct {
    int offset, count;
} Range;
typedef struct {
    int site, valid, ready;
    Vec rel, core, x, y, z, broad[5], nearOrigin;
    Vec *planes;
    double near, tolerance[5];
} Cone;
static double absd(double x) { return __builtin_fabs(x); }
static double root(double x) { return __builtin_sqrt(x); }
static double maxd(double a, double b) { return a > b ? a : b; }
static double mind(double a, double b) { return a < b ? a : b; }
static Vec add(Vec a, Vec b) { return (Vec){a.x + b.x, a.y + b.y, a.z + b.z}; }
static Vec sub(Vec a, Vec b) { return (Vec){a.x - b.x, a.y - b.y, a.z - b.z}; }
static Vec mul(Vec a, double x) { return (Vec){a.x * x, a.y * x, a.z * x}; }
static double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static double length(Vec a) { return root(dot(a, a)); }
static Vec unit(Vec a) {
    double s = length(a);
    return (Vec){a.x / s, a.y / s, a.z / s};
}
static Vec cross(Vec a, Vec b) {
    return (Vec){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
static double extent(const Shape *s, Vec core, Vec dir) {
    return s->r + (core.x * absd(dot(dir, s->axis[0])) + core.y * absd(dot(dir, s->axis[1])) +
                   core.z * absd(dot(dir, s->axis[2])));
}
static void prepare(Cone *c, int site, const Shape *s, Vec eye) {
    c->site = site;
    c->ready = 0;
    c->rel = sub(s->center, eye);
    const double core[3] = {s->half.x - s->r, s->half.y - s->r, s->half.z - s->r};
    c->core = (Vec){core[0], core[1], core[2]};
    Vec closest = c->rel;
    for (int k = 0; k < 3; k++)
        closest = add(closest, mul(s->axis[k], maxd(-core[k], mind(core[k], -dot(c->rel, s->axis[k])))));
    double dist = length(closest);
    c->valid = dist > s->r + 1e-8;
    if (!c->valid)
        return;
    c->z = unit(closest);
    c->x = unit(cross(c->z, absd(c->z.x) < .8 ? (Vec){1, 0, 0} : (Vec){0, 1, 0}));
    c->y = cross(c->z, c->x);
    c->near = dist - s->r;
    c->nearOrigin = add(eye, mul(c->z, c->near));
    double far = dot(c->z, c->rel) + extent(s, c->core, c->z);
    c->broad[0] = c->z;
    for (int k = 0; k < 2; k++) {
        Vec axis = k ? c->y : c->x;
        double mid = dot(c->rel, axis), r = extent(s, c->core, axis), lo = mid - r, hi = mid + r;
        c->broad[1 + 2 * k] = sub(axis, mul(c->z, lo / (lo < 0 ? c->near : far)));
        c->broad[2 + 2 * k] = sub(mul(c->z, hi / (hi > 0 ? c->near : far)), axis);
    }
    for (int i = 0; i < 5; i++)
        c->tolerance[i] = 1e-9 * (1 + length(c->broad[i]));
}
static int broadHit(const Cone *c, const Query *q) {
    Vec d = sub(q->region.center, q->eye);
    for (int i = 0; i < 5; i++) {
        Vec n = c->broad[i];
        double a = dot(n, d), b = absd(dot(n, q->region.u)) + absd(dot(n, q->region.v)), h = i ? 0 : c->near;
        if (a + b < h - (c->tolerance[i] + 1e-9 * (absd(a) + b + h)))
            return 0;
    }
    return 1;
}
// At some common depth every ray fits inside the rounded solid's inscribed box.
static int covers(const Shape *s, const Query *q) {
    Vec dir = sub(q->region.center, q->eye), origin = sub(q->eye, s->center);
    const double half[3] = {s->half.x, s->half.y, s->half.z};
    double lo = 0, hi = 1;
    for (int i = 0; i < 3; i++) {
        Vec axis = s->axis[i];
        double o = dot(origin, axis), d = dot(dir, axis),
               w = absd(dot(q->region.u, axis)) + absd(dot(q->region.v, axis));
        double h = half[i] - s->r * (1 - 1 / root(3)) - 1e-9 * (1 + absd(o) + absd(d) + w + half[i]);
        double fast = d + w, slow = d - w;
        if (fast > 0)
            hi = mind(hi, (h - o) / fast);
        else if (fast < 0)
            lo = maxd(lo, (h - o) / fast);
        else if (o > h)
            return 0;
        if (slow > 0)
            lo = maxd(lo, (-h - o) / slow);
        else if (slow < 0)
            hi = mind(hi, (-h - o) / slow);
        else if (o < -h)
            return 0;
        if (hi <= lo)
            return 0;
    }
    return 1;
}
typedef struct {
    Vec corner[8], depthAxis;
    double depth[8], denom[8], radius;
} Support;
static double tangent(const Support *s, int i, Vec dir) {
    double x = dot(dir, s->corner[i]), z = s->depth[i], r = s->radius;
    return (x * z + r * root(maxd(0, x * x + z * z - r * r))) / s->denom[i];
}
// A support plane at slope h maximizes X - hZ. The three signs select
// the supporting core corner. All spherical corners have the same radius.
static int extreme(Vec v, Vec z, double h) {
    return (v.x > h * z.x) | ((v.y > h * z.y) << 1) | ((v.z > h * z.z) << 2);
}
static Vec supportPlane(const Cone *c, const Shape *shape, const Support *s, double ca, double sa) {
    Vec dir = add(mul(c->x, ca), mul(c->y, sa));
    Vec v = {dot(shape->axis[0], dir), dot(shape->axis[1], dir), dot(shape->axis[2], dir)};
    int i = extreme(v, s->depthAxis, 0);
    double h = tangent(s, i, dir);
    for (int step = 0;; step++) {
        int next = extreme(v, s->depthAxis, h);
        if (i == next)
            break;
        double candidate = tangent(s, next, dir);
        if (candidate <= h || step == 7) {
            for (int j = 0; j < 8; j++)
                h = maxd(h, tangent(s, j, dir));
            break;
        }
        i = next;
        h = candidate;
    }
    return sub(mul(c->z, h), dir);
}
static void silhouette(Cone *c, const Shape *s, const UV *angles, int angleCount) {
    Support h;
    h.radius = s->r;
    h.depthAxis = (Vec){dot(s->axis[0], c->z), dot(s->axis[1], c->z), dot(s->axis[2], c->z)};
    for (int i = 0; i < 8; i++) {
        Vec offset = add(add(mul(s->axis[0], i & 1 ? c->core.x : -c->core.x),
                             mul(s->axis[1], i & 2 ? c->core.y : -c->core.y)),
                         mul(s->axis[2], i & 4 ? c->core.z : -c->core.z));
        h.corner[i] = add(c->rel, offset);
        h.depth[i] = dot(c->z, h.corner[i]);
        h.denom[i] = h.depth[i] * h.depth[i] - s->r * s->r;
    }
    int n = 0;
    c->planes[n++] = c->z;
    for (int i = 0; i < angleCount; i++)
        c->planes[n++] = supportPlane(c, s, &h, angles[i].u, angles[i].v);
    for (int i = 0; i < 8; i++)
        for (int k = 0; k < 3; k++)
            if (!(i & (1 << k))) {
                Vec normal = cross(h.corner[i], h.corner[i | (1 << k)]);
                double x = dot(normal, c->x), y = dot(normal, c->y), len = root(x * x + y * y);
                double ca = len ? x / len : 1, sa = len ? y / len : 0;
                c->planes[n++] = supportPlane(c, s, &h, ca, sa);
                c->planes[n++] = supportPlane(c, s, &h, -ca, -sa);
            }
    c->ready = 1;
}
// Scratch belongs to a single sample. A capacity miss retries with a larger
// arena; polygon complexity is never truncated to fit a fixed vertex array.
typedef struct {
    unsigned char *data;
    unsigned left;
    int failed;
} Arena;
static void *allocate(Arena *a, unsigned count, unsigned stride) {
    if (a->failed || count > a->left / stride) {
        a->failed = 1;
        return 0;
    }
    unsigned bytes = (count * stride + 7u) & ~7u;
    if (bytes > a->left) {
        a->failed = 1;
        return 0;
    }
    void *p = a->data;
    a->data += bytes;
    a->left -= bytes;
    return p;
}
typedef struct {
    const UV *p;
    int n;
} Poly;
typedef struct Piece {
    Poly poly;
    struct Piece *next;
} Piece;
typedef struct {
    Piece *head, *tail;
} List;
static void append(Arena *a, List *list, Poly p) {
    Piece *node = allocate(a, 1, sizeof(Piece));
    if (!node)
        return;
    node->poly = p;
    node->next = 0;
    if (list->tail)
        list->tail->next = node;
    else
        list->head = node;
    list->tail = node;
}
static void join(List *list, List other) {
    if (!other.head)
        return;
    if (list->tail)
        list->tail->next = other.head;
    else
        list->head = other.head;
    list->tail = other.tail;
}
static int hasArea(Poly p) {
    if (p.n < 3)
        return 0;
    double area = 0;
    UV o = p.p[0];
    for (int j = 1; j + 1 < p.n; j++)
        area += (p.p[j].u - o.u) * (p.p[j + 1].v - o.v) - (p.p[j].v - o.v) * (p.p[j + 1].u - o.u);
    return area > 0;
}
typedef struct {
    double a, b, c, tol;
} Line;
static double side(Line l, UV p) {
    double t = l.a * p.u + l.b * p.v + l.c;
    return absd(t) <= l.tol ? 0 : t;
}
static Poly clip(Arena *a, Poly p, Line line, Poly *outside) {
    int positive = 0, negative = 0;
    for (int i = 0; i < p.n; i++) {
        double d = side(line, p.p[i]);
        positive |= d > 0;
        negative |= d < 0;
        if (positive && negative)
            break;
    }
    if (outside)
        *outside = (Poly){0};
    if (!negative)
        return p;
    if (!positive) {
        if (outside)
            *outside = p;
        return (Poly){0};
    }
    UV *in = allocate(a, p.n, 2 * sizeof(UV));
    UV *out = outside ? allocate(a, p.n, 2 * sizeof(UV)) : 0;
    if (a->failed)
        return (Poly){0};
    int n = 0, m = 0;
    double d = side(line, p.p[0]);
    for (int j = 0; j < p.n; j++) {
        UV v = p.p[j], w = p.p[j + 1 == p.n ? 0 : j + 1];
        double e = side(line, w);
        if (d >= 0)
            in[n++] = v;
        if (out && d <= 0)
            out[m++] = v;
        if ((d > 0 && e < 0) || (d < 0 && e > 0)) {
            double t = d / (d - e);
            UV crossing = {v.u + t * (w.u - v.u), v.v + t * (w.v - v.v)};
            in[n++] = crossing;
            if (out)
                out[m++] = crossing;
        }
        d = e;
    }
    if (outside)
        *outside = (Poly){out, m};
    return (Poly){in, n};
}
static const UV square[4] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
static Poly footprint(Arena *arena, const Cone *c, const Query *q, int planeCount) {
    Poly p = {square, 4};
    Vec delta = sub(q->source.center, c->nearOrigin);
    double size = length(q->source.u) + length(q->source.v), tol = (length(delta) + size) * 0x1p-47;
    for (int i = 0; i < planeCount; i++) {
        if (i == 1) {
            delta = sub(q->source.center, q->eye);
            tol = (length(delta) + size) * 0x1p-47;
        }
        Vec n = c->planes[i];
        double a = dot(n, q->source.u), b = dot(n, q->source.v), h = dot(n, delta), rad = absd(a) + absd(b);
        if ((!i && h + rad <= tol) || h + rad < -tol)
            return (Poly){0};
        if (h - rad >= -tol)
            continue;
        Poly inside = clip(arena, p, (Line){a, b, h, tol}, 0);
        if (inside.p != p.p && !hasArea(inside))
            return (Poly){0};
        p = inside;
    }
    return p;
}
typedef struct {
    double left, right, bottom, top;
} Bounds;
static Bounds bounds(Poly poly) {
    Bounds b = {poly.p[0].u, poly.p[0].u, poly.p[0].v, poly.p[0].v};
    for (int i = 1; i < poly.n; i++) {
        UV p = poly.p[i];
        b.left = mind(b.left, p.u);
        b.right = maxd(b.right, p.u);
        b.bottom = mind(b.bottom, p.v);
        b.top = maxd(b.top, p.v);
    }
    return b;
}
// Preserve the original visible piece if a later edge of the mask proves
// it disjoint. A needless split changes the polygonal source-moment rule.
static List partition(Arena *a, List lit, Poly mask, int onlyCovered, int *changed) {
    Bounds mb = bounds(mask);
    Line *edges = allocate(a, mask.n, sizeof(Line));
    if (!edges)
        return (List){0};
    for (int i = 0; i < mask.n; i++) {
        UV p = mask.p[i], q = mask.p[i + 1 == mask.n ? 0 : i + 1];
        double x = p.v - q.v, y = q.u - p.u, c = -(x * p.u + y * p.v);
        edges[i] = (Line){x, y, c, 0x1p-47 * (absd(x) + absd(y) + absd(c))};
    }
    List result = {0};
    for (Piece *piece = lit.head; piece && !a->failed; piece = piece->next) {
        Poly orig = piece->poly, intersect = orig;
        Bounds b = bounds(orig);
        if (mb.left >= b.right || mb.right <= b.left || mb.bottom >= b.top || mb.top <= b.bottom) {
            if (!onlyCovered)
                append(a, &result, orig);
            continue;
        }
        if (onlyCovered && mask.p == square) {
            append(a, &result, orig);
            continue;
        }
        if (onlyCovered && !piece->next && piece == lit.head && orig.p == square) {
            append(a, &result, mask);
            continue;
        }
        List outside = {0};
        for (int i = 0; i < mask.n; i++) {
            Poly rest;
            Poly inside = clip(a, intersect, edges[i], onlyCovered ? 0 : &rest);
            if (inside.p == intersect.p)
                continue;
            if (!hasArea(inside)) {
                intersect = (Poly){0};
                break;
            }
            intersect = inside;
            if (!onlyCovered && hasArea(rest))
                append(a, &outside, rest);
        }
        if (intersect.n) {
            *changed = 1;
            if (onlyCovered)
                append(a, &result, intersect);
            else
                join(&result, outside);
        } else if (!onlyCovered)
            append(a, &result, orig);
    }
    return result;
}
typedef struct {
    float first, count, row, sign;
} Job;
typedef struct {
    int cached, changed, vertices, maxVertices;
} Counts;
enum { moreScratch = -1, moreJobs = -2, moreVertices = -3 };
_Static_assert(sizeof(Vec) == 24 && sizeof(Rect) == 72 && sizeof(Shape) == 128, "geometry packet layout");
_Static_assert(sizeof(Address) == 32 && sizeof(Range) == 8 && sizeof(Job) == 16, "contact packet layout");
// One near cap and two tangents for each of the twelve core edges.
static int planeCountFor(int angleCount) { return 1 + angleCount + 2 * 12; }
int coneBytes(int angleCount) { return sizeof(Cone) + planeCountFor(angleCount) * sizeof(Vec); }
// Output owns the same two job streams used by the resident GPU passes.
// Sector coefficients, color and lighting never enter this kernel.
int contactPackets(const Vec *eyes, const Rect *rectangles, const Address *addresses, int count,
                   const Shape *shapes, int shapeCount, const Range *apertures, const UV *fixed,
                   const int *groups, const int *indices, const UV *angles, int angleCount, Cone *cones,
                   unsigned char *scratch, int scratchBytes, float *cached, Job *changed, int jobCapacity,
                   float *vertices, int vertexCapacity, Counts *out) {
    *out = (Counts){0};
    Vec *planes = (Vec *)(cones + shapeCount);
    const int planeCount = planeCountFor(angleCount);
    for (int i = 0; i < shapeCount; i++) {
        cones[i].site = -1;
        cones[i].planes = planes + i * planeCount;
    }
    for (int i = 0; i < count; i++) {
        Address r = addresses[i];
        Query query = {eyes[r.site], rectangles[r.sourceRect], rectangles[r.regionRect]};
        Arena arena = {scratch, scratchBytes, 0};
        List masks = {0};
        int numMasks = 0, hidden = 0;
        for (int j = groups[r.group]; j < groups[r.group + 1]; j++) {
            int index = indices[j];
            Cone *cone = cones + index;
            const Query *q = &query;
            if (cone->site != r.site)
                prepare(cone, r.site, shapes + index, q->eye);
            if (!cone->valid || !broadHit(cone, q))
                continue;
            if (covers(shapes + index, q)) {
                hidden = 1;
                break;
            }
            if (!cone->ready)
                silhouette(cone, shapes + index, angles, angleCount);
            Poly poly = footprint(&arena, cone, q, planeCount);
            if (poly.n) {
                append(&arena, &masks, poly);
                numMasks++;
            }
            if (arena.failed)
                return moreScratch;
        }
        if (hidden)
            continue;
        if (masks.head) {
            List lit = {0};
            for (int k = r.first; k < r.last; k++) {
                Range p = apertures[k];
                append(&arena, &lit, (Poly){p.offset < 0 ? square : fixed + p.offset, p.count});
            }
            List original = lit;
            int coveredOnly = numMasks == 1 && r.cell, modified = 0;
            for (Piece *mask = masks.head; mask && lit.head; mask = mask->next)
                lit = partition(&arena, lit, mask->poly, coveredOnly, &modified);
            if (arena.failed)
                return moreScratch;
            if (coveredOnly) {
                Piece *l = lit.head, *o = original.head;
                while (l && o && l->poly.p == o->poly.p) {
                    l = l->next;
                    o = o->next;
                }
                if (!l && !o)
                    continue;
            }
            if (coveredOnly || modified) {
                for (Piece *p = lit.head; p; p = p->next) {
                    Poly poly = p->poly;
                    if (out->changed == jobCapacity)
                        return moreJobs;
                    if (poly.n > vertexCapacity - out->vertices)
                        return moreVertices;
                    changed[out->changed++] = (Job){out->vertices, poly.n, r.row, coveredOnly ? -1 : 1};
                    if (poly.n > out->maxVertices)
                        out->maxVertices = poly.n;
                    for (int k = 0; k < poly.n; k++) {
                        vertices[2 * out->vertices] = poly.p[k].u;
                        vertices[2 * out->vertices + 1] = poly.p[k].v;
                        out->vertices++;
                    }
                }
                if (!coveredOnly)
                    continue;
            }
        }
        cached[out->cached++] = r.row;
    }
    return 0;
}
