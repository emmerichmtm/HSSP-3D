"""Numba kernels for the 3-D hypervolume subset selection algorithm.

Conventions
-----------
All points are in *maximisation form*: a point p spans the anchored box
[0,p_x] x [0,p_y] x [0,p_z].  The n input points are mutually non-dominated
and in general position (pairwise distinct coordinates).

The xy-projection lives on an (n+3) x (n+3) index grid.  Grid index

    0        <->  coordinate  -margin      (left / bottom side of the square sigma)
    1        <->  coordinate  0            (the axes)
    1 + r    <->  r-th smallest coordinate among the n points (r = 1..n)
    n + 2    <->  coordinate  M + margin   (right / top side of sigma)

A vertex of any graph T(Q), Q a subset of P, is identified by the *global*
vertex id  ``vid = I * (n+3) + J``.  This makes vertices, edges and triangles
of triangulations T(Q) for different Q directly comparable, which is what the
notion of "Q-compliance" of the paper needs.

Bit masks over triangles are stored in ``WORDS`` int64 words of ``BITS`` bits.
"""
import numpy as np
from numba import njit, types
from numba.typed import Dict

WORDS = 5
BITS = 62
MAX_TRIANGLES = WORDS * BITS
_KEY_T = types.UniTuple(types.int64, WORDS)
_STAMP_LIMIT = 2_000_000_000


# ----------------------------------------------------------------------------
# exact hypervolume of a union of anchored boxes, and the brute-force reference
# ----------------------------------------------------------------------------
@njit(cache=True)
def hv3d(X, Y, Z, idx, m):
    """Volume of the union of the anchored boxes of the points idx[:m]."""
    if m == 0:
        return 0.0
    order = np.empty(m, np.int64)
    for a in range(m):
        p = idx[a]
        b = a
        while b > 0 and Z[order[b - 1]] < Z[p]:
            order[b] = order[b - 1]
            b -= 1
        order[b] = p
    # 2-D staircase (x ascending, y descending) of the points seen so far
    sx = np.empty(m, np.float64)
    sy = np.empty(m, np.float64)
    tx = np.empty(m, np.float64)
    ty = np.empty(m, np.float64)
    ns = 0
    vol = 0.0
    for t in range(m):
        p = order[t]
        px = X[p]
        py = Y[p]
        dominated = False
        for a in range(ns):
            if sx[a] >= px and sy[a] >= py:
                dominated = True
                break
        if not dominated:
            nt = 0
            placed = False
            for a in range(ns):
                if sx[a] <= px and sy[a] <= py:
                    continue
                if (not placed) and sx[a] > px:
                    tx[nt] = px
                    ty[nt] = py
                    nt += 1
                    placed = True
                tx[nt] = sx[a]
                ty[nt] = sy[a]
                nt += 1
            if not placed:
                tx[nt] = px
                ty[nt] = py
                nt += 1
            for a in range(nt):
                sx[a] = tx[a]
                sy[a] = ty[a]
            ns = nt
        area = 0.0
        prev = 0.0
        for a in range(ns):
            area += (sx[a] - prev) * sy[a]
            prev = sx[a]
        znext = 0.0
        if t + 1 < m:
            znext = Z[order[t + 1]]
        vol += area * (Z[p] - znext)
    return vol


@njit(cache=True)
def _next_combination(comb, k, n):
    """Advance comb[:k] (strictly increasing, values < n). False when exhausted."""
    i = k - 1
    while i >= 0 and comb[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    comb[i] += 1
    for j in range(i + 1, k):
        comb[j] = comb[j - 1] + 1
    return True


@njit(cache=True)
def brute_force(X, Y, Z, n, k):
    """Enumerate all k-subsets.  Returns (best volume, best subset, #subsets)."""
    comb = np.arange(k).astype(np.int64)
    best = -1.0
    best_comb = comb.copy()
    count = 0
    while True:
        v = hv3d(X, Y, Z, comb, k)
        count += 1
        if v > best:
            best = v
            best_comb[:] = comb
        if not _next_combination(comb, k, n):
            break
    return best, best_comb, count


@njit(cache=True)
def greedy(X, Y, Z, n, k):
    """Classical (1-1/e) greedy.  Returns (volume, subset)."""
    sel = np.empty(k + 1, np.int64)
    used = np.zeros(n, np.bool_)
    cur = 0.0
    for t in range(k):
        bp = -1
        bv = -1.0
        for p in range(n):
            if used[p]:
                continue
            sel[t] = p
            v = hv3d(X, Y, Z, sel, t + 1)
            if v > bv:
                bv = v
                bp = p
        sel[t] = bp
        used[bp] = True
        cur = bv
    return cur, sel[:k].copy()


# ----------------------------------------------------------------------------
# the triangulated projection graph T(Q)   (Section 4 of the paper)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _add_tri(tri, owner, F, a, b, c, own, gx, gy, W):
    ax = gx[a // W]
    ay = gy[a % W]
    cr = (gx[b // W] - ax) * (gy[c % W] - ay) - (gy[b % W] - ay) * (gx[c // W] - ax)
    tri[F, 0] = a
    if cr > 0.0:
        tri[F, 1] = b
        tri[F, 2] = c
    else:
        tri[F, 1] = c
        tri[F, 2] = b
    owner[F] = own
    return F + 1


@njit(cache=True)
def build_T(sel, m, gi, gj, Z, n, gx, gy, tri, owner):
    """Build the canonical triangulation T(Q) of the square sigma, Q = sel[:m].

    Writes counter-clockwise triangles (global vertex ids) into ``tri`` and the
    point whose top face contains the triangle into ``owner`` (-1: not covered
    by any box).  Returns the number of triangles, which equals 2|V| - 6.
    The output only depends on the *set* Q (points are processed by z).
    """
    W = n + 3
    SW = 0
    NW = n + 2
    SE = (n + 2) * W
    NE = (n + 2) * W + (n + 2)
    F = 0
    if m == 0:
        F = _add_tri(tri, owner, F, SW, SE, NE, -1, gx, gy, W)
        F = _add_tri(tri, owner, F, SW, NE, NW, -1, gx, gy, W)
        return F

    # points by decreasing z: position a < t  <=>  a is higher than t
    o = np.empty(m, np.int64)
    for a in range(m):
        p = sel[a]
        b = a
        while b > 0 and Z[o[b - 1]] < Z[p]:
            o[b] = o[b - 1]
            b -= 1
        o[b] = p
    qi = np.empty(m, np.int64)
    qj = np.empty(m, np.int64)
    for t in range(m):
        qi[t] = gi[o[t]]
        qj[t] = gj[o[t]]
    # positions sorted by x- and by y-grid index
    ordI = np.empty(m, np.int64)
    ordJ = np.empty(m, np.int64)
    for a in range(m):
        b = a
        while b > 0 and qi[ordI[b - 1]] > qi[a]:
            ordI[b] = ordI[b - 1]
            b -= 1
        ordI[b] = a
        b = a
        while b > 0 and qj[ordJ[b - 1]] > qj[a]:
            ordJ[b] = ordJ[b - 1]
            b -= 1
        ordJ[b] = a
    # visible part of the top edge of t starts at x-index TL[t]; visible part
    # of the right edge starts at y-index BR[t]   (1 = the edge reaches the axis)
    TL = np.ones(m, np.int64)
    BR = np.ones(m, np.int64)
    for t in range(m):
        for a in range(t):
            if qj[a] > qj[t] and qi[a] > TL[t]:
                TL[t] = qi[a]
            if qi[a] > qi[t] and qj[a] > BR[t]:
                BR[t] = qj[a]

    g = np.empty(4 * m + 8, np.int64)   # interior vertices of the monotone path gamma
    ch = np.empty(m + 4, np.int64)      # top / right chain

    for t in range(m):
        ti = qi[t]
        tj = qj[t]
        own = o[t]
        vq = ti * W + tj
        # --- monotone path gamma(q,Q) from the top-left to the bottom-right corner
        ng = 0
        ci = TL[t]
        cj = tj
        while True:
            h1 = 1
            b = -1
            for a in range(t):
                if qi[a] > ci and qj[a] > h1:
                    h1 = qj[a]
                    b = a
            if ci == 1:                      # walking down the y-axis
                for r in range(m - 1, -1, -1):
                    c = ordJ[r]
                    if qj[c] < cj and qj[c] > h1:
                        g[ng] = W + qj[c]
                        ng += 1
            g[ng] = ci * W + h1
            ng += 1
            if h1 == 1:                      # walking right on the x-axis
                for r in range(m):
                    c = ordI[r]
                    if qi[c] > ci and qi[c] < ti:
                        g[ng] = qi[c] * W + 1
                        ng += 1
                break
            if qi[b] > ti:
                break
            g[ng] = qi[b] * W + h1
            ng += 1
            ci = qi[b]
            cj = h1
        # --- diagonals from v_q to the interior vertices of gamma
        for a in range(ng - 1):
            F = _add_tri(tri, owner, F, vq, g[a], g[a + 1], own, gx, gy, W)
        # --- top segment: fan from the first interior vertex of gamma
        nc = 0
        ch[nc] = TL[t] * W + tj
        nc += 1
        for r in range(m):
            c = ordI[r]
            if c > t and BR[c] == tj:
                ch[nc] = qi[c] * W + tj
                nc += 1
        ch[nc] = vq
        nc += 1
        for a in range(nc - 1):
            F = _add_tri(tri, owner, F, g[0], ch[a], ch[a + 1], own, gx, gy, W)
        # --- right segment: fan from the last interior vertex of gamma
        nc = 0
        ch[nc] = ti * W + BR[t]
        nc += 1
        for r in range(m):
            c = ordJ[r]
            if c > t and TL[c] == ti:
                ch[nc] = ti * W + qj[c]
                nc += 1
        ch[nc] = vq
        nc += 1
        for a in range(nc - 1):
            F = _add_tri(tri, owner, F, g[ng - 1], ch[a], ch[a + 1], own, gx, gy, W)

    # --- outer face: fan from NE over the outer staircase
    ng = 0
    maxj = 1
    for r in range(m - 1, -1, -1):          # x descending -> 2-D skyline
        c = ordI[r]
        if qj[c] > maxj:
            maxj = qj[c]
            g[ng] = c
            ng += 1
    prev = W + qj[g[ng - 1]]                 # (0, y_max)
    top_vertex = prev
    for a in range(ng - 1, -1, -1):
        c = g[a]
        cur = qi[c] * W + qj[c]
        F = _add_tri(tri, owner, F, NE, prev, cur, -1, gx, gy, W)
        lowj = 1
        if a > 0:
            lowj = qj[g[a - 1]]
        nxt = qi[c] * W + lowj
        F = _add_tri(tri, owner, F, NE, cur, nxt, -1, gx, gy, W)
        prev = nxt
    right_vertex = prev                      # (x_max, 0)
    # --- fan from SW over the two axes
    prev = top_vertex
    for r in range(m - 2, -1, -1):
        cur = W + qj[ordJ[r]]
        F = _add_tri(tri, owner, F, SW, prev, cur, -1, gx, gy, W)
        prev = cur
    cur = W + 1
    F = _add_tri(tri, owner, F, SW, prev, cur, -1, gx, gy, W)
    prev = cur
    for r in range(m):
        cur = qi[ordI[r]] * W + 1
        F = _add_tri(tri, owner, F, SW, prev, cur, -1, gx, gy, W)
        prev = cur
    # --- the two remaining corners of sigma
    F = _add_tri(tri, owner, F, NW, SW, top_vertex, -1, gx, gy, W)
    F = _add_tri(tri, owner, F, NW, top_vertex, NE, -1, gx, gy, W)
    F = _add_tri(tri, owner, F, SE, SW, right_vertex, -1, gx, gy, W)
    F = _add_tri(tri, owner, F, SE, right_vertex, NE, -1, gx, gy, W)
    return F


@njit(cache=True)
def tri_area(tri, t, gx, gy, W):
    a = tri[t, 0]
    b = tri[t, 1]
    c = tri[t, 2]
    ax = gx[a // W]
    ay = gy[a % W]
    return 0.5 * ((gx[b // W] - ax) * (gy[c % W] - ay) - (gy[b % W] - ay) * (gx[c // W] - ax))


# ----------------------------------------------------------------------------
# stamped directed-edge table:  key(u -> v) = u * NV + v
# ----------------------------------------------------------------------------
@njit(cache=True)
def _new_stamp(ctr, arr):
    ctr[0] += 1
    if ctr[0] >= _STAMP_LIMIT:
        arr[:] = 0
        ctr[0] = 1
    return ctr[0]


@njit(cache=True)
def _stamp_edges(tri, F, NV, estamp, etri, ectr):
    s = _new_stamp(ectr, estamp)
    for t in range(F):
        for e in range(3):
            key = tri[t, e] * NV + tri[t, (e + 1) % 3]
            estamp[key] = s
            etri[key] = t
    return s


@njit(cache=True)
def boundary_edges(tri, F, W, inside, estamp, etri, ectr):
    """Directed boundary edges (u,v) of the domain D = union of flagged
    triangles; D lies to the left of u -> v."""
    NV = W * W
    s = _stamp_edges(tri, F, NV, estamp, etri, ectr)
    out = np.empty((3 * F, 2), np.int64)
    nb = 0
    for t in range(F):
        if not inside[t]:
            continue
        for e in range(3):
            u = tri[t, e]
            v = tri[t, (e + 1) % 3]
            rk = v * NV + u
            if estamp[rk] == s and inside[etri[rk]]:
                continue
            out[nb, 0] = u
            out[nb, 1] = v
            nb += 1
    return out[:nb]


@njit(cache=True)
def mark_boundary(bed, NV, bstamp, bctr):
    s = _new_stamp(bctr, bstamp)
    for b in range(bed.shape[0]):
        bstamp[bed[b, 0] * NV + bed[b, 1]] = s
    return s


@njit(cache=True)
def eval_in_domain(tri, owner, F, W, gx, gy, Z, bed, bs, estamp, etri, ectr, bstamp,
                   inside, stack):
    """Is the domain D (directed boundary ``bed``, already marked with stamp
    ``bs``) compliant with the triangulation (tri, F)?  If so, flag the
    triangles inside D and return vol( U(Q) cap (D x R) )."""
    NV = W * W
    s = _stamp_edges(tri, F, NV, estamp, etri, ectr)
    for t in range(F):
        inside[t] = False
    top = 0
    for b in range(bed.shape[0]):
        key = bed[b, 0] * NV + bed[b, 1]
        if estamp[key] != s:
            return False, 0.0
        t = etri[key]
        if not inside[t]:
            inside[t] = True
            stack[top] = t
            top += 1
    while top > 0:
        top -= 1
        t = stack[top]
        for e in range(3):
            u = tri[t, e]
            v = tri[t, (e + 1) % 3]
            if bstamp[u * NV + v] == bs:
                continue
            rk = v * NV + u
            if estamp[rk] == s:
                t2 = etri[rk]
                if not inside[t2]:
                    inside[t2] = True
                    stack[top] = t2
                    top += 1
    vol = 0.0
    for t in range(F):
        if inside[t] and owner[t] >= 0:
            vol += tri_area(tri, t, gx, gy, W) * Z[owner[t]]
    return True, vol


# ----------------------------------------------------------------------------
# base case:  Phi_comp(S, D, l) by enumeration   (Lemma 4.2)
# ----------------------------------------------------------------------------
@njit(cache=True)
def base_case(Ssel, mS, cand, nc, ell, bed, gi, gj, Z, n, gx, gy,
              estamp, etri, ectr, bstamp, bctr):
    """max vol(U(S+Q) cap (D x R)) over Q subset of cand, |Q| <= ell, such that
    D is (S+Q)-compliant.  Returns (value, best Q, |best Q|, #subsets tried)."""
    W = n + 3
    NV = W * W
    bs = mark_boundary(bed, NV, bstamp, bctr)
    mmax = mS + ell
    tri = np.empty((10 * mmax + 16, 3), np.int64)
    owner = np.empty(10 * mmax + 16, np.int64)
    inside = np.zeros(10 * mmax + 16, np.bool_)
    stack = np.empty(10 * mmax + 16, np.int64)
    sel = np.empty(mmax, np.int64)
    for a in range(mS):
        sel[a] = Ssel[a]
    best = -1.0
    best_q = np.empty(max(ell, 1), np.int64)
    best_size = 0
    tried = 0
    top = min(ell, nc)
    comb = np.empty(max(top, 1), np.int64)
    for size in range(top, -1, -1):
        for a in range(size):
            comb[a] = a
        while True:
            for a in range(size):
                sel[mS + a] = cand[comb[a]]
            F = build_T(sel, mS + size, gi, gj, Z, n, gx, gy, tri, owner)
            ok, vol = eval_in_domain(tri, owner, F, W, gx, gy, Z, bed, bs, estamp, etri,
                                     ectr, bstamp, inside, stack)
            tried += 1
            if ok and vol > best:
                best = vol
                best_size = size
                for a in range(size):
                    best_q[a] = cand[comb[a]]
            if size == 0 or not _next_combination(comb, size, nc):
                break
    return best, best_q, best_size, tried


# ----------------------------------------------------------------------------
# helpers on a fixed triangulation
# ----------------------------------------------------------------------------
@njit(cache=True)
def topology(tri, F, W, estamp, etri, ectr):
    """Local vertex numbering, local triangles and triangle neighbours."""
    NV = W * W
    s = _stamp_edges(tri, F, NV, estamp, etri, ectr)
    allv = np.sort(tri[:F].copy().reshape(3 * F))
    verts = np.empty(3 * F, np.int64)
    V = 0
    for a in range(3 * F):
        if a == 0 or allv[a] != allv[a - 1]:
            verts[V] = allv[a]
            V += 1
    verts = verts[:V].copy()
    ltri = np.empty((F, 3), np.int64)
    nbr = np.empty((F, 3), np.int64)
    for t in range(F):
        for e in range(3):
            ltri[t, e] = np.searchsorted(verts, tri[t, e])
            rk = tri[t, (e + 1) % 3] * NV + tri[t, e]
            if estamp[rk] == s:
                nbr[t, e] = etri[rk]
            else:
                nbr[t, e] = -1
    return verts, ltri, nbr


@njit(cache=True)
def vertex_cover_bits(verts, W, pt_of_gi, pt_of_gj, pos_in_s0):
    """For each vertex the bit set of the points of S_0 that define it (P_v)."""
    V = verts.shape[0]
    cover = np.zeros(V, np.int64)
    for a in range(V):
        I = verts[a] // W
        J = verts[a] % W
        p = pt_of_gi[I]
        if p >= 0 and pos_in_s0[p] >= 0:
            cover[a] |= (1 << pos_in_s0[p])
        p = pt_of_gj[J]
        if p >= 0 and pos_in_s0[p] >= 0:
            cover[a] |= (1 << pos_in_s0[p])
    return cover


@njit(cache=True)
def locate_points(tri, F, W, gx, gy, inside, gi, gj, skip, n):
    """Triangle of the domain containing v_p for every point p (or -1)."""
    out = np.full(n, -1, np.int64)
    eps = -1e-13
    for p in range(n):
        if skip[p]:
            continue
        x = gx[gi[p]]
        y = gy[gj[p]]
        for t in range(F):
            if not inside[t]:
                continue
            ok = True
            for e in range(3):
                u = tri[t, e]
                v = tri[t, (e + 1) % 3]
                ux = gx[u // W]
                uy = gy[u % W]
                if (gx[v // W] - ux) * (y - uy) - (gy[v % W] - uy) * (x - ux) < eps:
                    ok = False
                    break
            if ok:
                out[p] = t
                break
    return out


@njit(cache=True)
def flags_to_words(inside, F):
    w = np.zeros(WORDS, np.int64)
    for t in range(F):
        if inside[t]:
            w[t // BITS] |= (1 << (t % BITS))
    return w


@njit(cache=True)
def words_to_flags(w, F):
    inside = np.zeros(F, np.bool_)
    for t in range(F):
        if (w[t // BITS] >> (t % BITS)) & 1:
            inside[t] = True
    return inside


@njit(cache=True)
def _popcount(x):
    c = 0
    while x != 0:
        x &= x - 1
        c += 1
    return c


# ----------------------------------------------------------------------------
# enumeration of the partitions of D induced by short simple cycles of T(S+S_0)
# ----------------------------------------------------------------------------
@njit(cache=True)
def enumerate_partitions(ltri, nbr, F, V, inside, cover, full, L, first_only):
    """All distinct ways in which a simple cycle gamma of T(S+S_0) with
    |gamma| <= L and  P_gamma \\ S = S_0  cuts the domain D into two non-empty
    sides.  Each partition is returned as the bit mask (over triangles) of the
    side of D not containing the first triangle of D.

    Every such cycle contains an edge interior to D; it is generated exactly
    once, from its lowest-ranked interior edge.
    Returns (masks [count x WORDS], number of cycles found).
    """
    adj = np.zeros((V, V), np.bool_)
    for t in range(F):
        for e in range(3):
            u = ltri[t, e]
            v = ltri[t, (e + 1) % 3]
            adj[u, v] = True
            adj[v, u] = True
    nbl = np.empty((V, V), np.int64)
    deg = np.zeros(V, np.int64)
    for u in range(V):
        for v in range(V):
            if adj[u, v]:
                nbl[u, deg[u]] = v
                deg[u] += 1
    # edges interior to D
    erank = np.full((V, V), -1, np.int64)
    eu = np.empty(3 * F, np.int64)
    ev = np.empty(3 * F, np.int64)
    ne = 0
    t0 = -1
    for t in range(F):
        if not inside[t]:
            continue
        if t0 < 0:
            t0 = t
        for e in range(3):
            t2 = nbr[t, e]
            if t2 > t and inside[t2]:
                u = ltri[t, e]
                v = ltri[t, (e + 1) % 3]
                erank[u, v] = ne
                erank[v, u] = ne
                eu[ne] = u
                ev[ne] = v
                ne += 1

    cyc = np.zeros((V, V), np.int64)
    cstamp = 0
    side = np.zeros(F, np.bool_)
    stack = np.empty(F, np.int64)
    seen = Dict.empty(key_type=_KEY_T, value_type=types.int64)
    out = np.empty((64, WORDS), np.int64)
    nout = 0
    ncycles = 0

    path = np.empty(L + 2, np.int64)
    ptr = np.empty(L + 2, np.int64)
    covs = np.empty(L + 2, np.int64)
    onpath = np.zeros(V, np.bool_)
    dist = np.empty(V, np.int64)
    queue = np.empty(V, np.int64)
    w = np.zeros(WORDS, np.int64)

    for e in range(ne):
        u = eu[e]
        v = ev[e]
        # BFS distances to u (admissible bound for closing the cycle)
        for a in range(V):
            dist[a] = V + L
        dist[u] = 0
        qh = 0
        qt = 0
        queue[qt] = u
        qt += 1
        while qh < qt:
            a = queue[qh]
            qh += 1
            for r in range(deg[a]):
                b = nbl[a, r]
                if dist[b] > dist[a] + 1:
                    dist[b] = dist[a] + 1
                    queue[qt] = b
                    qt += 1
        path[0] = u
        path[1] = v
        onpath[u] = True
        onpath[v] = True
        covs[1] = cover[u] | cover[v]
        ptr[1] = 0
        d = 1
        while d >= 1:
            cur = path[d]
            if ptr[d] >= deg[cur]:
                onpath[cur] = False
                d -= 1
                continue
            x = nbl[cur, ptr[d]]
            ptr[d] += 1
            r = erank[cur, x]
            if r >= 0 and r < e:
                continue
            if x == u:
                if d >= 2 and covs[d] == full:
                    ncycles += 1
                    # ---- record the partition induced by the cycle path[0..d]
                    cstamp += 1
                    for a in range(d + 1):
                        b = path[a]
                        c = path[(a + 1) % (d + 1)]
                        cyc[b, c] = cstamp
                        cyc[c, b] = cstamp
                    for t in range(F):
                        side[t] = False
                    side[0] = True
                    top = 0
                    stack[top] = 0
                    top += 1
                    while top > 0:
                        top -= 1
                        t = stack[top]
                        for e2 in range(3):
                            t2 = nbr[t, e2]
                            if t2 < 0 or side[t2]:
                                continue
                            if cyc[ltri[t, e2], ltri[t, (e2 + 1) % 3]] == cstamp:
                                continue
                            side[t2] = True
                            stack[top] = t2
                            top += 1
                    flip = side[t0]
                    for a in range(WORDS):
                        w[a] = 0
                    for t in range(F):
                        if inside[t] and (side[t] != flip):
                            w[t // BITS] |= (1 << (t % BITS))
                    key = (w[0], w[1], w[2], w[3], w[4])
                    if key not in seen:
                        seen[key] = 1
                        if nout == out.shape[0]:
                            out2 = np.empty((2 * nout, WORDS), np.int64)
                            out2[:nout] = out
                            out = out2
                        for a in range(WORDS):
                            out[nout, a] = w[a]
                        nout += 1
                        if first_only:
                            return out[:nout], ncycles
                continue
            if onpath[x]:
                continue
            if d + 1 + dist[x] > L:
                continue
            ncov = covs[d] | cover[x]
            if _popcount(full & ~ncov) > 2 * (L - (d + 2)):
                continue
            d += 1
            path[d] = x
            onpath[x] = True
            covs[d] = ncov
            ptr[d] = 0
        onpath[u] = False
    return out[:nout], ncycles


@njit(cache=True)
def split_components(tri, owner, nbr, F, W, gx, gy, Z, inside, words, cand_tri, n):
    """Connected components D_1..D_t of the two sides of a partition of D.

    Returns (ncomp, candidate count per component, vol(U(S') cap D_i x R) per
    component, triangle bit mask per component, component of every point)."""
    label = np.full(F, -1, np.int64)
    stack = np.empty(F, np.int64)
    ncomp = 0
    for s in range(F):
        if (not inside[s]) or label[s] >= 0:
            continue
        sd = (words[s // BITS] >> (s % BITS)) & 1
        label[s] = ncomp
        top = 0
        stack[top] = s
        top += 1
        while top > 0:
            top -= 1
            t = stack[top]
            for e in range(3):
                t2 = nbr[t, e]
                if t2 < 0 or (not inside[t2]) or label[t2] >= 0:
                    continue
                if ((words[t2 // BITS] >> (t2 % BITS)) & 1) != sd:
                    continue
                label[t2] = ncomp
                stack[top] = t2
                top += 1
        ncomp += 1
    cnt = np.zeros(ncomp, np.int64)
    cvol = np.zeros(ncomp, np.float64)
    cwords = np.zeros((ncomp, WORDS), np.int64)
    for t in range(F):
        c = label[t]
        if c >= 0:
            cwords[c, t // BITS] |= (1 << (t % BITS))
            if owner[t] >= 0:
                cvol[c] += tri_area(tri, t, gx, gy, W) * Z[owner[t]]
    pcomp = np.full(n, -1, np.int64)
    for p in range(n):
        if cand_tri[p] >= 0:
            pcomp[p] = label[cand_tri[p]]
            cnt[pcomp[p]] += 1
    return ncomp, cnt, cvol, cwords, pcomp


# ----------------------------------------------------------------------------
# fused helpers used by the Python-level recursion
# ----------------------------------------------------------------------------
@njit(cache=True)
def prepare(sel, mS, dwords, gi, gj, Z, n, gx, gy, estamp, etri, ectr):
    """T(S), the flags of D, its directed boundary and the candidates P cap D."""
    W = n + 3
    tri = np.empty((10 * mS + 16, 3), np.int64)
    owner = np.empty(10 * mS + 16, np.int64)
    F = build_T(sel, mS, gi, gj, Z, n, gx, gy, tri, owner)
    inside = words_to_flags(dwords, F)
    bed = boundary_edges(tri, F, W, inside, estamp, etri, ectr)
    skip = np.zeros(n, np.bool_)
    for a in range(mS):
        skip[sel[a]] = True
    loc = locate_points(tri, F, W, gx, gy, inside, gi, gj, skip, n)
    nc = 0
    for p in range(n):
        if loc[p] >= 0:
            nc += 1
    cand = np.empty(nc, np.int64)
    nc = 0
    for p in range(n):
        if loc[p] >= 0:
            cand[nc] = p
            nc += 1
    return tri, owner, F, inside, bed, cand


@njit(cache=True)
def greedy_in_domain(Ssel, mS, cand, nc, ell, bed, gi, gj, Z, n, gx, gy,
                     estamp, etri, ectr, bstamp, bctr):
    """Greedy feasible solution of Phi_comp(S, D, ell): an incumbent for pruning."""
    W = n + 3
    NV = W * W
    bs = mark_boundary(bed, NV, bstamp, bctr)
    mmax = mS + ell
    tri = np.empty((10 * mmax + 16, 3), np.int64)
    owner = np.empty(10 * mmax + 16, np.int64)
    inside = np.zeros(10 * mmax + 16, np.bool_)
    stack = np.empty(10 * mmax + 16, np.int64)
    sel = np.empty(mmax + 1, np.int64)
    for a in range(mS):
        sel[a] = Ssel[a]
    used = np.zeros(nc, np.bool_)
    F = build_T(sel, mS, gi, gj, Z, n, gx, gy, tri, owner)
    ok, best = eval_in_domain(tri, owner, F, W, gx, gy, Z, bed, bs, estamp, etri, ectr,
                              bstamp, inside, stack)
    m = mS
    for step in range(ell):
        bc = -1
        bv = best
        for c in range(nc):
            if used[c]:
                continue
            sel[m] = cand[c]
            F = build_T(sel, m + 1, gi, gj, Z, n, gx, gy, tri, owner)
            ok, v = eval_in_domain(tri, owner, F, W, gx, gy, Z, bed, bs, estamp, etri, ectr,
                                   bstamp, inside, stack)
            if ok and v > bv:
                bv = v
                bc = c
        if bc < 0:
            break
        used[bc] = True
        sel[m] = cand[bc]
        m += 1
        best = bv
    return best, sel[mS:m].copy()


@njit(cache=True)
def s0_upper_bound(tri, owner, F, W, gx, gy, X, Y, Z, inside, sel, m, loc, n, ell2, vol2):
    """Upper bound on every partition value below S' = sel[:m]:
    vol(U(S') cap D) + the ell2 largest marginal-gain bounds (submodularity)."""
    gain_pt = np.zeros(n, np.float64)
    if ell2 == 0:
        return vol2, gain_pt
    base = hv3d(X, Y, Z, sel, m)
    tmp = np.empty(m + 1, np.int64)
    for a in range(m):
        tmp[a] = sel[a]
    gains = np.zeros(n, np.float64)
    ng = 0
    for p in range(n):
        if loc[p] < 0:
            continue
        tmp[m] = p
        g_glob = hv3d(X, Y, Z, tmp, m + 1) - base
        g_dom = 0.0
        for t in range(F):
            if not inside[t]:
                continue
            h = 0.0
            if owner[t] >= 0:
                h = Z[owner[t]]
            if Z[p] <= h:
                continue
            a = tri[t, 0]
            b = tri[t, 1]
            c = tri[t, 2]
            mnx = min(gx[a // W], gx[b // W], gx[c // W])
            mny = min(gy[a % W], gy[b % W], gy[c % W])
            if mnx < X[p] and mny < Y[p]:
                g_dom += tri_area(tri, t, gx, gy, W) * (Z[p] - h)
        gains[ng] = min(g_glob, g_dom)
        gain_pt[p] = gains[ng]
        ng += 1
    gs = np.sort(gains[:ng])
    ub = vol2
    for a in range(min(ell2, ng)):
        ub += gs[ng - 1 - a]
    return ub, gain_pt


@njit(cache=True)
def partition_upper_bound(ncomp, cnt, cvol, pcomp, gain_pt, n, cap, ell2):
    """Upper bound on  sum_i Psi(S', D_i, l_i)  over all budgets with l_i <= cap."""
    ub = 0.0
    pool = np.empty(n, np.float64)
    npool = 0
    buf = np.empty(n, np.float64)
    for c in range(ncomp):
        ub += cvol[c]
        if cnt[c] == 0:
            continue
        nb = 0
        for p in range(n):
            if pcomp[p] == c:
                buf[nb] = gain_pt[p]
                nb += 1
        bs = np.sort(buf[:nb])
        for a in range(min(nb, cap, ell2)):
            pool[npool] = bs[nb - 1 - a]
            npool += 1
    ps = np.sort(pool[:npool])
    for a in range(min(npool, ell2)):
        ub += ps[npool - 1 - a]
    return ub
