/**
 * mu_kernel.cc -- the robot-vs-robot SIMD collision kernel VAMP does not ship.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * VAMP vectorises robot-vs-SELF and robot-vs-ENVIRONMENT (``fkcc``,
 * ``sphere_environment_in_collision``), but it has no robot-vs-ROBOT routine, and its
 * Python ``fk()`` binding runs the SCALAR path (``sphere_fk<1>``) and then allocates one
 * Python object per sphere. So the ``mu`` computation -- which is *entirely*
 * robot-vs-robot -- had to be hand-written, and until now it was numpy.
 *
 * This is that routine, written on VAMP's own ``FloatVector`` abstraction (hence AVX2,
 * NEON and WASM for free, 8 float32 lanes on this AVX2 host). It is the kernel the
 * VAMP-MR paper (Huang, Gao & Li) describes as ``INTERCC`` inside ``FK_CC_MULTI``.
 *
 * WHAT IT COMPUTES
 * ----------------
 * For one pair of trajectories, the exact set of forbidden relative start offsets
 *
 *     D = { k - l : robot A at sample k collides with robot B at sample l }
 *
 * ``mu`` itself is never formed. An offset ``d`` is forbidden iff AT LEAST ONE pair on
 * the diagonal ``k - l = d`` collides, so each diagonal is abandoned at its first hit.
 * In C++ there is no per-call overhead to amortise, so pairs are tested ONE AT A TIME
 * and the early exit is exact -- the numpy port had to batch them in chunks of 128 and
 * therefore always overshot past the first collision.
 *
 * THE PADDING CONTRACT (why there is no masking anywhere below)
 * ------------------------------------------------------------
 * Every lane the caller does not fill with a real sphere must be parked at radius 0 and
 * a centre far from the workspace -- ``+1e6`` for robot A, ``-1e6`` for robot B (see
 * mu_kernel.py). Padded-vs-real and padded-vs-padded are then separated by ~1e6 m, so
 * they can never register a hit. That single convention removes every lane mask, every
 * tail branch, and the ``-inf`` radius the numpy engine used for a detached object.
 *
 * Sphere counts need no padding at all: 79 robot spheres + 1 object slot = 80, already a
 * multiple of the 8-wide rake. Only the link-group bounds pad (17 -> 24).
 *
 * Loads are UNALIGNED on purpose: the layout is such that every plane would in fact be
 * 32-byte aligned given an aligned base, but numpy makes no alignment guarantee, and on
 * this microarchitecture an unaligned load of aligned data costs nothing.
 *
 * Build: scripts/build_mu_kernel.sh   Call: scripts/mu_kernel.py (ctypes)
 */

#include <vamp/vector.hh>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{
    using V = vamp::FloatVector<vamp::FloatVectorWidth>;
    constexpr int W = static_cast<int>(vamp::FloatVectorWidth);

    /// One sample's spheres in struct-of-arrays form: four contiguous planes.
    struct Planes
    {
        const float *x;
        const float *y;
        const float *z;
        const float *r;
    };

    /// Plane view of sample ``k``. Layout is (K, 4, stride) float32, C-contiguous.
    inline Planes plane_at(const float *base, int k, int stride) noexcept
    {
        const float *p = base + static_cast<std::size_t>(k) * 4 * static_cast<std::size_t>(stride);
        return {p, p + stride, p + 2 * stride, p + 3 * stride};
    }

    /// True iff ANY sphere of A touches ANY sphere of B (``|ca-cb| <= ra+rb``).
    ///
    /// ``na`` walks only the REAL entries of A (scalar broadcast, so the padded tail can
    /// simply be skipped); ``nb`` walks A's counterpart in full vectors and MUST be the
    /// padded count, since the rake reads 8 at a time.
    inline bool sets_touch(const Planes &A, int na, const Planes &B, int nb) noexcept
    {
        for (int i = 0; i < na; ++i)
        {
            const V axi(A.x[i]), ayi(A.y[i]), azi(A.z[i]), ari(A.r[i]);

            for (int j = 0; j < nb; j += W)
            {
                const V bx(B.x + j, false), by(B.y + j, false);
                const V bz(B.z + j, false), br(B.r + j, false);

                const V dx = axi - bx, dy = ayi - by, dz = azi - bz;
                const V d2 = dx * dx + dy * dy + dz * dz;
                const V rs = ari + br;

                if ((d2 <= rs * rs).any())
                {
                    return true;
                }
            }
        }
        return false;
    }

    /// One cell ``mu[k][l]``: per-link broad phase, then the full sphere-set test.
    inline bool cell_hit(const float *a_sph, const float *a_grp, const float *b_sph,
                         const float *b_grp, int k, int l, int n_sph, int n_grp_real,
                         int n_grp_pad) noexcept
    {
        const Planes ag = plane_at(a_grp, k, n_grp_pad);
        const Planes bg = plane_at(b_grp, l, n_grp_pad);
        if (not sets_touch(ag, n_grp_real, bg, n_grp_pad))
        {
            return false;
        }
        const Planes as = plane_at(a_sph, k, n_sph);
        const Planes bs = plane_at(b_sph, l, n_sph);
        return sets_touch(as, n_sph, bs, n_sph);
    }

    /// ``end[k]`` = last sample of the frozen run that STARTS at ``k`` (``k`` itself when no
    /// run of length >= 2 starts there). ``same[k]`` = 1 iff sample k is identical to k - 1.
    inline std::vector<int> run_ends(const unsigned char *same, int K)
    {
        std::vector<int> end(static_cast<std::size_t>(K));
        for (int k = K - 1; k >= 0; --k)
        {
            end[k] = (k + 1 < K and same[k + 1]) ? end[k + 1] : k;
        }
        // end[k] now holds the run's last sample for EVERY member; keep it on starts only.
        for (int k = 0; k < K; ++k)
        {
            if (same[k])
            {
                end[k] = k;
            }
        }
        return end;
    }
}  // namespace

extern "C"
{
    /// SIMD lane count this library was compiled for (8 on AVX2). Lets the Python side
    /// report the rake it actually got rather than assuming one.
    int mu_kernel_simd_width() noexcept
    {
        return W;
    }

    /**
     * The FULL collision matrix ``mu[k][l]`` for one trajectory pair, row-major into
     * ``out`` (sized ``KA * KB`` bytes, 1 = collision).
     *
     * The offset reduction ``D = {k - l}`` that :c:func:`mu_forbidden_offsets` returns is
     * LOSSY -- it keeps the difference and discards which pairs collide. The temporal plan
     * graph needs precisely what it discards, so this entry point keeps the matrix.
     *
     * Consequently there is no diagonal walk and no early exit here: every surviving pair
     * must be evaluated, because every entry is an answer. The per-link broad phase still
     * rejects ~61 % of pairs outright, so this costs a few times a ``mu_forbidden_offsets``
     * call rather than orders of magnitude more.
     */
    void mu_matrix(
        const float *a_sph,
        const float *a_grp,
        int KA,
        const float *b_sph,
        const float *b_grp,
        int KB,
        int n_sph,
        int n_grp_real,
        int n_grp_pad,
        unsigned char *out) noexcept
    {
        for (int k = 0; k < KA; ++k)
        {
            const Planes ag = plane_at(a_grp, k, n_grp_pad);
            const Planes as = plane_at(a_sph, k, n_sph);
            unsigned char *row = out + static_cast<std::size_t>(k) * KB;

            for (int l = 0; l < KB; ++l)
            {
                const Planes bg = plane_at(b_grp, l, n_grp_pad);
                if (not sets_touch(ag, n_grp_real, bg, n_grp_pad))
                {
                    row[l] = 0;
                    continue;
                }
                const Planes bs = plane_at(b_sph, l, n_sph);
                row[l] = sets_touch(as, n_sph, bs, n_sph) ? 1 : 0;
            }
        }
    }

    /**
     * Forbidden offsets for one trajectory pair. Returns how many were written to
     * ``out``, which the caller sizes at ``KA + KB - 1`` (every possible offset).
     * Results are emitted in ascending ``d``, so they arrive already sorted.
     *
     * ``a_sph``/``b_sph``  (K, 4, n_sph)      float32 C-contiguous  -- x,y,z,r planes
     * ``a_grp``/``b_grp``  (K, 4, n_grp_pad)  float32 C-contiguous  -- link-group bounds
     */
    int mu_forbidden_offsets(
        const float *a_sph,
        const float *a_grp,
        int KA,
        const float *b_sph,
        const float *b_grp,
        int KB,
        int n_sph,
        int n_grp_real,
        int n_grp_pad,
        int *out) noexcept
    {
        int count = 0;

        for (int d = -(KB - 1); d < KA; ++d)
        {
            const int k0 = std::max(0, d);
            const int k1 = std::min(KA, KB + d);

            for (int k = k0; k < k1; ++k)
            {
                const int l = k - d;

                // Broad phase: per-link bounding spheres. Cheap enough (17 x 3 vectors)
                // that it pays for itself on the ~60 % of pairs it rejects outright.
                const Planes ag = plane_at(a_grp, k, n_grp_pad);
                const Planes bg = plane_at(b_grp, l, n_grp_pad);
                if (not sets_touch(ag, n_grp_real, bg, n_grp_pad))
                {
                    continue;
                }

                // Narrow phase: the real 80 x 80 sphere check.
                const Planes as = plane_at(a_sph, k, n_sph);
                const Planes bs = plane_at(b_sph, l, n_sph);
                if (sets_touch(as, n_sph, bs, n_sph))
                {
                    out[count++] = d;
                    break;  // this diagonal is settled -- skip the rest of it
                }
            }
        }

        return count;
    }

    /*
     * FROZEN RUNS (B1, 2026-10-11)
     * ----------------------------
     * A cell ``mu[k][l]`` depends on sample k of A and sample l of B and on nothing else. A
     * robot that dwells -- a gripper closing, a hold tail, a process on/off, HOME at either
     * end -- repeats one configuration for many consecutive samples, and the caller marks
     * every sample bit-identical to its predecessor (spheres AND link groups, as packed) in
     * ``a_same`` / ``b_same``. A run of identical samples is then one row (or column) of
     * checks, not one per sample. Both entry points below return EXACTLY what their plain
     * counterparts return; they only skip cells whose value is already known.
     */

    /// :c:func:`mu_matrix`, with each frozen row copied from the previous row and each
    /// frozen column from the previous column instead of being evaluated.
    void mu_matrix_runs(
        const float *a_sph,
        const float *a_grp,
        int KA,
        const float *b_sph,
        const float *b_grp,
        int KB,
        int n_sph,
        int n_grp_real,
        int n_grp_pad,
        const unsigned char *a_same,
        const unsigned char *b_same,
        unsigned char *out) noexcept
    {
        for (int k = 0; k < KA; ++k)
        {
            unsigned char *row = out + static_cast<std::size_t>(k) * KB;
            if (k > 0 and a_same[k])
            {
                std::memcpy(row, row - KB, static_cast<std::size_t>(KB));
                continue;
            }
            for (int l = 0; l < KB; ++l)
            {
                if (l > 0 and b_same[l])
                {
                    row[l] = row[l - 1];
                    continue;
                }
                row[l] = cell_hit(a_sph, a_grp, b_sph, b_grp, k, l, n_sph, n_grp_real,
                                  n_grp_pad) ? 1 : 0;
            }
        }
    }

    /**
     * :c:func:`mu_forbidden_offsets`, frozen runs first.
     *
     * Rows [a, b] of one A run against columns [l, e] (one column, or one B run) are ONE
     * value; when it is a hit, every offset ``[a - e, b - l]`` is forbidden at once. So:
     *   1. each A run is checked against every B column (B runs by their first column);
     *   2. each B run against every A sample outside an A run;
     *   3. the ordinary diagonal walk runs over the remaining diagonals, skipping every cell
     *      in a run (its value was settled in 1 or 2).
     * A check whose whole offset interval is already forbidden is skipped. Every cell's
     * value is established exactly once (or proven irrelevant), so the set is identical to
     * the plain walk's; it is emitted ascending, as there.
     */
    int mu_forbidden_offsets_runs(
        const float *a_sph,
        const float *a_grp,
        int KA,
        const float *b_sph,
        const float *b_grp,
        int KB,
        int n_sph,
        int n_grp_real,
        int n_grp_pad,
        const unsigned char *a_same,
        const unsigned char *b_same,
        int *out) noexcept
    {
        const std::vector<int> a_end = run_ends(a_same, KA);
        const std::vector<int> b_end = run_ends(b_same, KB);
        std::vector<unsigned char> forb(static_cast<std::size_t>(KA + KB - 1), 0);
        const int off = KB - 1;
        auto settled = [&](int lo, int hi) {
            for (int d = lo; d <= hi; ++d)
            {
                if (not forb[d + off])
                {
                    return false;
                }
            }
            return true;
        };
        auto in_run_a = [&](int k) { return a_same[k] or a_end[k] > k; };
        auto in_run_b = [&](int l) { return b_same[l] or b_end[l] > l; };

        // 1. A runs x every B column (a B run represented by its first column).
        for (int a = 0; a < KA; ++a)
        {
            const int b = a_end[a];
            if (b == a)
            {
                continue;
            }
            for (int l = 0; l < KB; ++l)
            {
                if (b_same[l])
                {
                    continue;
                }
                const int lo = a - b_end[l], hi = b - l;
                if (settled(lo, hi))
                {
                    continue;
                }
                if (cell_hit(a_sph, a_grp, b_sph, b_grp, a, l, n_sph, n_grp_real, n_grp_pad))
                {
                    std::fill(forb.begin() + (lo + off), forb.begin() + (hi + off + 1), 1);
                }
            }
        }
        // 2. B runs x every A sample outside an A run.
        for (int c = 0; c < KB; ++c)
        {
            const int e = b_end[c];
            if (e == c)
            {
                continue;
            }
            for (int k = 0; k < KA; ++k)
            {
                if (in_run_a(k))
                {
                    continue;
                }
                const int lo = k - e, hi = k - c;
                if (settled(lo, hi))
                {
                    continue;
                }
                if (cell_hit(a_sph, a_grp, b_sph, b_grp, k, c, n_sph, n_grp_real, n_grp_pad))
                {
                    std::fill(forb.begin() + (lo + off), forb.begin() + (hi + off + 1), 1);
                }
            }
        }
        // 3. The diagonal walk over what is left.
        int count = 0;
        for (int d = -(KB - 1); d < KA; ++d)
        {
            if (forb[d + off])
            {
                out[count++] = d;
                continue;
            }
            const int k0 = std::max(0, d);
            const int k1 = std::min(KA, KB + d);
            for (int k = k0; k < k1; ++k)
            {
                const int l = k - d;
                if (in_run_a(k) or in_run_b(l))
                {
                    continue;
                }
                if (cell_hit(a_sph, a_grp, b_sph, b_grp, k, l, n_sph, n_grp_real, n_grp_pad))
                {
                    out[count++] = d;
                    break;
                }
            }
        }
        return count;
    }
}
