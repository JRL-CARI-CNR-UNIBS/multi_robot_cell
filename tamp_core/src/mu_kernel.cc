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
}
