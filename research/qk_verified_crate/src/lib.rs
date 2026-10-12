//! Verified sketches: Q8_0 x Q8_0 and Q4_0 x Q8_0 dot kernels (scalar, AVX2, AVX-VNNI, NEON/dotprod).
use core::mem::size_of;

pub mod vecops;

pub const QK: usize = 32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 { pub d: u16 /* f16 bits */, pub qs: [i8; 32] }
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 { pub d: u16 /* f16 bits */, pub qs: [u8; 16] }
const _: () = assert!(size_of::<BlockQ8_0>() == 34 && size_of::<BlockQ4_0>() == 18);

/// Software f16 -> f32 (reference). Hardware: F16C `_mm_cvtph_ps` / NEON `vcvt_f32_f16`.
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            let mut e2 = 113u32; let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            (s << 31) | (e2 << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 { (s << 31) | (0xff << 23) | (m << 13) } else { (s << 31) | ((e + 112) << 23) | (m << 13) };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------- scalar reference
pub fn dot_q8_0_q8_0_ref(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let mut sumf = 0f32;
    for (a, b) in x.iter().zip(y) {
        let mut s = 0i32;
        for j in 0..32 { s += a.qs[j] as i32 * b.qs[j] as i32; }
        sumf += s as f32 * (f16_to_f32(a.d) * f16_to_f32(b.d));
    }
    sumf
}
pub fn dot_q4_0_q8_0_ref(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let mut sumf = 0f32;
    for (a, b) in x.iter().zip(y) {
        let mut s = 0i32;
        for j in 0..16 {
            let lo = (a.qs[j] & 0x0F) as i32 - 8;
            let hi = (a.qs[j] >> 4) as i32 - 8;
            s += lo * b.qs[j] as i32 + hi * b.qs[j + 16] as i32;
        }
        sumf += s as f32 * (f16_to_f32(a.d) * f16_to_f32(b.d));
    }
    sumf
}
/// Activation quantizer (scalar, ties-away like ggml's reference).
pub fn quantize_row_q8_0(x: &[f32], out: &mut [BlockQ8_0]) {
    for (blk, xs) in out.iter_mut().zip(x.chunks_exact(32)) {
        let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        blk.d = f32_to_f16(d);
        for j in 0..32 { blk.qs[j] = (xs[j] * id).round() as i8; }
    }
}
pub fn f32_to_f16(f: f32) -> u16 {
    // simple round-to-nearest-even for normal range; good enough for tests
    let b = f.to_bits(); let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32 - 127 + 15; let m = b & 0x7fffff;
    if f == 0.0 { return s; }
    if e <= 0 { return s; } // flush subnormals to zero (tests only)
    if e >= 31 { return s | 0x7c00; }
    let mut h = ((e as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) { h += 1; }
    s | h as u16
}


// ---------------------------------------------------------------- K-quants (256-wide super-blocks)
pub const QK_K: usize = 256;
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4K { pub d: u16, pub dmin: u16, pub scales: [u8; 12], pub qs: [u8; 128] }   // 144 B = 4.5 bpw
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ6K { pub ql: [u8; 128], pub qh: [u8; 64], pub scales: [i8; 16], pub d: u16 } // 210 B = 6.5625 bpw
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8K { pub d: f32, pub qs: [i8; 256], pub bsums: [i16; 16] }                   // 292 B
const _: () = assert!(size_of::<BlockQ4K>() == 144 && size_of::<BlockQ6K>() == 210 && size_of::<BlockQ8K>() == 292);

/// ggml's 6-bit scale/min unpack (same bit tricks as the C `utmp`/kmask code). Returns (scales[8], mins[8]).
pub fn unpack_q4k_scales(s: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    const K1: u32 = 0x3f3f3f3f; const K2: u32 = 0x0f0f0f0f; const K3: u32 = 0x03030303;
    let mut u = [u32::from_le_bytes([s[0], s[1], s[2], s[3]]), u32::from_le_bytes([s[4], s[5], s[6], s[7]]),
                 u32::from_le_bytes([s[8], s[9], s[10], s[11]]), 0];
    u[3] = ((u[2] >> 4) & K2) | (((u[1] >> 6) & K3) << 4);
    let uaux = u[1] & K1;
    u[1] = (u[2] & K2) | (((u[0] >> 6) & K3) << 4);
    u[2] = uaux;
    u[0] &= K1;
    let mut sc = [0u8; 8]; let mut mn = [0u8; 8];
    sc[..4].copy_from_slice(&u[0].to_le_bytes()); sc[4..].copy_from_slice(&u[1].to_le_bytes());
    mn[..4].copy_from_slice(&u[2].to_le_bytes()); mn[4..].copy_from_slice(&u[3].to_le_bytes());
    (sc, mn)
}
pub fn dot_q4k_q8k_ref(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (a, b) in x.iter().zip(y) {
        let (sc, mn) = unpack_q4k_scales(&a.scales);
        let mut sumi = 0i32;
        for sb in 0..8 {                       // 8 sub-blocks of 32 elements
            let (grp, hi) = (sb / 2, sb % 2);  // 64-element groups: low nibbles = even sub-block, high nibbles = odd
            let mut t = 0i32;
            for l in 0..32 {
                let q = a.qs[grp * 32 + l];
                let v = if hi == 0 { q & 0xF } else { q >> 4 };
                t += v as i32 * b.qs[sb * 32 + l] as i32;
            }
            sumi += t * sc[sb] as i32;
        }
        let mut summ = 0i32;
        for j in 0..16 { summ += b.bsums[j] as i32 * mn[j / 2] as i32; }
        sumf += f16_to_f32(a.d) * b.d * sumi as f32 - f16_to_f32(a.dmin) * b.d * summ as f32;
    }
    sumf
}
/// Q6_K element order (ggml): per 128-element half, 4 x 32-element quarters; quarter k uses ql nibble (k%2 -> ql[32*(k%2)..], low for k<2,
/// high for k>=2) and 2 bits of qh[l] at bit offset 2*k.  Values are unsigned 0..63; stored value is q-32.
pub fn unpack_q6k(a: &BlockQ6K) -> [i8; 256] {
    let mut aux = [0i8; 256];
    for j in 0..2 {
        let (ql, qh) = (&a.ql[j * 64..], &a.qh[j * 32..]);
        for l in 0..32 {
            aux[j*128 + l]      = (((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i8) - 32;
            aux[j*128 + l + 32] = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8) - 32;
            aux[j*128 + l + 64] = (((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) as i8) - 32;
            aux[j*128 + l + 96] = (((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) as i8) - 32;
        }
    }
    aux
}
pub fn dot_q6k_q8k_ref(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
    let mut sumf = 0f32;
    for (a, b) in x.iter().zip(y) {
        let aux = unpack_q6k(a);
        let mut sumi = 0i32;
        for sb in 0..16 {                       // 16 sub-blocks of 16 elements, int8 scales
            let mut t = 0i32;
            for l in 0..16 { t += aux[sb * 16 + l] as i32 * b.qs[sb * 16 + l] as i32; }
            sumi += t * a.scales[sb] as i32;
        }
        sumf += f16_to_f32(a.d) * b.d * sumi as f32;
    }
    sumf
}

// ---------------------------------------------------------------- x86-64
#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::*;
    use core::arch::x86_64::*;

    /// hardware f16 -> f32 (F16C, every AVX2-capable x86 core has it).
    /// PITFALL (measured, see notes): written with `_mm_cvtsi32_si128(h as i32)` LLVM emits `vpinsrw $0, mem, %xmm_acc, %xmm`
    /// and picks the f32 *accumulator* register as the (don't-care) merge source, which turns a block-scale conversion into
    /// part of the loop-carried FMA dependency chain: single-row Q4_0 ran at 3.6 GB/s instead of ~9 GB/s (same loop in C: ~9).
    /// `vmovd r32, xmm` zero-extends and has no merge dependency; inline asm is the only way I found to force it.
    /// Miri cannot execute `asm!`, so under Miri the intrinsic version is used.
    #[inline]
    #[target_feature(enable = "f16c")]
    pub fn h2f(h: u16) -> f32 {
        #[cfg(not(miri))]
        {
            let r: f32;
            // SAFETY: register-only (no memory, no stack); needs F16C, which this fn's target_feature guarantees.
            unsafe {
                core::arch::asm!("vmovd {t}, {h:e}", "vcvtph2ps {t}, {t}",
                    h = in(reg) h as u32, t = out(xmm_reg) r, options(pure, nomem, nostack));
            }
            r
        }
        #[cfg(miri)]
        { _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h as i32))) }
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn hsum_ps(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }
    /// signed x signed i8 -> 8 x i32 partial sums (llama.cpp: sign trick + maddubs + madd(1))
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn dot_i8_avx2(x: __m256i, y: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(x, x);       // |x| as unsigned
        let sy = _mm256_sign_epi8(y, x);       // y * sign(x)
        let p16 = _mm256_maddubs_epi16(ax, sy); // u8*s8 pair sums -> i16 (saturating!)
        _mm256_madd_epi16(_mm256_set1_epi16(1), p16)
    }
    /// same with AVX-VNNI (no i16 intermediate, no saturation)
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    pub fn dot_i8_avxvnni(x: __m256i, y: __m256i) -> __m256i {
        let ax = _mm256_sign_epi8(x, x);
        let sy = _mm256_sign_epi8(y, x);
        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ax, sy)
    }
    /// unsigned (0..15 nibbles) x signed: no sign trick needed
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn dot_u8s8_avx2(ux: __m256i, y: __m256i) -> __m256i {
        _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(ux, y))
    }
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    pub fn dot_u8s8_avxvnni(ux: __m256i, y: __m256i) -> __m256i {
        _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ux, y)
    }
    /// 32 nibbles -> 32 bytes in [0,15]: low nibbles first (bytes 0..15), then high nibbles (16..31)
    #[inline]
    #[target_feature(enable = "avx2")]
    pub fn nibbles_to_bytes(q: &[u8; 16]) -> __m256i {
        // SAFETY: q is exactly 16 readable bytes; loadu has no alignment requirement.
        let t = unsafe { _mm_loadu_si128(q.as_ptr().cast()) };
        let m = _mm_set1_epi8(0x0F);
        let lo = _mm_and_si128(t, m);
        let hi = _mm_and_si128(_mm_srli_epi16::<4>(t), m);
        _mm256_set_m128i(hi, lo)
    }
    #[inline]
    #[target_feature(enable = "avx2")]
    fn load32(q: &[i8; 32]) -> __m256i {
        // SAFETY: 32 readable bytes, unaligned load.
        unsafe { _mm256_loadu_si256(q.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn dot_q8_0_q8_0_avx2(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
        assert_eq!(x.len(), y.len());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 2 <= x.len() {
            let d0 = _mm256_set1_ps(h2f(x[i].d) * h2f(y[i].d));
            let d1 = _mm256_set1_ps(h2f(x[i + 1].d) * h2f(y[i + 1].d));
            let p0 = dot_i8_avx2(load32(&x[i].qs), load32(&y[i].qs));
            let p1 = dot_i8_avx2(load32(&x[i + 1].qs), load32(&y[i + 1].qs));
            acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p0), acc0);
            acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(p1), acc1);
            i += 2;
        }
        if i < x.len() {
            let d0 = _mm256_set1_ps(h2f(x[i].d) * h2f(y[i].d));
            let p0 = dot_i8_avx2(load32(&x[i].qs), load32(&y[i].qs));
            acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p0), acc0);
        }
        hsum_ps(_mm256_add_ps(acc0, acc1))
    }

    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub fn dot_q8_0_q8_0_avxvnni(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
        assert_eq!(x.len(), y.len());
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut i = 0;
        while i + 2 <= x.len() {
            let d0 = _mm256_set1_ps(h2f(x[i].d) * h2f(y[i].d));
            let d1 = _mm256_set1_ps(h2f(x[i + 1].d) * h2f(y[i + 1].d));
            let p0 = dot_i8_avxvnni(load32(&x[i].qs), load32(&y[i].qs));
            let p1 = dot_i8_avxvnni(load32(&x[i + 1].qs), load32(&y[i + 1].qs));
            acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p0), acc0);
            acc1 = _mm256_fmadd_ps(d1, _mm256_cvtepi32_ps(p1), acc1);
            i += 2;
        }
        if i < x.len() {
            let d0 = _mm256_set1_ps(h2f(x[i].d) * h2f(y[i].d));
            let p0 = dot_i8_avxvnni(load32(&x[i].qs), load32(&y[i].qs));
            acc0 = _mm256_fmadd_ps(d0, _mm256_cvtepi32_ps(p0), acc0);
        }
        hsum_ps(_mm256_add_ps(acc0, acc1))
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn dot_q4_0_q8_0_avx2(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        assert_eq!(x.len(), y.len());
        let mut acc = _mm256_setzero_ps();
        let off = _mm256_set1_epi8(8);
        for (a, b) in x.iter().zip(y) {
            let d = _mm256_set1_ps(h2f(a.d) * h2f(b.d));
            let qx = _mm256_sub_epi8(nibbles_to_bytes(&a.qs), off); // [-8,7]
            let p = dot_i8_avx2(qx, load32(&b.qs));
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p), acc);
        }
        hsum_ps(acc)
    }

    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub fn dot_q4_0_q8_0_avxvnni(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        assert_eq!(x.len(), y.len());
        let mut acc = _mm256_setzero_ps();
        let off = _mm256_set1_epi8(8);
        for (a, b) in x.iter().zip(y) {
            let d = _mm256_set1_ps(h2f(a.d) * h2f(b.d));
            let qx = _mm256_sub_epi8(nibbles_to_bytes(&a.qs), off);
            let p = dot_i8_avxvnni(qx, load32(&b.qs));
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p), acc);
        }
        hsum_ps(acc)
    }

    /// Variant: activation block carries s = sum(qs) (like ggml Q8_1).  dot = sum(q*y) - 8*sum(y); no sign trick.
    #[derive(Clone, Copy)]
    pub struct BlockQ8S { pub d: f32, pub s: i32, pub qs: [i8; 32] }
    pub fn quantize_with_sum(x: &[f32], out: &mut [BlockQ8S]) {
        for (blk, xs) in out.iter_mut().zip(x.chunks_exact(32)) {
            let amax = xs.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = f16_to_f32(f32_to_f16(amax / 127.0)); // match f16 rounding of scale
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            blk.d = d; let mut s = 0i32;
            for j in 0..32 { let q = (xs[j] * id).round() as i8; blk.qs[j] = q; s += q as i32; }
            blk.s = s;
        }
    }
    #[target_feature(enable = "avx2,fma,f16c,avxvnni")]
    pub fn dot_q4_0_q8s_avxvnni(x: &[BlockQ4_0], y: &[BlockQ8S]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        for (a, b) in x.iter().zip(y) {
            let d = _mm256_set1_ps(h2f(a.d) * b.d);
            // SAFETY: 32 readable bytes
            let qy = unsafe { _mm256_loadu_si256(b.qs.as_ptr().cast()) };
            let p = dot_u8s8_avxvnni(nibbles_to_bytes(&a.qs), qy); // sum q*y, q in 0..15
            let p = _mm256_cvtepi32_ps(p);
            // fold the -8*s correction in as a scalar-per-block float; cheap relative to the vector work
            acc = _mm256_fmadd_ps(d, p, acc);
            acc = _mm256_sub_ps(acc, _mm256_set_ps(0.,0.,0.,0.,0.,0.,0., h2f(a.d) * b.d * 8.0 * b.s as f32));
        }
        hsum_ps(acc)
    }

    /// 4 weight rows x 1 activation vector: activation loaded once, 4 independent accumulators (ILP).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn dot4_q8_0_q8_0_avx2(w: [&[BlockQ8_0]; 4], y: &[BlockQ8_0]) -> [f32; 4] {
        let n = y.len();
        // re-slice once so LLVM knows every row has exactly n blocks (removes 4 bounds checks per iteration)
        let w = [&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]];
        let mut acc = [_mm256_setzero_ps(); 4];
        for i in 0..n {
            let qy = load32(&y[i].qs);
            let dy = h2f(y[i].d);
            for r in 0..4 {
                let p = dot_i8_avx2(load32(&w[r][i].qs), qy);
                let d = _mm256_set1_ps(h2f(w[r][i].d) * dy);
                acc[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p), acc[r]);
            }
        }
        [hsum_ps(acc[0]), hsum_ps(acc[1]), hsum_ps(acc[2]), hsum_ps(acc[3])]
    }


    /// Q4_0: 4 weight rows x 1 activation vector (reuses the activation load and scale across 4 rows; measured gain over the
    /// 1-row kernel was only ~1.2x in L2 on the test VM, see notes section 4).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn dot4_q4_0_q8_0_avx2(w: [&[BlockQ4_0]; 4], y: &[BlockQ8_0]) -> [f32; 4] {
        let n = y.len();
        // re-slice once so LLVM knows every row has exactly n blocks (removes 4 bounds checks per iteration)
        let w = [&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]];
        let mut acc = [_mm256_setzero_ps(); 4];
        let off = _mm256_set1_epi8(8);
        for i in 0..n {
            let qy = load32(&y[i].qs);
            let dy = h2f(y[i].d);
            for r in 0..4 {
                let qx = _mm256_sub_epi8(nibbles_to_bytes(&w[r][i].qs), off);
                let p = dot_i8_avx2(qx, qy);
                let d = _mm256_set1_ps(h2f(w[r][i].d) * dy);
                acc[r] = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(p), acc[r]);
            }
        }
        [hsum_ps(acc[0]), hsum_ps(acc[1]), hsum_ps(acc[2]), hsum_ps(acc[3])]
    }
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn gemv4_q4_0_avx2(w: &[BlockQ4_0], nb: usize, y: &[BlockQ8_0], out: &mut [f32]) {
        for (r, o) in out.chunks_exact_mut(4).enumerate() {
            let b = r * 4 * nb;
            let res = dot4_q4_0_q8_0_avx2([&w[b..b + nb], &w[b + nb..b + 2 * nb], &w[b + 2 * nb..b + 3 * nb], &w[b + 3 * nb..b + 4 * nb]], y);
            o.copy_from_slice(&res);
        }
    }


    #[inline]
    #[target_feature(enable = "avx2")]
    fn load32u(q: &[u8; 32]) -> __m256i {
        // SAFETY: &[u8;32] guarantees 32 readable bytes; unaligned load.
        unsafe { _mm256_loadu_si256(q.as_ptr().cast()) }
    }

    /// Q4_K x Q8_K (llama.cpp AVX2 algorithm).  q4 nibbles are unsigned (0..15) so no sign trick:
    /// maddubs(q4, q8) -> i16 pair sums (<= 15*127*2), then madd_epi16(scale_i16, p16) fuses the pairwise add and the
    /// per-sub-block scale multiply.  The "mins" term is dmin * sum_j(min_j * bsum_j) and is accumulated separately.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn dot_q4k_q8k_avx2(x: &[BlockQ4K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());
        const K1: u32 = 0x3f3f3f3f; const K2: u32 = 0x0f0f0f0f; const K3: u32 = 0x03030303;
        let m4 = _mm256_set1_epi8(0xF);
        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();
        for (a, b) in x.iter().zip(y) {
            let d = b.d * h2f(a.d);
            let dmin = -b.d * h2f(a.dmin);
            // 12 packed 6-bit scales/mins -> 8 scale bytes + 8 min bytes (same bit-twiddling as the scalar reference)
            let s = &a.scales;
            let mut u = [u32::from_le_bytes([s[0], s[1], s[2], s[3]]), u32::from_le_bytes([s[4], s[5], s[6], s[7]]),
                         u32::from_le_bytes([s[8], s[9], s[10], s[11]]), 0];
            u[3] = ((u[2] >> 4) & K2) | (((u[1] >> 6) & K3) << 4);
            let uaux = u[1] & K1;
            u[1] = (u[2] & K2) | (((u[0] >> 6) & K3) << 4);
            u[2] = uaux;
            u[0] &= K1;
            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(u[3] as i32, u[2] as i32, u[1] as i32, u[0] as i32));
            // mins part: pair-sum bsums (16 x i16 -> 8 x i16, one per 32-block) then madd with the 8 mins
            // SAFETY: bsums is [i16;16] = 32 bytes.
            let q8sums = unsafe { _mm256_loadu_si256(b.bsums.as_ptr().cast()) };
            let q8s = _mm_hadd_epi16(_mm256_extracti128_si256::<0>(q8sums), _mm256_extracti128_si256::<1>(q8sums));
            let prod = _mm_madd_epi16(_mm256_extracti128_si256::<1>(mins_and_scales), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);
            let sc128 = _mm256_extracti128_si256::<0>(mins_and_scales);
            let scales = _mm256_set_m128i(sc128, sc128);
            let (q4, q8) = (a.qs.as_chunks::<32>().0, b.qs.as_chunks::<32>().0);
            let mut sumi = _mm256_setzero_si256();
            for j in 0..4 {
                // broadcast i16 scale (2j) / (2j+1) to every lane: shuffle bytes (2k, 2k+1) repeated
                let sh_l = _mm256_set1_epi16((((2 * (2 * j) + 1) << 8) | (2 * (2 * j))) as i16);
                let sh_h = _mm256_set1_epi16((((2 * (2 * j + 1) + 1) << 8) | (2 * (2 * j + 1))) as i16);
                let scale_l = _mm256_shuffle_epi8(scales, sh_l);
                let scale_h = _mm256_shuffle_epi8(scales, sh_h);
                let q4bits = load32u(&q4[j]);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16::<4>(q4bits), m4);
                let p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, load32(&q8[2 * j])));
                let p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, load32(&q8[2 * j + 1])));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
            }
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }
        acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
        hsum_ps(acc) + _mm_cvtss_f32(acc_m)
    }

    /// Q6_K x Q8_K.  6-bit value = (ql nibble) | (2 high bits from qh) << 4, stored UNSIGNED (0..63); the -32 offset is
    /// removed algebraically: sum(scale*(q-32)*y) = sum(scale*q*y) - 32*sum(scale_sb * bsum_sb).
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn dot_q6k_q8k_avx2(x: &[BlockQ6K], y: &[BlockQ8K]) -> f32 {
        assert_eq!(x.len(), y.len());
        let m3 = _mm256_set1_epi8(3);
        let m15 = _mm256_set1_epi8(15);
        let mut acc = _mm256_setzero_ps();
        for (a, b) in x.iter().zip(y) {
            let d = b.d * h2f(a.d);
            // SAFETY: bsums = 32 bytes, scales = 16 bytes.
            let q8sums = unsafe { _mm256_loadu_si256(b.bsums.as_ptr().cast()) };
            let scales = unsafe { _mm_loadu_si128(a.scales.as_ptr().cast()) };
            let scales_16 = _mm256_cvtepi8_epi16(scales);
            let q8sclsub = _mm256_slli_epi32::<5>(_mm256_madd_epi16(q8sums, scales_16));
            let (ql, qh, q8) = (a.ql.as_chunks::<32>().0, a.qh.as_chunks::<32>().0, b.qs.as_chunks::<32>().0);
            let mut sumi = _mm256_setzero_si256();
            for j in 0..2 {
                let q4bits1 = load32u(&ql[2 * j]);
                let q4bits2 = load32u(&ql[2 * j + 1]);
                let q4bits_h = load32u(&qh[j]);
                let q4h_0 = _mm256_slli_epi16::<4>(_mm256_and_si256(q4bits_h, m3));
                let q4h_1 = _mm256_slli_epi16::<2>(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(12)));
                let q4h_2 = _mm256_and_si256(q4bits_h, _mm256_set1_epi8(48));
                let q4h_3 = _mm256_srli_epi16::<2>(_mm256_and_si256(q4bits_h, _mm256_set1_epi8(-64)));
                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m15), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m15), q4h_1);
                let q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(q4bits1), m15), q4h_2);
                let q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16::<4>(q4bits2), m15), q4h_3);
                let mut p = [
                    _mm256_maddubs_epi16(q4_0, load32(&q8[4 * j])),
                    _mm256_maddubs_epi16(q4_1, load32(&q8[4 * j + 1])),
                    _mm256_maddubs_epi16(q4_2, load32(&q8[4 * j + 2])),
                    _mm256_maddubs_epi16(q4_3, load32(&q8[4 * j + 3])),
                ];
                for k in 0..4 {
                    let i = 4 * j + k; // row i of ggml's get_scale_shuffle table: 8x scale[2i] then 8x scale[2i+1]
                    let lo = 0x0101_0101_0101_0101u64.wrapping_mul((2 * i) as u64) as i64;
                    let hi = 0x0101_0101_0101_0101u64.wrapping_mul((2 * i + 1) as u64) as i64;
                    let sc = _mm_shuffle_epi8(scales, _mm_set_epi64x(hi, lo));
                    p[k] = _mm256_madd_epi16(_mm256_cvtepi8_epi16(sc), p[k]);
                }
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[0], p[1]));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p[2], p[3]));
            }
            sumi = _mm256_sub_epi32(sumi, q8sclsub);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_ps(acc)
    }

    /// whole-GEMV inside one target_feature function so the dot kernels inline (no per-row call)
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn gemv_q8_0_avx2(w: &[BlockQ8_0], nb: usize, y: &[BlockQ8_0], out: &mut [f32]) {
        for (r, o) in out.iter_mut().enumerate() {
            *o = dot_q8_0_q8_0_avx2(&w[r * nb..(r + 1) * nb], y);
        }
    }
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn gemv4_q8_0_avx2(w: &[BlockQ8_0], nb: usize, y: &[BlockQ8_0], out: &mut [f32]) {
        for (r, o) in out.chunks_exact_mut(4).enumerate() {
            let b = r * 4 * nb;
            let res = dot4_q8_0_q8_0_avx2([&w[b..b + nb], &w[b + nb..b + 2 * nb], &w[b + 2 * nb..b + 3 * nb], &w[b + 3 * nb..b + 4 * nb]], y);
            o.copy_from_slice(&res);
        }
    }
}

// ---------------------------------------------------------------- aarch64
#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::*;
    use core::arch::aarch64::*;

    /// f16 bits -> f32 with the hardware FCVT (via `vcvt_f32_f16`, stable since 1.94; plain NEON, no `fp16` feature needed).
    /// LLVM folds the lane extract into a single `fcvt s0, h0` (checked on aarch64-apple-darwin and aarch64-unknown-linux-gnu).
    #[inline]
    #[target_feature(enable = "neon")]
    fn h2f(d: u16) -> f32 {
        vget_lane_f32::<0>(vget_low_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(d)))))
    }

    /// Q8_0 x Q8_0, SDOT. Caller must ensure dotprod (default on aarch64-apple-darwin; detect elsewhere).
    #[target_feature(enable = "neon,dotprod")]
    pub fn dot_q8_0_q8_0(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
        assert_eq!(x.len(), y.len());
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let z = vdupq_n_s32(0);
        let n = x.len();
        let mut i = 0;
        while i + 2 <= n {
            // SAFETY: i+1 < n, each qs is [i8;32] (32 readable bytes); vld1q_s8 needs only 1-byte alignment.
            let (xa0, xa1, ya0, ya1, xb0, xb1, yb0, yb1) = unsafe {
                (vld1q_s8(x[i].qs.as_ptr()), vld1q_s8(x[i].qs.as_ptr().add(16)),
                 vld1q_s8(y[i].qs.as_ptr()), vld1q_s8(y[i].qs.as_ptr().add(16)),
                 vld1q_s8(x[i + 1].qs.as_ptr()), vld1q_s8(x[i + 1].qs.as_ptr().add(16)),
                 vld1q_s8(y[i + 1].qs.as_ptr()), vld1q_s8(y[i + 1].qs.as_ptr().add(16)))
            };
            let pa = vdotq_s32(vdotq_s32(z, xa0, ya0), xa1, ya1);
            let pb = vdotq_s32(vdotq_s32(z, xb0, yb0), xb1, yb1);
            s0 = vfmaq_n_f32(s0, vcvtq_f32_s32(pa), h2f(x[i].d) * h2f(y[i].d));
            s1 = vfmaq_n_f32(s1, vcvtq_f32_s32(pb), h2f(x[i + 1].d) * h2f(y[i + 1].d));
            i += 2;
        }
        let mut sum = vaddvq_f32(vaddq_f32(s0, s1));
        while i < n { // scalar tail
            let mut t = 0i32;
            for j in 0..32 { t += x[i].qs[j] as i32 * y[i].qs[j] as i32; }
            sum += t as f32 * (h2f(x[i].d) * h2f(y[i].d));
            i += 1;
        }
        sum
    }

    /// Q4_0 x Q8_0, SDOT.
    #[target_feature(enable = "neon,dotprod")]
    pub fn dot_q4_0_q8_0(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
        assert_eq!(x.len(), y.len());
        let m4 = vdupq_n_u8(0x0F);
        let s8 = vdupq_n_s8(8);
        let z = vdupq_n_s32(0);
        let mut s0 = vdupq_n_f32(0.0);
        let mut s1 = vdupq_n_f32(0.0);
        let n = x.len();
        let mut i = 0;
        while i + 2 <= n {
            // SAFETY: i+1 < n; qs arrays have 16 / 32 readable bytes.
            let (v0, v1, ya0, ya1, yb0, yb1) = unsafe {
                (vld1q_u8(x[i].qs.as_ptr()), vld1q_u8(x[i + 1].qs.as_ptr()),
                 vld1q_s8(y[i].qs.as_ptr()), vld1q_s8(y[i].qs.as_ptr().add(16)),
                 vld1q_s8(y[i + 1].qs.as_ptr()), vld1q_s8(y[i + 1].qs.as_ptr().add(16)))
            };
            // low nibbles are elements 0..15, high nibbles elements 16..31 (ggml Q4_0 layout)
            let xl0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
            let xh0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0)), s8);
            let xl1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4)), s8);
            let xh1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1)), s8);
            let p0 = vdotq_s32(vdotq_s32(z, xl0, ya0), xh0, ya1);
            let p1 = vdotq_s32(vdotq_s32(z, xl1, yb0), xh1, yb1);
            s0 = vfmaq_n_f32(s0, vcvtq_f32_s32(p0), h2f(x[i].d) * h2f(y[i].d));
            s1 = vfmaq_n_f32(s1, vcvtq_f32_s32(p1), h2f(x[i + 1].d) * h2f(y[i + 1].d));
            i += 2;
        }
        let mut sum = vaddvq_f32(vaddq_f32(s0, s1));
        while i < n {
            let mut t = 0i32;
            for j in 0..16 {
                t += ((x[i].qs[j] & 0xF) as i32 - 8) * y[i].qs[j] as i32
                   + ((x[i].qs[j] >> 4) as i32 - 8) * y[i].qs[j + 16] as i32;
            }
            sum += t as f32 * (h2f(x[i].d) * h2f(y[i].d));
            i += 1;
        }
        sum
    }

    /// Q8_0: 4 weight rows x 1 activation vector. The activation block is loaded once and 4 independent
    /// f32 accumulators hide the FMA latency (cross-compiled only, NOT executed: no ARM hardware in this session).
    #[target_feature(enable = "neon,dotprod")]
    pub fn dot4_q8_0_q8_0(w: [&[BlockQ8_0]; 4], y: &[BlockQ8_0]) -> [f32; 4] {
        let n = y.len();
        let w = [&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]]; // one check here instead of 4 per iteration
        let z = vdupq_n_s32(0);
        let mut acc = [vdupq_n_f32(0.0); 4];
        for i in 0..n {
            // SAFETY: each qs is [i8;32]; vld1q_s8 needs only byte alignment.
            let (ya, yb) = unsafe { (vld1q_s8(y[i].qs.as_ptr()), vld1q_s8(y[i].qs.as_ptr().add(16))) };
            let dy = h2f(y[i].d);
            for r in 0..4 {
                let b = &w[r][i];
                let (xa, xb) = unsafe { (vld1q_s8(b.qs.as_ptr()), vld1q_s8(b.qs.as_ptr().add(16))) };
                let p = vdotq_s32(vdotq_s32(z, xa, ya), xb, yb);
                acc[r] = vfmaq_n_f32(acc[r], vcvtq_f32_s32(p), h2f(b.d) * dy);
            }
        }
        [vaddvq_f32(acc[0]), vaddvq_f32(acc[1]), vaddvq_f32(acc[2]), vaddvq_f32(acc[3])]
    }

    /// Q4_0: 4 weight rows x 1 activation vector (cross-compiled only, NOT executed).
    #[target_feature(enable = "neon,dotprod")]
    pub fn dot4_q4_0_q8_0(w: [&[BlockQ4_0]; 4], y: &[BlockQ8_0]) -> [f32; 4] {
        let n = y.len();
        let w = [&w[0][..n], &w[1][..n], &w[2][..n], &w[3][..n]]; // one check here instead of 4 per iteration
        let m4 = vdupq_n_u8(0x0F);
        let s8 = vdupq_n_s8(8);
        let z = vdupq_n_s32(0);
        let mut acc = [vdupq_n_f32(0.0); 4];
        for i in 0..n {
            // SAFETY: qs arrays have 32 / 16 readable bytes.
            let (ya, yb) = unsafe { (vld1q_s8(y[i].qs.as_ptr()), vld1q_s8(y[i].qs.as_ptr().add(16))) };
            let dy = h2f(y[i].d);
            for r in 0..4 {
                let b = &w[r][i];
                let v = unsafe { vld1q_u8(b.qs.as_ptr()) };
                let xl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, m4)), s8);
                let xh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(v)), s8);
                let p = vdotq_s32(vdotq_s32(z, xl, ya), xh, yb);
                acc[r] = vfmaq_n_f32(acc[r], vcvtq_f32_s32(p), h2f(b.d) * dy);
            }
        }
        [vaddvq_f32(acc[0]), vaddvq_f32(acc[1]), vaddvq_f32(acc[2]), vaddvq_f32(acc[3])]
    }

    /// i8mm 2x2 output tile for Q8_0: two weight rows (x0, x1) against two activation vectors (y0, y1), the layout used
    /// by llama.cpp's nrc==2 kernels. zip1/zip2 of the 64-bit halves build [row0 k0..7 | row1 k0..7] and [row0 k8..15 | row1 k8..15]
    /// operands; four SMMLA cover k = 0..31. Output order: [x0.y0, x0.y1, x1.y0, x1.y1].
    /// Cross-compiled only (NOT executed). Needs FEAT_I8MM: M2 and later on Apple, not M1.
    #[target_feature(enable = "neon,i8mm")]
    pub fn tile2x2_q8_0(x0: &[BlockQ8_0], x1: &[BlockQ8_0], y0: &[BlockQ8_0], y1: &[BlockQ8_0]) -> [f32; 4] {
        let n = x0.len();
        let (x1, y0, y1) = (&x1[..n], &y0[..n], &y1[..n]);
        let mut acc = vdupq_n_f32(0.0);
        let z = vdupq_n_s32(0);
        let zip = |a: int8x16_t, b: int8x16_t| -> (int8x16_t, int8x16_t) {
            let (a, b) = (vreinterpretq_s64_s8(a), vreinterpretq_s64_s8(b));
            (vreinterpretq_s8_s64(vzip1q_s64(a, b)), vreinterpretq_s8_s64(vzip2q_s64(a, b)))
        };
        for i in 0..n {
            // SAFETY: each qs is [i8;32]; loads are byte-aligned.
            let ld = |b: &BlockQ8_0| unsafe { (vld1q_s8(b.qs.as_ptr()), vld1q_s8(b.qs.as_ptr().add(16))) };
            let ((a0, a1), (b0, b1)) = (ld(&x0[i]), ld(&x1[i]));
            let ((c0, c1), (d0, d1)) = (ld(&y0[i]), ld(&y1[i]));
            let (l0, l1) = zip(a0, b0); // rows x0,x1 over k=0..7 / k=8..15
            let (l2, l3) = zip(a1, b1); // k=16..23 / k=24..31
            let (r0, r1) = zip(c0, d0); // cols y0,y1
            let (r2, r3) = zip(c1, d1);
            let p = smmla(smmla(smmla(smmla(z, l0, r0), l1, r1), l2, r2), l3, r3);
            let (dx0, dx1, dy0, dy1) = (h2f(x0[i].d), h2f(x1[i].d), h2f(y0[i].d), h2f(y1[i].d));
            let scale = [dx0 * dy0, dx0 * dy1, dx1 * dy0, dx1 * dy1];
            // SAFETY: scale is 4 f32.
            acc = vfmaq_f32(acc, vcvtq_f32_s32(p), unsafe { vld1q_f32(scale.as_ptr()) });
        }
        let mut out = [0f32; 4];
        // SAFETY: out is 4 f32.
        unsafe { vst1q_f32(out.as_mut_ptr(), acc) };
        out
    }

    /// i8mm 2x2 tile via inline asm: stable-Rust workaround because `vmmlaq_s32` is still nightly-only.
    /// Computes acc[2x2] += A[2x8] * B[8x2]; A rows = bytes 0..7 / 8..15 of `a`; B columns = bytes 0..7 / 8..15 of `b`.
    #[inline]
    #[target_feature(enable = "neon,i8mm")]
    pub fn smmla(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut out = acc;
        // SAFETY: pure register arithmetic; requires FEAT_I8MM, guaranteed by target_feature + caller's runtime check.
        unsafe {
            core::arch::asm!("smmla {o:v}.4s, {a:v}.16b, {b:v}.16b",
                o = inout(vreg) out, a = in(vreg) a, b = in(vreg) b,
                options(pure, nomem, nostack));
        }
        out
    }

    /// L1 prefetch hint. `core::arch::aarch64::_prefetch` is unstable (`stdarch_aarch64_prefetch`, #117217), so use asm.
    /// Not benchmarked (no ARM hardware); compiles to a single `prfm pldl1keep` on aarch64-apple-darwin.
    #[inline(always)]
    pub fn prefetch_l1(p: *const u8) {
        // SAFETY: PRFM is a hint; it never faults, even for invalid addresses.
        unsafe { core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p, options(nostack, readonly, preserves_flags)) };
    }
}

// ---------------------------------------------------------------- dispatch
pub type DotQ8 = fn(&[BlockQ8_0], &[BlockQ8_0]) -> f32;
pub type DotQ4 = fn(&[BlockQ4_0], &[BlockQ8_0]) -> f32;

/// Resolve once at startup (store in a OnceLock / struct field). The fn-pointer call is per *row*, so the
/// wrappers below are the only place `unsafe` + feature detection meet.
pub fn select_dot_q8_0() -> (DotQ8, &'static str) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
            // SAFETY: features verified at runtime just above.
            return (|x, y| unsafe { x86::dot_q8_0_q8_0_avxvnni(x, y) }, "avxvnni");
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c") {
            // SAFETY: features verified at runtime just above.
            return (|x, y| unsafe { x86::dot_q8_0_q8_0_avx2(x, y) }, "avx2");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            // SAFETY: dotprod verified at runtime just above.
            return (|x, y| unsafe { neon::dot_q8_0_q8_0(x, y) }, "neon+dotprod");
        }
    }
    (dot_q8_0_q8_0_ref, "scalar")
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
        fn i8(&mut self) -> i8 { ((self.next() % 255) as i32 - 127) as i8 } // [-127,127]
        fn half_scale(&mut self) -> u16 { (((8 + self.next() % 10) as u16) << 10) | (self.next() % 1024) as u16 }
    }
    fn gen_q8(r: &mut Rng, nb: usize) -> Vec<BlockQ8_0> {
        (0..nb).map(|_| { let mut b = BlockQ8_0 { d: r.half_scale(), qs: [0; 32] }; for q in &mut b.qs { *q = r.i8(); } b }).collect()
    }
    fn gen_q4(r: &mut Rng, nb: usize) -> Vec<BlockQ4_0> {
        (0..nb).map(|_| { let mut b = BlockQ4_0 { d: r.half_scale(), qs: [0; 16] }; for q in &mut b.qs { *q = r.next() as u8; } b }).collect()
    }
    /// Error bound must scale with sum_j |d*x_j*y_j| (standard dot-product error analysis), not with |result|:
    /// f32 accumulation order differs between scalar and SIMD and per-lane partial sums can cancel.
    fn mag_q8(x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f64 {
        x.iter().zip(y).map(|(a, b)| {
            let s: i32 = (0..32).map(|j| (a.qs[j] as i32 * b.qs[j] as i32).abs()).sum();
            s as f64 * f16_to_f32(a.d) as f64 * f16_to_f32(b.d) as f64 }).sum()
    }
    fn mag_q4(x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f64 {
        x.iter().zip(y).map(|(a, b)| {
            let s: i32 = (0..16).map(|j| (((a.qs[j] & 15) as i32 - 8) * b.qs[j] as i32).abs() + (((a.qs[j] >> 4) as i32 - 8) * b.qs[j + 16] as i32).abs()).sum();
            s as f64 * f16_to_f32(a.d) as f64 * f16_to_f32(b.d) as f64 }).sum()
    }
    fn close(a: f32, b: f32, mag: f64) -> bool { ((a - b).abs() as f64) <= 5e-6 * mag + 1e-6 }

    #[test]
    fn f16_roundtrip() {
        for h in 0u16..=0xffff { let e = (h >> 10) & 0x1f; if e == 31 { continue; }
            if e != 0 { assert_eq!(f32_to_f16(f16_to_f32(h)), h, "{h:#x}"); } }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_matches_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) { return; }
        let mut r = Rng(0x1234_5678_9abc_def1);
        for nb in [1usize, 2, 3, 7, 8, 64, 65, 128] {
            for _ in 0..50 {
                let (w8, y) = (gen_q8(&mut r, nb), gen_q8(&mut r, nb));
                let w4 = gen_q4(&mut r, nb);
                let ref8 = dot_q8_0_q8_0_ref(&w8, &y);
                let ref4 = dot_q4_0_q8_0_ref(&w4, &y);
                let (m8, m4) = (mag_q8(&w8, &y), mag_q4(&w4, &y));
                { let g = unsafe { x86::dot_q8_0_q8_0_avx2(&w8, &y) }; assert!(close(g, ref8, m8), "nb={nb} got={g} ref={ref8} mag={m8}"); }
                assert!(close(unsafe { x86::dot_q4_0_q8_0_avx2(&w4, &y) }, ref4, m4));
                if is_x86_feature_detected!("avxvnni") {
                    assert!(close(unsafe { x86::dot_q8_0_q8_0_avxvnni(&w8, &y) }, ref8, m8));
                    assert!(close(unsafe { x86::dot_q4_0_q8_0_avxvnni(&w4, &y) }, ref4, m4));
                }
            }
        }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_dot4_matches() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) { return; }
        let mut r = Rng(42);
        let nb = 37; let y = gen_q8(&mut r, nb); let w = gen_q8(&mut r, 4 * nb);
        let got = unsafe { x86::dot4_q8_0_q8_0_avx2([&w[0..nb], &w[nb..2*nb], &w[2*nb..3*nb], &w[3*nb..4*nb]], &y) };
        for k in 0..4 { let wk = &w[k*nb..(k+1)*nb]; assert!(close(got[k], dot_q8_0_q8_0_ref(wk, &y), mag_q8(wk, &y))); }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_dot4_q4_matches() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) { return; }
        let mut r = Rng(4242);
        let nb = 29; let y = gen_q8(&mut r, nb); let w = gen_q4(&mut r, 4 * nb);
        let got = unsafe { x86::dot4_q4_0_q8_0_avx2([&w[0..nb], &w[nb..2*nb], &w[2*nb..3*nb], &w[3*nb..4*nb]], &y) };
        for k in 0..4 { let wk = &w[k*nb..(k+1)*nb]; assert!(close(got[k], dot_q4_0_q8_0_ref(wk, &y), mag_q4(wk, &y))); }
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_q8s_variant_matches_when_scale_is_f16() {
        if !(is_x86_feature_detected!("avxvnni") && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) { return; }
        let mut r = Rng(7);
        let nb = 16;
        let xs: Vec<f32> = (0..nb * 32).map(|_| (r.next() % 2001) as f32 / 1000.0 - 1.0).collect();
        let w4 = gen_q4(&mut r, nb);
        let mut yb = vec![x86::BlockQ8S { d: 0.0, s: 0, qs: [0; 32] }; nb];
        x86::quantize_with_sum(&xs, &mut yb);
        let y8: Vec<BlockQ8_0> = yb.iter().map(|b| BlockQ8_0 { d: f32_to_f16(b.d), qs: b.qs }).collect();
        let r1 = dot_q4_0_q8_0_ref(&w4, &y8);
        let r2 = unsafe { x86::dot_q4_0_q8s_avxvnni(&w4, &yb) };
        assert!(close(r1, r2, mag_q4(&w4, &y8)), "{r1} vs {r2}");
    }

    fn gen_q8k(r: &mut Rng, nb: usize) -> Vec<BlockQ8K> {
        (0..nb).map(|_| { let mut b = BlockQ8K { d: (1 + r.next() % 1000) as f32 / 4000.0, qs: [0; 256], bsums: [0; 16] };
            for q in &mut b.qs { *q = r.i8(); }
            for j in 0..16 { b.bsums[j] = b.qs[j*16..(j+1)*16].iter().map(|&v| v as i16).sum(); }
            b }).collect()
    }
    fn gen_q4k(r: &mut Rng, nb: usize) -> Vec<BlockQ4K> {
        (0..nb).map(|_| { let mut b = BlockQ4K { d: r.half_scale(), dmin: r.half_scale(), scales: [0; 12], qs: [0; 128] };
            for v in &mut b.scales { *v = r.next() as u8; } for v in &mut b.qs { *v = r.next() as u8; } b }).collect()
    }
    fn gen_q6k(r: &mut Rng, nb: usize) -> Vec<BlockQ6K> {
        (0..nb).map(|_| { let mut b = BlockQ6K { ql: [0; 128], qh: [0; 64], scales: [0; 16], d: r.half_scale() };
            for v in &mut b.ql { *v = r.next() as u8; } for v in &mut b.qh { *v = r.next() as u8; }
            for v in &mut b.scales { *v = (r.next() % 255) as i32 as i8; } b }).collect()
    }
    /// sum of |terms| bound for K-quant dots (see mag_q8 comment)
    fn mag_k(dx: f64, dmin: f64, dy: f64, abs_terms: f64, abs_min_terms: f64) -> f64 { dx.abs() * dy.abs() * abs_terms + dmin.abs() * dy.abs() * abs_min_terms }
    fn mag_q4k(x: &[BlockQ4K], y: &[BlockQ8K]) -> f64 {
        x.iter().zip(y).map(|(a, b)| { let (sc, mn) = unpack_q4k_scales(&a.scales);
            let mut t = 0f64;
            for sb in 0..8 { for l in 0..32 { let q = a.qs[(sb/2)*32 + l]; let v = if sb % 2 == 0 { q & 0xF } else { q >> 4 };
                t += (v as i32 * b.qs[sb*32 + l] as i32).abs() as f64 * sc[sb] as f64; } }
            let mm: f64 = (0..16).map(|j| (b.bsums[j] as i32 * mn[j/2] as i32).abs() as f64).sum();
            mag_k(f16_to_f32(a.d) as f64, f16_to_f32(a.dmin) as f64, b.d as f64, t, mm) }).sum()
    }
    fn mag_q6k(x: &[BlockQ6K], y: &[BlockQ8K]) -> f64 {
        x.iter().zip(y).map(|(a, b)| { let aux = unpack_q6k(a);
            let t: f64 = (0..256).map(|i| (aux[i] as i32 * b.qs[i] as i32).abs() as f64 * (a.scales[i / 16] as f64).abs()).sum();
            f16_to_f32(a.d).abs() as f64 * b.d as f64 * t }).sum()
    }
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_kquants_match_scalar() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) { return; }
        let mut r = Rng(0xdead_beef_1234_5678);
        let iters = if cfg!(miri) { 2 } else { 100 }; // Miri is ~1000x slower
        for nb in [1usize, 2, 5, 8] {
            for _ in 0..iters {
                let y = gen_q8k(&mut r, nb);
                let (w4, w6) = (gen_q4k(&mut r, nb), gen_q6k(&mut r, nb));
                let (r4, r6) = (dot_q4k_q8k_ref(&w4, &y), dot_q6k_q8k_ref(&w6, &y));
                let (g4, g6) = (unsafe { x86::dot_q4k_q8k_avx2(&w4, &y) }, unsafe { x86::dot_q6k_q8k_avx2(&w6, &y) });
                assert!(close(g4, r4, mag_q4k(&w4, &y)), "q4k nb={nb} got={g4} ref={r4}");
                assert!(close(g6, r6, mag_q6k(&w6, &y)), "q6k nb={nb} got={g6} ref={r6}");
            }
        }
    }
    #[test]
    fn q4k_scale_unpack_matches_ggml_get_scale_min_k4() {
        let mut r = Rng(5);
        for _ in 0..1000 {
            let mut s = [0u8; 12]; for v in &mut s { *v = r.next() as u8; }
            let (sc, mn) = unpack_q4k_scales(&s);
            for j in 0..8 {
                let (d, m) = if j < 4 { (s[j] & 63, s[j + 4] & 63) }
                             else { ((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)) };
                assert_eq!((sc[j], mn[j]), (d, m));
            }
        }
    }


    /// Documents a real pitfall: llama.cpp's sign trick (|x| via sign_epi8(x,x), y*sign(x)) mis-computes when BOTH
    /// operands are -128 (|x|=128 fits u8, but -y = +128 wraps to -128 in i8). Quantizers must clamp to [-127,127].
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_sign_trick_breaks_only_on_minus128_pairs() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) { return; }
        let one = 0x3c00u16; // f16 1.0
        let bad = [BlockQ8_0 { d: one, qs: [-128; 32] }];
        let r = dot_q8_0_q8_0_ref(&bad, &bad);
        let g = unsafe { x86::dot_q8_0_q8_0_avx2(&bad, &bad) };
        assert_eq!(r, 32.0 * 16384.0);
        assert_ne!(g, r, "expected the -128*-128 case to be wrong (got {g})");
        // -128 against anything else (e.g. 127) is fine
        let y = [BlockQ8_0 { d: one, qs: [127; 32] }];
        assert_eq!(unsafe { x86::dot_q8_0_q8_0_avx2(&bad, &y) }, dot_q8_0_q8_0_ref(&bad, &y));
    }

    #[test]
    fn quantize_error_is_bounded() {
        let mut r = Rng(99);
        let xs: Vec<f32> = (0..320).map(|_| (r.next() % 20001) as f32 / 10000.0 - 1.0).collect();
        let mut q = vec![BlockQ8_0 { d: 0, qs: [0; 32] }; 10];
        quantize_row_q8_0(&xs, &mut q);
        for (b, c) in q.iter().zip(xs.chunks(32)) { let d = f16_to_f32(b.d);
            for j in 0..32 { assert!((b.qs[j] as f32 * d - c[j]).abs() <= d * 0.5 * 1.01 + 1e-3); } }
    }
}

