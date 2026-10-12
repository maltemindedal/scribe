//! Non-matmul kernels: exp / softmax / RMSNorm / RoPE (NeoX layout) / f16 K-cache dot.
//! x86 (AVX2+FMA+F16C) versions are tested natively; the NEON versions are only cross-compiled
//! (aarch64-apple-darwin, aarch64-unknown-linux-gnu), NOT executed.
//! The exp polynomial is a port of ggml's `ggml_v_expf` (llama.cpp ggml/src/ggml-cpu/vec.h), itself adapted from
//! Arm's optimized-routines expf: stated max error 1.45358 ulp + 0.5 ulp, flush to inf above 88.38, to 0 below -103.97.

const LN2_HI: f32 = f32::from_bits(0x3f317200); // 0x1.62e4p-1
const LN2_LO: f32 = f32::from_bits(0x35bfbe8e); // 0x1.7f7d1cp-20
const LOG2E: f32 = f32::from_bits(0x3fb8aa3b); // 0x1.715476p+0
const SHIFT: f32 = f32::from_bits(0x4b400000); // 0x1.8p23  (round-to-int trick)
const C1: f32 = f32::from_bits(0x3f7ffff6); // 0x1.ffffecp-1
const C2: f32 = f32::from_bits(0x3efffedb); // 0x1.fffdb6p-2
const C3: f32 = f32::from_bits(0x3e2aaf33); // 0x1.555e66p-3
const C5: f32 = f32::from_bits(0x3d2b9f17); // 0x1.573e2ep-5
const C7: f32 = f32::from_bits(0x3c072010); // 0x1.0e4020p-7

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::*;
    use core::arch::x86_64::*;

    /// 8-lane expf, same algorithm as ggml_v_expf (AVX2+FMA branch).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub fn expf8(x: __m256) -> __m256 {
        let r = _mm256_set1_ps(SHIFT);
        let z = _mm256_fmadd_ps(x, _mm256_set1_ps(LOG2E), r);
        let n = _mm256_sub_ps(z, r); // round(x * log2(e)) as float
        let b = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO), _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI), x)); // x - n*ln2
        let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z)); // n << 23: exponent field
        let k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1.0)))); // 2^n
        let abs_n = _mm256_andnot_ps(_mm256_set1_ps(-0.0), n);
        let c = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(126.0));
        let u = _mm256_mul_ps(b, b);
        let j = _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_fmadd_ps(_mm256_set1_ps(C7), b, _mm256_set1_ps(C5)),
                u,
                _mm256_fmadd_ps(_mm256_set1_ps(C3), b, _mm256_set1_ps(C2)),
            ),
            u,
            _mm256_mul_ps(_mm256_set1_ps(C1), b),
        );
        if _mm256_movemask_ps(c) == 0 {
            return _mm256_fmadd_ps(j, k, k); // fast path: no lane is near over/underflow
        }
        // slow path for |n| > 126: split 2^n into s1*s2 so intermediate values do not overflow
        let g = _mm256_and_si256(
            _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(n, _mm256_setzero_ps())),
            _mm256_set1_epi32(0x8200_0000u32 as i32),
        );
        let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f00_0000)));
        let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
        let d = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(192.0));
        let mid = _mm256_or_ps(
            _mm256_and_ps(c, _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
            _mm256_andnot_ps(c, _mm256_fmadd_ps(k, j, k)),
        );
        _mm256_or_ps(_mm256_and_ps(d, _mm256_mul_ps(s1, s1)), _mm256_andnot_ps(d, mid))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hmax(v: __m256) -> f32 {
        let m = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
        let m = _mm_max_ss(m, _mm_movehdup_ps(m));
        _mm_cvtss_f32(m)
    }
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_movehdup_ps(s));
        _mm_cvtss_f32(s)
    }

    /// In-place softmax: max pass, exp(x - max) with running sum, normalize.
    #[target_feature(enable = "avx2,fma")]
    pub fn softmax(x: &mut [f32]) {
        let (chunks, tail) = x.as_chunks_mut::<8>();
        // pass 1: max
        let mut vm = _mm256_set1_ps(f32::NEG_INFINITY);
        for c in chunks.iter() {
            // SAFETY: &[f32;8] = 32 readable bytes, unaligned load.
            vm = _mm256_max_ps(vm, unsafe { _mm256_loadu_ps(c.as_ptr()) });
        }
        let mut m = hmax(vm);
        for &t in tail.iter() { m = m.max(t); }
        // pass 2: exp + sum
        let vmax = _mm256_set1_ps(m);
        let mut vs = _mm256_setzero_ps();
        for c in chunks.iter_mut() {
            // SAFETY: as above; store is to the same 8 floats.
            let e = expf8(_mm256_sub_ps(unsafe { _mm256_loadu_ps(c.as_ptr()) }, vmax));
            unsafe { _mm256_storeu_ps(c.as_mut_ptr(), e) };
            vs = _mm256_add_ps(vs, e);
        }
        let mut sum = hsum(vs);
        for t in tail.iter_mut() { *t = (*t - m).exp(); sum += *t; }
        // pass 3: normalize
        let inv = 1.0 / sum;
        let vinv = _mm256_set1_ps(inv);
        for c in chunks.iter_mut() {
            let p = c.as_mut_ptr();
            // SAFETY: 8 floats, unaligned.
            unsafe { _mm256_storeu_ps(p, _mm256_mul_ps(_mm256_loadu_ps(p), vinv)) };
        }
        for t in tail.iter_mut() { *t *= inv; }
    }

    /// y = x * rsqrt(mean(x^2) + eps) * w.  Four independent accumulators, f32 (ggml accumulates in f64).
    #[target_feature(enable = "avx2,fma")]
    pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
        assert!(x.len() == w.len() && x.len() == y.len());
        let n = x.len();
        let mut acc = [_mm256_setzero_ps(); 4];
        let (xc, xt) = x.as_chunks::<32>();
        for c in xc {
            for k in 0..4 {
                // SAFETY: c is [f32;32]; offset 8k..8k+8 in range.
                let v = unsafe { _mm256_loadu_ps(c.as_ptr().add(8 * k)) };
                acc[k] = _mm256_fmadd_ps(v, v, acc[k]);
            }
        }
        let mut sum = hsum(_mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3])));
        for &t in xt { sum += t * t; }
        let scale = 1.0 / (sum / n as f32 + eps).sqrt();
        let vs = _mm256_set1_ps(scale);
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: i+8 <= n for x, w and y (equal lengths asserted above).
            unsafe {
                let v = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x.as_ptr().add(i)), vs), _mm256_loadu_ps(w.as_ptr().add(i)));
                _mm256_storeu_ps(y.as_mut_ptr().add(i), v);
            }
            i += 8;
        }
        while i < n { y[i] = x[i] * scale * w[i]; i += 1; }
    }

    /// RoPE, NeoX layout (rotate element i with element i + n/2): x0' = x0*c - x1*s ; x1' = x0*s + x1*c.
    #[target_feature(enable = "avx2,fma")]
    pub fn rope_neox(x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = x.len() / 2;
        assert!(x.len() % 2 == 0 && cos.len() >= half && sin.len() >= half);
        let (a, b) = x.split_at_mut(half);
        let mut i = 0;
        while i + 8 <= half {
            // SAFETY: i+8 <= half for a, b, cos and sin.
            unsafe {
                let (x0, x1) = (_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)));
                let (c, s) = (_mm256_loadu_ps(cos.as_ptr().add(i)), _mm256_loadu_ps(sin.as_ptr().add(i)));
                _mm256_storeu_ps(a.as_mut_ptr().add(i), _mm256_fmsub_ps(x0, c, _mm256_mul_ps(x1, s)));
                _mm256_storeu_ps(b.as_mut_ptr().add(i), _mm256_fmadd_ps(x0, s, _mm256_mul_ps(x1, c)));
            }
            i += 8;
        }
        while i < half {
            let (x0, x1) = (a[i], b[i]);
            a[i] = x0 * cos[i] - x1 * sin[i];
            b[i] = x0 * sin[i] + x1 * cos[i];
            i += 1;
        }
    }

    /// q (f32) . k (f16 bits) with f32 accumulation: the KV-cache attention score for one position.
    #[target_feature(enable = "avx2,fma,f16c")]
    pub fn dot_f32_f16(q: &[f32], k: &[u16]) -> f32 {
        assert_eq!(q.len(), k.len());
        let n = q.len();
        let mut acc = [_mm256_setzero_ps(); 2];
        let mut i = 0;
        while i + 16 <= n {
            for u in 0..2 {
                // SAFETY: i + 16 <= n.
                unsafe {
                    let kv = _mm256_cvtph_ps(_mm_loadu_si128(k.as_ptr().add(i + 8 * u).cast()));
                    acc[u] = _mm256_fmadd_ps(kv, _mm256_loadu_ps(q.as_ptr().add(i + 8 * u)), acc[u]);
                }
            }
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(acc[0], acc[1]));
        while i < n { s += q[i] * crate::f16_to_f32(k[i]); i += 1; }
        s
    }
}

#[cfg(target_arch = "aarch64")]
pub mod neon {
    use super::*;
    use core::arch::aarch64::*;

    /// 4-lane expf, same algorithm as ggml_v_expf (NEON branch). Cross-compiled only, NOT executed.
    #[inline]
    #[target_feature(enable = "neon")]
    pub fn expf4(x: float32x4_t) -> float32x4_t {
        let r = vdupq_n_f32(SHIFT);
        let z = vfmaq_f32(r, x, vdupq_n_f32(LOG2E));
        let n = vsubq_f32(z, r);
        let b = vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(LN2_HI)), n, vdupq_n_f32(LN2_LO));
        let e = vshlq_n_u32::<23>(vreinterpretq_u32_f32(z));
        let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
        let c = vcagtq_f32(n, vdupq_n_f32(126.0));
        let u = vmulq_f32(b, b);
        let j = vfmaq_f32(
            vmulq_f32(vdupq_n_f32(C1), b),
            vfmaq_f32(
                vfmaq_f32(vdupq_n_f32(C2), vdupq_n_f32(C3), b),
                vfmaq_f32(vdupq_n_f32(C5), vdupq_n_f32(C7), b),
                u,
            ),
            u,
        );
        if vpaddd_u64(vreinterpretq_u64_u32(c)) == 0 {
            return vfmaq_f32(k, j, k);
        }
        let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
        let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f00_0000)));
        let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
        vbslq_f32(
            vcagtq_f32(n, vdupq_n_f32(192.0)),
            vmulq_f32(s1, s1),
            vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
        )
    }

    /// q (f32) . k (f16 bits), f32 accumulation (vcvt_f32_f16 is stable since Rust 1.94). Cross-compiled only.
    #[target_feature(enable = "neon")]
    pub fn dot_f32_f16(q: &[f32], k: &[u16]) -> f32 {
        assert_eq!(q.len(), k.len());
        let n = q.len();
        let mut acc = [vdupq_n_f32(0.0); 4];
        let mut i = 0;
        while i + 16 <= n {
            for u in 0..4 {
                // SAFETY: i + 16 <= n; vld1_u16 / vld1q_f32 need only element alignment.
                unsafe {
                    let kv = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(k.as_ptr().add(i + 4 * u))));
                    acc[u] = vfmaq_f32(acc[u], kv, vld1q_f32(q.as_ptr().add(i + 4 * u)));
                }
            }
            i += 16;
        }
        let mut s = vaddvq_f32(vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3])));
        while i < n { s += q[i] * crate::f16_to_f32(k[i]); i += 1; }
        s
    }

    /// RoPE NeoX layout with 4 lanes; for the adjacent-pair layout use `vld2q_f32` to de-interleave. Cross-compiled only.
    #[target_feature(enable = "neon")]
    pub fn rope_neox(x: &mut [f32], cos: &[f32], sin: &[f32]) {
        let half = x.len() / 2;
        assert!(x.len() % 2 == 0 && cos.len() >= half && sin.len() >= half);
        let (a, b) = x.split_at_mut(half);
        let mut i = 0;
        while i + 4 <= half {
            // SAFETY: i+4 <= half for a, b, cos and sin.
            unsafe {
                let (x0, x1) = (vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
                let (c, s) = (vld1q_f32(cos.as_ptr().add(i)), vld1q_f32(sin.as_ptr().add(i)));
                vst1q_f32(a.as_mut_ptr().add(i), vfmsq_f32(vmulq_f32(x0, c), x1, s));
                vst1q_f32(b.as_mut_ptr().add(i), vfmaq_f32(vmulq_f32(x1, c), x0, s));
            }
            i += 4;
        }
        while i < half {
            let (x0, x1) = (a[i], b[i]);
            a[i] = x0 * cos[i] - x1 * sin[i];
            b[i] = x0 * sin[i] + x1 * cos[i];
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86_64")]
    use super::x86;

    #[cfg(target_arch = "x86_64")]
    fn have() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")
    }
    /// distance in ulps between two finite f32 of the same sign (monotone integer mapping)
    fn ulps(a: f32, b: f32) -> u32 {
        let m = |f: f32| { let i = f.to_bits() as i32; if i < 0 { i32::MIN.wrapping_sub(i) } else { i } };
        (m(a) as i64 - m(b) as i64).unsigned_abs() as u32
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn expf8_accuracy_and_edges() {
        if !have() { return; }
        let mut worst = 0u32;
        let mut worst_x = 0f32;
        let step = if cfg!(miri) { 4.0 } else { 0.001 };
        let mut x = -87.0f32;
        while x < 88.0 {
            let lanes: [f32; 8] = core::array::from_fn(|i| x + i as f32 * 0.000123);
            // SAFETY: features checked by have().
            let out: [f32; 8] = unsafe {
                use core::arch::x86_64::*;
                let r = x86::expf8(_mm256_loadu_ps(lanes.as_ptr()));
                let mut o = [0f32; 8]; _mm256_storeu_ps(o.as_mut_ptr(), r); o
            };
            for i in 0..8 {
                let want = (lanes[i] as f64).exp() as f32;
                let u = ulps(out[i], want);
                if u > worst { worst = u; worst_x = lanes[i]; }
            }
            x += step;
        }
        eprintln!("expf8 worst error vs f64 exp (rounded to f32): {worst} ulp at x={worst_x}");
        assert!(worst <= 3, "worst {worst} ulp at {worst_x}");
        // edges (documented behaviour of ggml_v_expf): +inf above 88.38, 0 below -103.97, exp(0)=1
        let edge: [f32; 8] = [89.0, 100.0, -104.5, -120.0, 0.0, 88.0, -87.0, -1.0];
        let out: [f32; 8] = unsafe {
            use core::arch::x86_64::*;
            let r = x86::expf8(_mm256_loadu_ps(edge.as_ptr()));
            let mut o = [0f32; 8]; _mm256_storeu_ps(o.as_mut_ptr(), r); o
        };
        assert!(out[0].is_infinite() && out[1].is_infinite());
        assert_eq!(out[2], 0.0); assert_eq!(out[3], 0.0);
        assert_eq!(out[4], 1.0);
        assert!(ulps(out[5], (88.0f64).exp() as f32) <= 3);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn softmax_matches_f64() {
        if !have() { return; }
        for n in [1usize, 7, 8, 9, 33, 128, 1000] {
            let mut s = 12345u32 + n as u32;
            let mut x: Vec<f32> = (0..n).map(|_| { s = s.wrapping_mul(1664525).wrapping_add(1013904223); ((s >> 8) as f32 / 16777216.0 - 0.5) * 40.0 }).collect();
            let r = {
                let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let e: Vec<f64> = x.iter().map(|&v| (v as f64 - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                e.into_iter().map(|v| v / sum).collect::<Vec<_>>()
            };
            unsafe { x86::softmax(&mut x) };
            let sum: f32 = x.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "n={n} sum={sum}");
            for (a, b) in x.iter().zip(&r) { assert!(((*a as f64) - b).abs() <= 1e-6 + 2e-6 * b, "n={n} {a} vs {b}"); }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rms_norm_and_rope_and_f16dot() {
        if !have() { return; }
        let mut s = 7u32;
        let mut rnd = || { s = s.wrapping_mul(1664525).wrapping_add(1013904223); (s >> 8) as f32 / 16777216.0 - 0.5 };
        for n in [3usize, 32, 64, 896, 2048, 2049] {
            let x: Vec<f32> = (0..n).map(|_| rnd() * 4.0).collect();
            let w: Vec<f32> = (0..n).map(|_| rnd() + 1.0).collect();
            let mut y = vec![0f32; n];
            unsafe { x86::rms_norm(&x, &w, 1e-6, &mut y) };
            let ms: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n as f64;
            let sc = 1.0 / (ms + 1e-6).sqrt();
            for i in 0..n { let want = x[i] as f64 * sc * w[i] as f64; assert!(((y[i] as f64) - want).abs() <= 1e-5 * want.abs() + 1e-6, "rms n={n} i={i}"); }
        }
        for half in [4usize, 8, 32, 37] {
            let mut x: Vec<f32> = (0..2 * half).map(|_| rnd()).collect();
            let c: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).cos()).collect();
            let sn: Vec<f32> = (0..half).map(|i| (i as f32 * 0.37).sin()).collect();
            let r: Vec<f32> = (0..half).map(|i| x[i] * c[i] - x[half + i] * sn[i]).chain((0..half).map(|i| x[i] * sn[i] + x[half + i] * c[i])).collect();
            unsafe { x86::rope_neox(&mut x, &c, &sn) };
            for i in 0..2 * half { assert!((x[i] - r[i]).abs() <= 1e-6, "rope half={half} i={i}"); }
        }
        for n in [1usize, 15, 16, 64, 100, 128] {
            let q: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let kf: Vec<f32> = (0..n).map(|_| rnd()).collect();
            let kh: Vec<u16> = kf.iter().map(|&v| crate::f32_to_f16(v)).collect();
            let want: f64 = q.iter().zip(&kh).map(|(&a, &h)| a as f64 * crate::f16_to_f32(h) as f64).sum();
            let got = unsafe { x86::dot_f32_f16(&q, &kh) } as f64;
            assert!((got - want).abs() <= 1e-5 * (n as f64).sqrt() + 1e-6, "f16dot n={n} {got} vs {want}");
        }
    }
}
