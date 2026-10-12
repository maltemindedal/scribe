use qk::*;
use std::hint::black_box;
use std::time::Instant;

struct Rng(u64);
impl Rng { fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 } }

fn mk_q8(r: &mut Rng, n: usize) -> Vec<BlockQ8_0> {
    (0..n).map(|_| { let mut b = BlockQ8_0 { d: 0x3000 | (r.next() % 1024) as u16, qs: [0; 32] };
        for q in &mut b.qs { *q = ((r.next() % 255) as i32 - 127) as i8; } b }).collect()
}
fn mk_q4(r: &mut Rng, n: usize) -> Vec<BlockQ4_0> {
    (0..n).map(|_| { let mut b = BlockQ4_0 { d: 0x3000 | (r.next() % 1024) as u16, qs: [0; 16] };
        for q in &mut b.qs { *q = r.next() as u8; } b }).collect()
}
fn mk_q4k(r: &mut Rng, n: usize) -> Vec<BlockQ4K> {
    (0..n).map(|_| { let mut b = BlockQ4K { d: 0x3000 | (r.next() % 1024) as u16, dmin: 0x3000 | (r.next() % 1024) as u16, scales: [0; 12], qs: [0; 128] };
        for v in &mut b.scales { *v = r.next() as u8; } for v in &mut b.qs { *v = r.next() as u8; } b }).collect()
}
fn mk_q6k(r: &mut Rng, n: usize) -> Vec<BlockQ6K> {
    (0..n).map(|_| { let mut b = BlockQ6K { ql: [0; 128], qh: [0; 64], scales: [0; 16], d: 0x3000 | (r.next() % 1024) as u16 };
        for v in &mut b.ql { *v = r.next() as u8; } for v in &mut b.qh { *v = r.next() as u8; }
        for v in &mut b.scales { *v = (r.next() % 255) as i32 as i8; } b }).collect()
}
fn mk_q8k(r: &mut Rng, n: usize) -> Vec<BlockQ8K> {
    (0..n).map(|_| { let mut b = BlockQ8K { d: 0.01, qs: [0; 256], bsums: [0; 16] };
        for q in &mut b.qs { *q = ((r.next() % 255) as i32 - 127) as i8; }
        for j in 0..16 { b.bsums[j] = b.qs[j*16..(j+1)*16].iter().map(|&v| v as i16).sum(); } b }).collect()
}
fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..reps { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); }
    b
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn stream_read(buf: &[u8]) -> i64 {
    use core::arch::x86_64::*;
    let mut a = [_mm256_setzero_si256(); 4];
    let n = buf.len() / 128;
    for i in 0..n {
        let p = buf.as_ptr().wrapping_add(i * 128) as *const __m256i;
        // SAFETY: i*128+128 <= buf.len(); unaligned loads.
        unsafe { for k in 0..4 { a[k] = _mm256_add_epi64(a[k], _mm256_loadu_si256(p.add(k))); } }
    }
    let s = _mm256_add_epi64(_mm256_add_epi64(a[0], a[1]), _mm256_add_epi64(a[2], a[3]));
    let mut o = [0i64; 4];
    unsafe { _mm256_storeu_si256(o.as_mut_ptr().cast(), s) };
    o.iter().sum()
}

fn main() {
    let mut r = Rng(0x9e3779b97f4a7c15);
    let args: Vec<String> = std::env::args().collect();
    let k = 2048usize; let nb = k / 32;
    let y = mk_q8(&mut r, nb);
    for &(label, rows) in &[("L2-resident", 256usize), ("DRAM", 200_000usize)] {
        let w8 = mk_q8(&mut r, rows * nb);
        let w4 = mk_q4(&mut r, rows * nb);
        let nsb = k / 256;
        let yk = mk_q8k(&mut r, nsb);
        let wq4k = mk_q4k(&mut r, rows * nsb);
        let wq6k = mk_q6k(&mut r, rows * nsb);
        let mut out = vec![0f32; rows];
        let bytes8 = (w8.len() * 34) as f64; let bytes4 = (w4.len() * 18) as f64;
        let reps = if rows > 10_000 { 5 } else { 2000 };
        println!("--- {label}: rows={rows} K={k}  Q8_0 weights {:.1} MB, Q4_0 weights {:.1} MB", bytes8 / 1e6, bytes4 / 1e6);
        let rep = |name: &str, secs: f64, bytes: f64| println!("{name:<34} {:>8.2} GB/s   {:>7.1} ns/row", bytes / secs / 1e9, secs / rows as f64 * 1e9);

        #[cfg(target_arch = "x86_64")]
        {
            let b4k = (wq4k.len() * 144) as f64; let b6k = (wq6k.len() * 210) as f64;
            let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q4k_q8k_avx2(&wq4k[i*nsb..(i+1)*nsb], black_box(&yk)) }; } black_box(&out); });
            rep("Q4_K avx2 (per-row)", t, b4k);
            let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q6k_q8k_avx2(&wq6k[i*nsb..(i+1)*nsb], black_box(&yk)) }; } black_box(&out); });
            rep("Q6_K avx2 (per-row)", t, b6k);
            let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = dot_q4k_q8k_ref(&wq4k[i*nsb..(i+1)*nsb], black_box(&yk)); } black_box(&out); });
            rep("Q4_K scalar ref", t, b4k);
        }
        #[cfg(target_arch = "x86_64")]
        {
            let buf: &[u8] = unsafe { std::slice::from_raw_parts(w8.as_ptr().cast(), w8.len() * 34) };
            let t = best(reps, || { black_box(unsafe { stream_read(black_box(buf)) }); });
            rep("pure read (4x ymm adds)", t, bytes8);
        }
        let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = dot_q8_0_q8_0_ref(&w8[i*nb..(i+1)*nb], black_box(&y)); } black_box(&out); });
        rep("Q8_0 scalar ref (compiled flags)", t, bytes8);
        let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = dot_q4_0_q8_0_ref(&w4[i*nb..(i+1)*nb], black_box(&y)); } black_box(&out); });
        rep("Q4_0 scalar ref (compiled flags)", t, bytes4);
        #[cfg(target_arch = "x86_64")]
        {
            let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q8_0_q8_0_avx2(&w8[i*nb..(i+1)*nb], black_box(&y)) }; } black_box(&out); });
            rep("Q8_0 avx2 (per-row unsafe call)", t, bytes8);
            let t = best(reps, || { unsafe { x86::gemv_q8_0_avx2(&w8, nb, black_box(&y), &mut out) }; black_box(&out); });
            rep("Q8_0 avx2 (gemv inside tf fn)", t, bytes8);
            let t = best(reps, || { unsafe { x86::gemv4_q8_0_avx2(&w8, nb, black_box(&y), &mut out) }; black_box(&out); });
            rep("Q8_0 avx2 dot4 (4 rows/iter)", t, bytes8);
            let (f, name) = select_dot_q8_0();
            let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = f(&w8[i*nb..(i+1)*nb], black_box(&y)); } black_box(&out); });
            rep(&format!("Q8_0 fn-ptr dispatch ({name})"), t, bytes8);
            if is_x86_feature_detected!("avxvnni") {
                let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q8_0_q8_0_avxvnni(&w8[i*nb..(i+1)*nb], black_box(&y)) }; } black_box(&out); });
                rep("Q8_0 avx-vnni (per-row)", t, bytes8);
            }
            let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q4_0_q8_0_avx2(&w4[i*nb..(i+1)*nb], black_box(&y)) }; } black_box(&out); });
            rep("Q4_0 avx2 (per-row)", t, bytes4);
            let t = best(reps, || { unsafe { x86::gemv4_q4_0_avx2(&w4, nb, black_box(&y), &mut out) }; black_box(&out); });
            rep("Q4_0 avx2 dot4 (4 rows/iter)", t, bytes4);
            if is_x86_feature_detected!("avxvnni") {
                let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q4_0_q8_0_avxvnni(&w4[i*nb..(i+1)*nb], black_box(&y)) }; } black_box(&out); });
                rep("Q4_0 avx-vnni (per-row)", t, bytes4);
            }
        }
    }
    let _ = args;
}
