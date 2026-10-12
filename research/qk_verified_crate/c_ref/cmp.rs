// C-vs-Rust harness. NOT built by default (needs the C static lib). To run:
//   clang -O3 -mavx2 -mfma -mf16c -c c_ref/cref_k.c -o /tmp/cref_k.o && ar rcs /tmp/libcref_k.a /tmp/cref_k.o
//   cp c_ref/cmp.rs src/bin/cmp.rs
//   RUSTFLAGS="-L native=/tmp -l static=cref_k" cargo +1.99.0 run --release --bin cmp
// c_ref/cref_k.c is a transcription of llama.cpp's AVX2 Q8_0/Q4_0 vec_dot loops (single accumulator, one block per iteration)
// plus 4-row variants; it is NOT the llama.cpp binary.
use qk::*;
use std::hint::black_box;
use std::time::Instant;
unsafe extern "C" {
    fn gemv_q8(rows: i32, nb: i32, w: *const BlockQ8_0, y: *const BlockQ8_0, out: *mut f32);
    fn gemv4_q8(rows: i32, nb: i32, w: *const BlockQ8_0, y: *const BlockQ8_0, out: *mut f32);
    fn gemv_q4(rows: i32, nb: i32, w: *const BlockQ4_0, y: *const BlockQ8_0, out: *mut f32);
    fn gemv4_q4(rows: i32, nb: i32, w: *const BlockQ4_0, y: *const BlockQ8_0, out: *mut f32);
}
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
fn best<F: FnMut()>(reps: usize, mut f: F) -> f64 { let mut b = f64::MAX; for _ in 0..reps { let t = Instant::now(); f(); b = b.min(t.elapsed().as_secs_f64()); } b }
fn main() {
    let mut r = Rng(0x9e3779b97f4a7c15);
    let k = 2048usize; let nb = k / 32;
    let y = mk_q8(&mut r, nb);
    for &(label, rows) in &[("L2-resident", 256usize), ("DRAM", 200_000usize)] {
        let w8 = mk_q8(&mut r, rows * nb); let w4 = mk_q4(&mut r, rows * nb);
        let mut out = vec![0f32; rows]; let mut out2 = vec![0f32; rows];
        let (b8, b4) = ((w8.len() * 34) as f64, (w4.len() * 18) as f64);
        let reps = if rows > 10_000 { 8 } else { 3000 };
        println!("--- {label}: rows={rows} K={k}");
        let rep = |n: &str, t: f64, b: f64| println!("{n:<40} {:>7.2} GB/s", b / t / 1e9);
        let t = best(reps, || { unsafe { x86::gemv_q8_0_avx2(&w8, nb, black_box(&y), &mut out) }; black_box(&out); }); rep("Rust Q8_0 1 row/iter FIRST (before any C)", t, b8);
        let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q8_0_q8_0_avx2(&w8[i*nb..(i+1)*nb], black_box(&y)) }; } black_box(&out); }); rep("Rust Q8_0 per-row unsafe call FIRST", t, b8);
        // correctness: C vs Rust
        unsafe { gemv_q8(rows as i32, nb as i32, w8.as_ptr(), y.as_ptr(), out.as_mut_ptr()); x86::gemv_q8_0_avx2(&w8, nb, &y, &mut out2); }
        let md = out.iter().zip(&out2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("max |C - Rust| (Q8_0 gemv) = {md:e}");
        let t = best(reps, || { unsafe { gemv_q8(rows as i32, nb as i32, w8.as_ptr(), black_box(y.as_ptr()), out.as_mut_ptr()) }; black_box(&out); }); rep("C    Q8_0 1 row/iter (llama.cpp loop)", t, b8);
        let t = best(reps, || { unsafe { x86::gemv_q8_0_avx2(&w8, nb, black_box(&y), &mut out) }; black_box(&out); }); rep("Rust Q8_0 1 row/iter (2 acc, unroll 2)", t, b8);
        let t = best(reps, || { unsafe { gemv4_q8(rows as i32, nb as i32, w8.as_ptr(), black_box(y.as_ptr()), out.as_mut_ptr()) }; black_box(&out); }); rep("C    Q8_0 4 rows/iter", t, b8);
        let t = best(reps, || { unsafe { x86::gemv4_q8_0_avx2(&w8, nb, black_box(&y), &mut out) }; black_box(&out); }); rep("Rust Q8_0 4 rows/iter", t, b8);
        let t = best(reps, || { unsafe { gemv_q4(rows as i32, nb as i32, w4.as_ptr(), black_box(y.as_ptr()), out.as_mut_ptr()) }; black_box(&out); }); rep("C    Q4_0 1 row/iter (llama.cpp loop)", t, b4);
        let t = best(reps, || { for (i, o) in out.iter_mut().enumerate() { *o = unsafe { x86::dot_q4_0_q8_0_avx2(&w4[i*nb..(i+1)*nb], black_box(&y)) }; } black_box(&out); }); rep("Rust Q4_0 1 row/iter (same loop)", t, b4);
        let t = best(reps, || { unsafe { gemv4_q4(rows as i32, nb as i32, w4.as_ptr(), black_box(y.as_ptr()), out.as_mut_ptr()) }; black_box(&out); }); rep("C    Q4_0 4 rows/iter", t, b4);
        let t = best(reps, || { unsafe { x86::gemv4_q4_0_avx2(&w4, nb, black_box(&y), &mut out) }; black_box(&out); }); rep("Rust Q4_0 4 rows/iter", t, b4);
    }
}
