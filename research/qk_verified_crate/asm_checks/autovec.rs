// Auto-vectorization check (section 1 of the notes). Build and inspect:
//   rustc +1.99.0 -O -C target-cpu=x86-64-v3 --crate-type=lib --emit asm asm_checks/autovec.rs -o /tmp/autovec.s
//   sum_naive / dot_naive  -> scalar vaddss / vfmadd231ss (LLVM may not reassociate f32 sums)
//   sum_algebraic / dot_algebraic (f32::algebraic_*, stable 1.98) and dot_lanes (8 explicit lanes) -> ymm vaddps / vfmadd231ps
#[unsafe(no_mangle)] pub fn sum_naive(x: &[f32]) -> f32 { x.iter().sum() }
#[unsafe(no_mangle)] pub fn sum_algebraic(x: &[f32]) -> f32 { let mut s = 0f32; for &v in x { s = s.algebraic_add(v); } s }
#[unsafe(no_mangle)] pub fn dot_naive(a: &[f32], b: &[f32]) -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() }
#[unsafe(no_mangle)] pub fn dot_algebraic(a: &[f32], b: &[f32]) -> f32 { let mut s = 0f32; for (&x, &y) in a.iter().zip(b) { s = s.algebraic_add(x.algebraic_mul(y)); } s }
#[unsafe(no_mangle)] pub fn dot_lanes(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) { for i in 0..8 { acc[i] += x[i] * y[i]; } }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ar.iter().zip(br) { s += x * y; }
    s
}
