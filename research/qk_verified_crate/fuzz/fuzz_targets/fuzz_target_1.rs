#![no_main]
//! Differential fuzz: AVX2 Q8_0 x Q8_0 dot vs the scalar reference.
//! Input contract (as in llama.cpp): activations y are in [-127, 127]; weights x may be any i8 incl. -128.
use libfuzzer_sys::fuzz_target;
use qk::*;

fn blocks(data: &[u8], nb: usize, clamp: bool) -> Vec<BlockQ8_0> {
    (0..nb).map(|i| {
        let c = &data[i * 34..(i + 1) * 34];
        // finite, normal, modest-exponent f16 scale: keep exponent in 8..=20, any mantissa/sign
        let raw = u16::from_le_bytes([c[0], c[1]]);
        let d = (raw & 0x83ff) | (((8 + (raw >> 10) % 13) & 0x1f) << 10);
        let mut qs = [0i8; 32];
        for j in 0..32 { let v = c[2 + j] as i8; qs[j] = if clamp && v == -128 { -127 } else { v }; }
        BlockQ8_0 { d, qs }
    }).collect()
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() { return; }
    let nb = (data[0] % 4) as usize + 1;
    let data = &data[1..];
    if data.len() < 34 * 2 * nb { return; }
    if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c")) { return; }
    let x = blocks(&data[..34 * nb], nb, false);
    let y = blocks(&data[34 * nb..], nb, !cfg!(feature = "no_clamp"));
    let want = dot_q8_0_q8_0_ref(&x, &y);
    // SAFETY: features checked above.
    let got = unsafe { x86::dot_q8_0_q8_0_avx2(&x, &y) };
    let mag: f64 = x.iter().zip(&y).map(|(a, b)| {
        let s: i32 = (0..32).map(|j| (a.qs[j] as i32 * b.qs[j] as i32).abs()).sum();
        s as f64 * (f16_to_f32(a.d) as f64 * f16_to_f32(b.d) as f64).abs() }).sum();
    assert!(((got - want).abs() as f64) <= 5e-6 * mag + 1e-6, "got={got} want={want} mag={mag}");
});
