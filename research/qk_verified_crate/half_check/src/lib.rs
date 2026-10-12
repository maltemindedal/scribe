use half::f16;
#[target_feature(enable = "avx2,fma")]
pub fn sum_scales(xs: &[u16], ys: &[u16]) -> f32 {
    let mut acc = 0f32;
    for (x, y) in xs.iter().zip(ys) {
        acc += f16::from_bits(*x).to_f32() * f16::from_bits(*y).to_f32();
    }
    acc
}
