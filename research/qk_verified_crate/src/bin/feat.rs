fn main() {
    println!("avx2={} fma={} f16c={} avxvnni={} avx512f={} avx512vnni={}",
        is_x86_feature_detected!("avx2"), is_x86_feature_detected!("fma"), is_x86_feature_detected!("f16c"),
        is_x86_feature_detected!("avxvnni"), is_x86_feature_detected!("avx512f"), is_x86_feature_detected!("avx512vnni"));
    // exercise one VNNI intrinsic directly
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avxvnni") {
        use core::arch::x86_64::*;
        #[target_feature(enable = "avx2,avxvnni")]
        fn go() -> i32 { let a = _mm256_set1_epi8(3); let b = _mm256_set1_epi8(-2);
            _mm256_extract_epi32::<0>(_mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), a, b)) }
        println!("dpbusd lane0 = {}", unsafe { go() });
    }
}

// Not compiled (E0658 on rustc 1.99.0): x86 AMX feature detection is still unstable.
//   fn main() { println!("{}", is_x86_feature_detected!("amx-int8")); }
//   error[E0658]: use of unstable library feature `x86_amx_intrinsics` (tracking issue #126622)
// candle works around it with `__cpuid_count(7, 0)` (EDX bit 24 = AMX-TILE, bit 25 = AMX-INT8).
