//! Benchmarks and reference comparison.
//!
//! Today: reading reference tensor dumps ([`dump`]), comparing tensors by
//! NMSE ([`compare`]), and comparing logits by KL divergence ([`kld`]). The
//! llama.cpp dumps come from `tools/llama.cpp` (#21); the Hugging Face dumps
//! planned in #20 are meant to use the same format.
#![forbid(unsafe_code)]

pub mod compare;
pub mod dump;
pub mod kld;

/// `f64::max` that returns NaN when either input is NaN, so a NaN in the data
/// cannot be hidden by the comparison.
#[expect(clippy::disallowed_methods, reason = "the one NaN-propagating wrapper")]
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}
