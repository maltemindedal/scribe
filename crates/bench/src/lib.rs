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
