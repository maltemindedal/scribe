//! Block layouts for ggml tensors. Scalar and SIMD kernels, and the thread
//! pool, are not built yet.
//!
//! This crate is allowed to use `unsafe`. `mapfile` is the other crate.
//! [`cast_blocks`] is the only `unsafe` operation here.

mod block;

pub use block::{
    Block, BlockCastError, BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0,
    BlockQ5_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8K, cast_blocks,
};

/// Test fixtures for crates that forbid `unsafe` and so cannot write
/// `unsafe impl Block`. Enabled by the `testing` feature.
#[cfg(feature = "testing")]
pub mod testing {
    /// A one-byte block whose id, 999, is not a ggml type id.
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct UnknownIdBlock(pub u8);

    // SAFETY: `repr(C)` around one `u8`: alignment 1, no padding, and every
    // bit pattern is valid.
    unsafe impl crate::Block for UnknownIdBlock {
        const ID: u32 = 999;
    }
}
