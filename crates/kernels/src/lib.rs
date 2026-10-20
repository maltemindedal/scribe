//! Block layouts for ggml tensors. Scalar and SIMD kernels, and the thread
//! pool, are not built yet.
//!
//! This crate is allowed to use `unsafe`. The mmap module in `gguf` is the
//! other place. [`cast_blocks`] is the only `unsafe` operation here.

mod block;

pub use block::{
    Block, BlockCastError, BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0,
    BlockQ5_1, BlockQ5K, BlockQ6K, BlockQ8_0, BlockQ8K, cast_blocks,
};
