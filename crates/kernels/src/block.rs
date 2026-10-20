//! On-disk ggml block layouts.
//!
//! Each block is `repr(C)` and made only of `u8` and `i8` arrays, so its
//! alignment is 1, it has no padding, and every bit pattern is a valid value.
//! f16 scales are `[u8; 2]` in little-endian order. That is the layout
//! `research/gguf_check.rs` uses. The `qk` research crate stored the same
//! scales as `u16`, which forces alignment 2.
//!
//! Q8_K is the exception the C header writes as `f32 d; i8 qs[256]; i16
//! bsums[16]`. That struct is alignment 4. A cast from an arbitrary byte
//! offset would be undefined. The fields here are `[u8; 4]` and
//! `[[u8; 2]; 16]`, which is the same 292 bytes and stays alignment 1, so
//! [`cast_blocks`] applies to it with the same length check as the others.

use std::mem::{align_of, size_of};

/// A quantized block as stored in a GGUF tensor.
///
/// # Safety
///
/// The type is `repr(C)`, alignment 1, with no padding and no invalid bit
/// patterns. Every field is `u8`, `i8`, or an array of those. [`cast_blocks`]
/// relies on this and checks only that the byte length divides the block size.
pub unsafe trait Block: Copy {
    /// ggml type id of tensors stored as this block.
    const ID: u32;

    /// Bytes occupied by one block.
    const BYTES: usize = size_of::<Self>();
}

/// `bytes` is not a whole number of blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockCastError {
    /// Length of the slice that was rejected.
    pub len: usize,
    /// `B::BYTES` for the requested block type.
    pub block_size: usize,
}

impl std::fmt::Display for BlockCastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} bytes is not a multiple of the {} byte block",
            self.len, self.block_size
        )
    }
}

impl std::error::Error for BlockCastError {}

/// Reinterprets `bytes` as a slice of blocks.
///
/// Returns [`BlockCastError`] when `bytes.len()` is not a multiple of the
/// block size. Alignment is not checked: every [`Block`] has alignment 1.
pub fn cast_blocks<B: Block>(bytes: &[u8]) -> Result<&[B], BlockCastError> {
    const {
        assert!(align_of::<B>() == 1);
        assert!(size_of::<B>() > 0);
    }
    let block_size = size_of::<B>();
    if !bytes.len().is_multiple_of(block_size) {
        return Err(BlockCastError {
            len: bytes.len(),
            block_size,
        });
    }
    // SAFETY: `B: Block` is alignment 1 and has no invalid bit patterns, and
    // the length divides the block size. The slice is a valid `[B]`.
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<B>(), bytes.len() / block_size) })
}

/// Q4_0. 18 bytes, 32 weights. `d` is an f16 scale.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_0 {
    pub d: [u8; 2],
    pub qs: [u8; 16],
}

const _: () = assert!(size_of::<BlockQ4_0>() == 18);
const _: () = assert!(align_of::<BlockQ4_0>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ4_0 {
    const ID: u32 = 2;
}

/// Q4_1. 20 bytes, 32 weights. `d` and `m` are f16.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4_1 {
    pub d: [u8; 2],
    pub m: [u8; 2],
    pub qs: [u8; 16],
}

const _: () = assert!(size_of::<BlockQ4_1>() == 20);
const _: () = assert!(align_of::<BlockQ4_1>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ4_1 {
    const ID: u32 = 3;
}

/// Q5_0. 22 bytes, 32 weights.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ5_0 {
    pub d: [u8; 2],
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}

const _: () = assert!(size_of::<BlockQ5_0>() == 22);
const _: () = assert!(align_of::<BlockQ5_0>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ5_0 {
    const ID: u32 = 6;
}

/// Q5_1. 24 bytes, 32 weights.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ5_1 {
    pub d: [u8; 2],
    pub m: [u8; 2],
    pub qh: [u8; 4],
    pub qs: [u8; 16],
}

const _: () = assert!(size_of::<BlockQ5_1>() == 24);
const _: () = assert!(align_of::<BlockQ5_1>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ5_1 {
    const ID: u32 = 7;
}

/// Q8_0. 34 bytes, 32 weights.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8_0 {
    pub d: [u8; 2],
    pub qs: [i8; 32],
}

const _: () = assert!(size_of::<BlockQ8_0>() == 34);
const _: () = assert!(align_of::<BlockQ8_0>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ8_0 {
    const ID: u32 = 8;
}

/// Q2_K. 84 bytes, 256 weights. `d` is last, with `dmin`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ2K {
    pub scales: [u8; 16],
    pub qs: [u8; 64],
    pub d: [u8; 2],
    pub dmin: [u8; 2],
}

const _: () = assert!(size_of::<BlockQ2K>() == 84);
const _: () = assert!(align_of::<BlockQ2K>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ2K {
    const ID: u32 = 10;
}

/// Q3_K. 110 bytes, 256 weights.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ3K {
    pub hmask: [u8; 32],
    pub qs: [u8; 64],
    pub scales: [u8; 12],
    pub d: [u8; 2],
}

const _: () = assert!(size_of::<BlockQ3K>() == 110);
const _: () = assert!(align_of::<BlockQ3K>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ3K {
    const ID: u32 = 11;
}

/// Q4_K. 144 bytes, 256 weights.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ4K {
    pub d: [u8; 2],
    pub dmin: [u8; 2],
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

const _: () = assert!(size_of::<BlockQ4K>() == 144);
const _: () = assert!(align_of::<BlockQ4K>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ4K {
    const ID: u32 = 12;
}

/// Q5_K. 176 bytes, 256 weights.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ5K {
    pub d: [u8; 2],
    pub dmin: [u8; 2],
    pub scales: [u8; 12],
    pub qh: [u8; 32],
    pub qs: [u8; 128],
}

const _: () = assert!(size_of::<BlockQ5K>() == 176);
const _: () = assert!(align_of::<BlockQ5K>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ5K {
    const ID: u32 = 13;
}

/// Q6_K. 210 bytes, 256 weights. `d` is the last field, as in `ggml-common.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ6K {
    pub ql: [u8; 128],
    pub qh: [u8; 64],
    pub scales: [i8; 16],
    pub d: [u8; 2],
}

const _: () = assert!(size_of::<BlockQ6K>() == 210);
const _: () = assert!(align_of::<BlockQ6K>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ6K {
    const ID: u32 = 14;
}

/// Q8_K. 292 bytes, 256 weights. Activation format.
///
/// `d` is the little-endian bytes of an f32. Each `bsums` entry is the
/// little-endian bytes of an i16. See the module docs for why this is not
/// the C struct.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BlockQ8K {
    pub d: [u8; 4],
    pub qs: [i8; 256],
    pub bsums: [[u8; 2]; 16],
}

const _: () = assert!(size_of::<BlockQ8K>() == 292);
const _: () = assert!(align_of::<BlockQ8K>() == 1);
// SAFETY: `repr(C)`, alignment 1, only byte arrays. See [`Block`].
unsafe impl Block for BlockQ8K {
    const ID: u32 = 15;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cast_blocks_rejects_a_partial_block() {
        let err = cast_blocks::<BlockQ8_0>(&[0u8; 33]).unwrap_err();
        assert_eq!(
            err,
            BlockCastError {
                len: 33,
                block_size: 34
            }
        );
    }

    #[test]
    fn cast_blocks_rejects_q4_k_partial() {
        let err = cast_blocks::<BlockQ4K>(&[0u8; 143]).unwrap_err();
        assert_eq!(err.block_size, 144);
        assert_eq!(err.len, 143);
    }

    #[test]
    fn cast_blocks_views_exact_blocks_without_copying() {
        let mut bytes = [0u8; 68];
        bytes[0] = 0x34;
        bytes[1] = 0x12;
        bytes[2] = -5i8 as u8;
        bytes[34] = 0x78;
        let blocks = cast_blocks::<BlockQ8_0>(&bytes).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].d, [0x34, 0x12]);
        assert_eq!(blocks[0].qs[0], -5);
        assert_eq!(blocks[1].d[0], 0x78);
        assert_eq!(blocks.as_ptr() as *const u8, bytes.as_ptr());
    }

    #[test]
    fn empty_input_is_an_empty_block_slice() {
        assert!(cast_blocks::<BlockQ4_0>(&[]).unwrap().is_empty());
        assert!(cast_blocks::<BlockQ8K>(&[]).unwrap().is_empty());
    }
}
