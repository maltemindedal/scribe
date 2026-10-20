# GGUF parser reference

The `gguf` crate parses a GGUF header from a byte slice and, through
[`GgufFile::open`](#opening-a-file), from a read-only private mapping of a
file. Tensor views borrow those bytes. Nothing is copied and nothing is
dequantised. Source: [`crates/gguf/src/`](../../crates/gguf/src/). Block
layouts live in [`kernels`](../../crates/kernels/src/block.rs).

For the API with full signatures, run `cargo doc -p gguf --open`.

## Entry point

```rust
use gguf::{Gguf, GgufFile};
use kernels::BlockQ8_0;

// SAFETY: nothing truncates or writes the file while the mapping is alive.
// Truncation delivers SIGBUS on the next fault. A write is undefined behaviour
// for the slices this returns.
let file = unsafe { GgufFile::open("models/Qwen3-0.6B-Q8_0.gguf")? };
let model = Gguf::parse(file.bytes())?;
assert_eq!(
    model.get("general.architecture").unwrap().as_str(),
    Some("qwen3")
);
let q = file.tensor_blocks::<BlockQ8_0>("blk.0.attn_q.weight")?;
println!("{} Q8_0 blocks", q.len());
```

`Gguf::parse` remains the entry point for a buffer the caller already holds.
`Gguf<'a>` borrows from that buffer. Keys, string values, and arrays are
views, not copies.

## Opening a file

`GgufFile::open` maps the file with `PROT_READ` and `MAP_PRIVATE`, parses the
header once, and keeps a table of each tensor's name, ggml type, and byte
range. Later tensor lookups use that table. `open` rejects the file when any
tensor type is not one Scribe can load. The error lists every such type, in
tensor order, once each. `Gguf::parse` does not apply that check.

| Method | Returns |
|---|---|
| `unsafe fn open(path)` | `Result<GgufFile>`. Same as `open_with` with `MapOptions::default`. |
| `unsafe fn open_with(path, MapOptions)` | `Result<GgufFile>`. `MapOptions.populate` requests `MAP_POPULATE` on Linux and prefaults the whole file. It has no effect on other operating systems. |
| `bytes()` | `&[u8]` of the whole mapping. `Gguf::parse` of this slice borrows the file. |
| `parsed()` | `Result<Gguf>` borrowing the mapping. Walks the header on every call. Tensor methods do not call it. |
| `tensor_bytes(name)` | `&[u8]` for that tensor, from the table built at `open`. The slice is inside the mapping. |
| `tensor_blocks::<B>(name)` | `&[B]` when the tensor's ggml type id is `B::ID`. |
| `output_weight()` | Bytes of `output.weight`, or of `token_embd.weight` when `output.weight` is absent. Same address as that tensor. |
| `advise_willneed()` | `madvise(MADV_WILLNEED)` over the whole mapping. This reads the file in. |
| `advise_populate_read()` | `madvise(MADV_POPULATE_READ)` on Linux, which prefaults every page. No effect elsewhere. |
| `advise_hugepage()` | `madvise(MADV_HUGEPAGE)` on Linux. A kernel that rejects the advice is ignored. No effect elsewhere. |
| `lock()` | `mlock` of the whole mapping. Fails when the lock limit is too low. |

`open` does not call the advise or lock helpers. `MapOptions::default` does
not populate.

The caller of `open` and `open_with` must keep the file from being truncated
or written for the life of the `GgufFile`. Truncation delivers `SIGBUS` on
the next fault. A write, from this process or another, is undefined behaviour
for the slices `open` returns. `MAP_PRIVATE` does not take a private copy of
a read-only file. See [ADR 0003](../architecture/decisions/0003-mmap-and-align-1-blocks.md).

Scribe loads these tensor types:

| Kind | Types |
|---|---|
| Plain bytes, no block struct | `f32`, `f16`, `f64`, `bf16`, `i8`, `i16`, `i32`, `i64` |
| Blocks in `kernels` | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q2_K`, `q3_K`, `q4_K`, `q5_K`, `q6_K`, `q8_K` |

Every other known ggml type parses, and `open` rejects the file. That
includes `q8_1`, the `iq` types, `tq1_0`, `tq2_0`, `mxfp4`, `nvfp4`, `q1_0`,
and `q2_0`.

## Types

### `Gguf<'a>`

| Method | Returns |
|---|---|
| `Gguf::parse(data: &[u8])` | `Result<Gguf>`. Parses and validates the whole header and checks every tensor's data lies inside `data`. |
| `version()` | GGUF version, `2` or `3`. |
| `get(key)` | `Option<Value>` for a metadata key. |
| `kvs()` | All metadata pairs, in file order. |
| `tensors()` | All `TensorInfo` entries, in file order. |
| `tensor(name)` | `Option<&TensorInfo>` by name. |
| `alignment()` | Tensor-data alignment: `general.alignment`, or 32 when absent. |
| `tensor_data_offset()` | Absolute byte offset where the tensor data blob starts. |
| `tensor_bytes(data, name)` | `&[u8]` for that tensor inside `data`. `data` is the slice passed to `parse`. |
| `tensor_blocks::<B>(data, name)` | `&[B]` when the tensor's ggml type id is `B::ID`. |
| `output_weight(data)` | `output.weight`, or `token_embd.weight` when `output.weight` is absent. |
| `unsupported_types()` | Tensor types Scribe cannot load, once each, in tensor order. |

The crate also exports `DEFAULT_ALIGNMENT` (`32`) and `ALIGNMENT_KEY`
(`"general.alignment"`).

### `TensorInfo<'a>`

| Member | Meaning |
|---|---|
| `name` | Tensor name. |
| `ggml_type` | Element type, a `GgmlType`. |
| `offset` | Byte offset of the data, relative to `tensor_data_offset()`. |
| `size` | Size of the data in bytes. |
| `dims()` | Dimensions in ggml order: `dims()[0]` is the contiguous dimension, so a PyTorch `[out, in]` weight appears as `[in, out]`. |
| `n_elements()` | Product of the dimensions. |

### `Value<'a>`

One variant per GGUF value type: `U8`, `I8`, `U16`, `I16`, `U32`, `I32`,
`U64`, `I64`, `F32`, `F64`, `Bool`, `String(&str)` and `Array(Array)`.

| Accessor | Returns `Some` for |
|---|---|
| `as_str()` | `String` |
| `as_array()` | `Array` |
| `as_u64()` | Unsigned integers and non-negative signed integers |
| `as_f32()` | `F32` |
| `as_bool()` | `Bool` |

### `Array<'a>`

Arrays are stored as element type, length and a byte range, so a 151,936-entry
vocabulary is not decoded at load time.

| Method | Returns |
|---|---|
| `elem_type()` | The element `ValueType`. |
| `len()`, `is_empty()` | Element count. |
| `bytes()` | The raw encoded bytes. |
| `strings()` | Iterator over `&str`, for string arrays. |
| `bools()` | Iterator over `bool`, for bool arrays. |
| `u8s()`, `i8s()`, `u16s()`, `i16s()`, `u32s()`, `i32s()`, `u64s()`, `i64s()`, `f32s()`, `f64s()` | Typed iterator when the element type matches, otherwise `None`. |

### `GgmlType`

The tensor element types, with ids and sizes copied from ggml rather than from
the GGUF spec. `id()`, `name()`, `block_size()` (elements per block) and
`type_size()` (bytes per block) expose the table. Unknown or removed ids are
rejected at parse time.

| Ids | Types |
|---|---|
| 0, 1, 28, 30 | `f32`, `f16`, `f64`, `bf16` |
| 2, 3, 6, 7, 8, 9 | `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`, `q8_1` |
| 10 to 15 | `q2_K`, `q3_K`, `q4_K`, `q5_K`, `q6_K`, `q8_K` |
| 16 to 23, 29 | `iq2_xxs`, `iq2_xs`, `iq3_xxs`, `iq1_s`, `iq4_nl`, `iq3_s`, `iq2_s`, `iq4_xs`, `iq1_m` |
| 24 to 27 | `i8`, `i16`, `i32`, `i64` |
| 34, 35 | `tq1_0`, `tq2_0` |
| 39 to 42 | `mxfp4`, `nvfp4`, `q1_0`, `q2_0` |

_As of 2026-10; source of truth: the `TABLE` in
[`crates/gguf/src/ggml_type.rs`](../../crates/gguf/src/ggml_type.rs), which
follows llama.cpp's `ggml.h` and `ggml-common.h`._

The loadable quantized types have a matching struct in `kernels`. Each struct
is `repr(C)`, alignment 1, and made of `u8` and `i8` arrays. An f16 scale is
`[u8; 2]`. Q8_K stores its f32 scale as `[u8; 4]` and its sixteen i16 sums as
`[[u8; 2]; 16]`, which keeps the block at 292 bytes and alignment 1. The C
layout of Q8_K is alignment 4. `cast_blocks` returns an error when the byte
length is not a multiple of the block size. A `gguf` test checks each block's
id and size against this table.

## Validation rules

The parser treats its input as hostile. It is at least as strict as
llama.cpp's `gguf.cpp`:

- Magic must be `GGUF`. Versions 2 and 3 are accepted. A big-endian file is
  detected and rejected.
- Every count is checked against the remaining bytes before anything is
  allocated, and all offset and size arithmetic is checked.
- Strings are at most 1 GiB and arrays at most 2^30 elements.
- Metadata keys are unique. Arrays cannot nest.
- `general.alignment`, when present, must be a `u32` power of two.
- Tensor names are unique and shorter than 64 bytes.
- Tensors have at most 4 dimensions, each at most `i64::MAX`, and the element
  count must not overflow.
- `dims()[0]` must be a multiple of the type's block size.
- Tensor data is packed in info order: each tensor starts at the previous
  tensor's end rounded up to the alignment, with no gaps, and lies inside the
  input.

It is stricter than llama.cpp in three places: strings must be valid UTF-8,
keys must be non-empty, and bools must be `0` or `1`.

Each rule has a test in
[`crates/gguf/tests/strictness.rs`](../../crates/gguf/tests/strictness.rs).

## Errors

`Gguf::parse` returns `gguf::Error`, which carries the byte `offset` where the
problem was found and an `ErrorKind`. Its `Display` output starts with
`invalid GGUF at byte <offset>:`. `ErrorKind` is `#[non_exhaustive]`.

| `ErrorKind` | Raised when |
|---|---|
| `Truncated { field, needed }` | The input ends while reading `field`. |
| `BadMagic` | The first four bytes are not `GGUF`. |
| `BigEndian` | The version looks byte-swapped. |
| `UnsupportedVersion` | The version is not 2 or 3. |
| `CountTooLarge { field, count }` | A count cannot fit in the remaining bytes. |
| `StringTooLong`, `ArrayTooLong` | A string or array exceeds the cap. |
| `InvalidUtf8 { field }` | A string is not valid UTF-8. |
| `EmptyKey`, `DuplicateKey` | A metadata key is empty or repeated. |
| `InvalidValueType`, `NestedArray`, `InvalidBool` | A metadata value is malformed. |
| `AlignmentWrongType`, `BadAlignment` | `general.alignment` is not a `u32` power of two. |
| `TensorNameTooLong`, `DuplicateTensorName` | A tensor name is too long or repeated. |
| `TooManyDims`, `DimOutOfRange`, `ShapeOverflow` | A tensor shape is invalid. |
| `UnknownGgmlType` | A tensor type id is not in the table. |
| `RowNotBlockAligned` | `dims()[0]` is not a whole number of blocks. |
| `TensorOffsetMismatch` | A tensor does not start where the packed layout expects. |
| `TensorDataOutOfBounds` | A tensor's data runs past the end of the input. |
| `UnknownTensor` | No tensor has that name. `offset` is 0. |
| `TensorTypeMismatch` | `tensor_blocks` was asked for a different ggml type than the tensor's. `offset` is 0. |
| `BlockLength` | The tensor's byte length is not a multiple of the block size. `offset` is 0. |
| `UnsupportedTensorTypes` | `GgufFile::open` found one or more tensor types Scribe cannot load. The list is complete. `offset` is 0. |
| `Io` | Opening, mapping, advising, or locking the file failed. The message is the OS error, with no `invalid GGUF` prefix. |
| `UnknownBlockId` | `Block::ID` is not a ggml type id. `offset` is 0. |

Field names in errors are dotted paths such as `kv.key` or `tensor.dims`.
