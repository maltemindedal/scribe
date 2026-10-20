# 0003: Map GGUF files with memmap2, and keep blocks align-1

- **Status:** Accepted
- **Source:** [issue #8](https://github.com/maltemindedal/scribe/issues/8),
  implemented with the block layouts from
  [`research/gguf_check.rs`](../../../research/gguf_check.rs)

## Context

[`Gguf::parse`](../../reference/gguf-parser.md) already reads a header from a
byte slice. Loading a model has to map the file without reading the weights,
then hand out per-tensor bytes and typed blocks with no copy and no undefined
behaviour.

The workspace had no crates.io dependency and no dependency between crates.
`unsafe` is allowed in `kernels` and in the mmap module of `gguf`, and
nowhere else. `research/gguf_check.rs` stores f16 scales as `[u8; 2]`.
`research/qk_verified_crate` stores them as `u16`, which forces alignment 2.
The research README says to unify on `[u8; 2]` when porting.

Qwen3-0.6B-Q8_0 is the file `open` has to accept. It is Q8_0 weights plus F32
tensors, and it has no `output.weight`.

## Decision

- `gguf` depends on `kernels` and on `memmap2` 0.9.11. `memmap2` depends on
  `libc`. `kernels` does not depend on `gguf`.
- `GgufFile::open` is `unsafe`. It maps the file with `PROT_READ` and
  `MAP_PRIVATE` (`MmapOptions::map_copy_read_only`). `MmapOptions::map` is
  `MAP_SHARED`, so it is not used. The caller must keep the file from being
  truncated or written while the mapping is alive. Truncation delivers
  `SIGBUS` on the next fault. A write is undefined behaviour for the slices
  `memmap2` hands out. `MAP_PRIVATE` does not snapshot a read-only file.
- The mapped file stores the `Mmap` and a tensor table built at `open`:
  name, ggml type, and absolute byte range. `tensor_bytes` uses that table.
  `parsed` still walks the header when a caller wants metadata views, and
  `bytes` exposes the mapping so that caller can parse once and hold the
  `Gguf`. The header is not stored behind a `'static` lifetime. Storing the
  `Gguf` inside `GgufFile` would borrow the map from the value that owns it.
- Block structs and `cast_blocks` live in `kernels`. Each block is `repr(C)`
  and made only of `u8` and `i8` arrays, so alignment is 1. `cast_blocks`
  checks that the byte length divides the block size. `Block::ID` is the
  ggml type id. A `gguf` test fails if an id or a size disagrees with
  `GgmlType`.
- Q8_K is included. The C layout is `f32 d; i8 qs[256]; i16 bsums[16]`, which
  is alignment 4. The Rust struct stores `d` as `[u8; 4]` and `bsums` as
  `[[u8; 2]; 16]`. That is the same 292 bytes, and `cast_blocks` applies.
- `GgufFile::open` rejects a file that contains a tensor type Scribe cannot
  load, and the error lists every such type. Plain numeric types (F32, F16,
  F64, BF16, and the integer types) are loadable as bytes. The block types
  are loadable as blocks. IQ, ternary, MXFP4, NVFP4, Q8_1, Q1_0, and Q2_0
  are not. `Gguf::parse` still accepts those files.
- `MAP_POPULATE`, `MADV_POPULATE_READ`, and `MADV_HUGEPAGE` are opt-in.
  Populate and populate-read run only when asked, and only take effect on
  Linux. Huge-page advice is best-effort and ignores a kernel that rejects
  it. `open` calls none of them.

## Consequences

- The first `cargo build` downloads `memmap2` and `libc`.
- `kernels` now contains the block types. The scalar and SIMD kernels are
  still absent.
- A caller who only has a byte slice keeps using `Gguf::parse` and
  `tensor_bytes`. A caller with a path uses `GgufFile::open`.
- Calling `advise_populate_read`, or `open_with` with `populate: true`, reads
  the whole file. `advise_willneed` asks the kernel to read it in. The
  resident-set test does not call them.
