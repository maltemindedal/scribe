# Architecture overview

Scribe is a Cargo workspace of ten crates. Each crate owns one layer of an
LLM inference engine. Today only the GGUF parser is implemented; the other
crates are empty stubs whose boundaries are fixed in their manifests and
crate docs, so later work has a defined place to land.

The project goal, milestones and build order live in the
[roadmap](https://github.com/maltemindedal/scribe/issues/1).

## Crates

| Crate | Owns | Status |
|---|---|---|
| `gguf` | GGUF file parsing, mmap, tensor views | Header parser, read-only private mapping, and per-tensor byte and block views ([reference](../reference/gguf-parser.md)). |
| `tokenizer` | BPE (later SentencePiece) tokenizer, chat templates | Stub |
| `kernels` | Block types, scalar and SIMD kernels, thread pool | Align-1 block structs and `cast_blocks`. Scalar and SIMD kernels are not built. |
| `model` | Model config, weights, forward pass, KV cache | Stub |
| `sampler` | Logit processing and token sampling | Stub |
| `engine` | Slots, prefix cache, scheduler | Stub |
| `server` | HTTP server and OpenAI-compatible API | Stub |
| `bench` | Benchmarks, and comparison against reference dumps: the dump reader, NMSE bands and logit KL divergence (`compare` binary, see [`tools/llama.cpp`](../../tools/llama.cpp/README.md)) | Comparison tools; no benchmarks yet |
| `xtask` | Developer tasks run as `cargo xtask` | `models fetch` implemented ([MODELS.md](../../MODELS.md)) |
| `cli` | The `scribe` binary | `--help` and `--version` only ([reference](../reference/cli.md)) |

_As of 2026-10; source of truth: each crate's `Cargo.toml` `description` and
`src/lib.rs` (or `src/main.rs`) doc comment._

`gguf` depends on `kernels` and on `memmap2` (which depends on `libc`). No
other crate depends on another, and those two crates.io packages are the only
third-party dependencies. See
[ADR 0003](decisions/0003-mmap-and-align-1-blocks.md).

## Boundaries enforced in code

Some rules are already written into the crates, ahead of the code they
constrain:

- **`unsafe` is confined.** Every crate except `kernels` and `gguf` has
  `#![forbid(unsafe_code)]`. `gguf` has `#![deny(unsafe_code)]` instead,
  because `forbid` cannot be relaxed later for its planned mmap module
  (rationale from [PR #46](https://github.com/maltemindedal/scribe/pull/46)).
  `kernels` is the one crate where `unsafe` is allowed.
- **`unsafe` operations are explicit.** The workspace lint
  `unsafe_op_in_unsafe_fn = "deny"` requires an `unsafe` block around every
  unsafe operation, even inside an `unsafe fn`.
- **`engine` has no HTTP and no JSON**, per its crate doc. `server` is
  documented as the only crate allowed `serde_json`.

## The GGUF parser

`gguf::Gguf::parse` takes a `&[u8]` and returns a structure that borrows from
it. `GgufFile::open` maps a file read-only and private, parses that mapping
once, and keeps a tensor table for later lookups. `bytes` returns the mapping
so a caller can parse metadata once. The parser works in four passes over the
header:

1. **Header:** magic, version, tensor count, metadata count.
2. **Metadata:** key-value pairs. Scalars are decoded; arrays are kept as
   element type, length and byte range, and decoded lazily by typed
   iterators.
3. **Tensor infos:** name, dimensions, ggml type and offset for each tensor.
   Byte size is computed from the type's block size and bytes per block.
4. **Layout check:** the tensor data blob starts at the next multiple of the
   alignment, tensors are packed in info order without gaps, and every
   tensor's data lies inside the input.

Every count is checked against the remaining bytes before allocating, so a
hostile file cannot make the parser allocate more than the input could
describe. The parser is written against llama.cpp's `gguf.cpp` rather than
the GGUF spec prose, and the ggml type table is copied from `ggml.h`
([issue #7](https://github.com/maltemindedal/scribe/issues/7)). The full rule
list is in the [GGUF parser reference](../reference/gguf-parser.md#validation-rules).

## Verification

Correctness is checked by tests, and performance is never measured in CI
([ADR 0002](decisions/0002-correctness-only-ci.md)). CI runs on x86-64 Linux
and arm64 macOS. Tests that need real model files are `#[ignore]` and run on
demand ([Running the model tests](../guides/running-the-model-tests.md)).

## Decisions

- [0001: Rust 1.99 as the minimum toolchain](decisions/0001-rust-1-99-toolchain-floor.md)
- [0002: CI checks correctness only](decisions/0002-correctness-only-ci.md)
- [0003: Map GGUF files with memmap2, and keep blocks align-1](decisions/0003-mmap-and-align-1-blocks.md)
