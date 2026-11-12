# Architecture overview

Scribe is a Cargo workspace of eleven crates. Each crate owns one layer of an
LLM inference engine. Today the GGUF parser, the read-only file mapping, and
the align-1 block structs are implemented. `xtask` fetches pinned models, and
`bench` compares reference dumps. The `scribe` binary prints help and version.
The tokenizer, model, sampler, engine, and server crates are stubs whose
boundaries are fixed in their manifests and crate docs.

The project goal, milestones and build order live in the
[roadmap](https://github.com/maltemindedal/scribe/issues/1).

## Crates

| Crate | Owns | Status |
|---|---|---|
| `gguf` | GGUF file parsing, mmap, tensor views | Header parser, read-only private mapping (through `mapfile`), and per-tensor byte and block views ([reference](../reference/gguf-parser.md)). |
| `mapfile` | Read-only private file mapping; the only `mmap` call | `Mapped::open`, the advise and lock calls. One `unsafe` block. |
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

`gguf` depends on `kernels` and on `mapfile`. `mapfile` depends on `memmap2`
(which depends on `libc`) and on no workspace crate. No other crate depends
on another, and those two crates.io packages are the only third-party
dependencies. `gguf`'s tests also enable the `kernels` feature `testing`,
which adds a `Block` whose id is not a ggml type. See
[ADR 0003](decisions/0003-mmap-and-align-1-blocks.md) and
[ADR 0004](decisions/0004-mapfile-and-safe-open.md).

## Boundaries enforced in code

Some rules are already written into the crates, ahead of the code they
constrain:

- **`unsafe` is confined.** `unsafe` is allowed in `kernels` and `mapfile`
  and nowhere else. The workspace lint `unsafe_code = "forbid"` applies to
  every crate whose manifest has `[lints] workspace = true`, including that
  crate's binaries and tests. `forbid` cannot be relaxed, so
  `#[allow(unsafe_code)]` in those crates is error E0453. Each of those
  crates also has `#![forbid(unsafe_code)]`.
- **The two exceptions cannot drift.** Cargo cannot override one key of an
  inherited lint table, so `kernels` and `mapfile` copy the workspace tables
  with `unsafe_code` left out. A manifest with no `[lints]` table gets no
  workspace lints and no diagnostic.
  [`crates/xtask/tests/lint_policy.rs`](../../crates/xtask/tests/lint_policy.rs)
  fails when a member neither inherits the workspace lints nor is one of the
  two crates with matching tables.
- **`mapfile` cannot see parsed types.** It depends on no workspace crate.
  A `Gguf` that borrows the mapping cannot be stored next to it there, and
  `gguf` cannot extend the borrow, because that needs `unsafe`
  ([ADR 0004](decisions/0004-mapfile-and-safe-open.md)).
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
4. **Layout check:** when the file has tensors, the tensor data blob starts
   at the next multiple of the alignment. Tensors are packed in info order
   without gaps, and every tensor's data lies inside the input.

Every count is checked against the remaining bytes before allocating, so a
hostile file cannot make the parser allocate more than the input could
describe. The parser is written against llama.cpp's `gguf.cpp` rather than
the GGUF spec prose, and the ggml type table is copied from `ggml.h`
([issue #7](https://github.com/maltemindedal/scribe/issues/7)). The full rule
list is in the [GGUF parser reference](../reference/gguf-parser.md#validation-rules).

## Verification

Correctness is checked by tests, and performance is never measured in CI
([ADR 0002](decisions/0002-correctness-only-ci.md)). CI runs on x86-64 Linux
and arm64 macOS. Tests that need real model files are `#[ignore]`. CI fetches
the pinned models and runs those tests, and skips the llama.cpp dump test,
which still runs locally
([Running the model tests](../guides/running-the-model-tests.md)).

`card_matches_committed_goldens` compares `card::to_json` with the four files
in `crates/gguf/tests/golden/`. `tools/gguf_dump.py` regenerates one golden
from a model file. `fuzz/` at the repository root is a cargo-fuzz target for
`Gguf::parse`. The workspace `exclude` list contains `fuzz` and `research`.
CI does not build either one.

## Decisions

- [0001: Rust 1.99 as the minimum toolchain](decisions/0001-rust-1-99-toolchain-floor.md)
- [0002: CI checks correctness only](decisions/0002-correctness-only-ci.md)
- [0003: Map GGUF files with memmap2, and keep blocks align-1](decisions/0003-mmap-and-align-1-blocks.md)
- [0004: Confine mmap to mapfile, and make open a safe function](decisions/0004-mapfile-and-safe-open.md)
