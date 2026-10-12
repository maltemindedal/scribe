# Research artifacts

Material from the research phase that planned Scribe, kept here for provenance. It
gives later issues a head start; none of it is part of the build. `research/` is excluded
from the Cargo workspace, so CI never compiles it and `cargo test --workspace` never runs it.

Everything below is the original, **unmodified** copy, except `tools/` (see the last
section) and `report.md`, whose private research-notes paths were rewritten to repo paths
and whose note about extracting the oracle into `tools/` was updated. Do not "fix" files in this folder; port what you need into `crates/` in the
issue that uses it.

| Path | What it is | Starting point for |
|---|---|---|
| [`report.md`](report.md) | The research report: target model, architecture, traps, milestones, and a verified-versus-estimated ledger. | All milestones |
| [`gguf_check.rs`](gguf_check.rs) | One std-only file (485 lines): hardened GGUF v3 parser, align-1 `repr(C)` block structs with `cast_blocks`, scalar dequantisers, manual f16/bf16 conversion, raw-libc mmap wrapper, mutation fuzz. | #8, #12 |
| [`qk_verified_crate/`](qk_verified_crate) | Package `qk`: block types for Q4_0, Q8_0, Q4_K, Q6_K and Q8_K, scalar reference dots, AVX2 and AVX-VNNI kernels, a NEON module, vector ops (`expf8`, softmax, RMSNorm, NEOX RoPE, f32·f16 dot), a kernel benchmark, a feature probe, a differential fuzz target, and a C transcription of llama.cpp's AVX2 loops. | #24, #29, #30, #31, #32 |

The Python tokenizer oracle is not in this folder; it was extracted to
[`tools/tokenizer_oracle.py`](../tools/tokenizer_oracle.py) (see below).

## Known caveats

Read these before reusing anything.

### `gguf_check.rs`

- **Does not compile under edition 2024.** The `extern "C"` block at line 385 must become
  `unsafe extern "C"` (error: "extern blocks must be unsafe"). It builds with `--edition 2021`.
- Unix only (Linux and macOS constants). No Windows path.
- Q8_K dequantisation is untested: `gguf-py` has no reference for it.
- No IQ, MXFP4 or ternary types; big-endian files are unsupported.
- It accepts non-consecutive tensor offsets, which the reference reader (`gguf.cpp`) rejects.
- It rejects invalid UTF-8 strings, which may be stricter than some community files need.
- Evidence: bit-exact against `gguf-py` on 25 real tensor samples and 10 randomised cases (F16, BF16,
  Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K); f16 conversion matched numpy on all 65,536 bit
  patterns; 3 x 3,000 mutated or truncated real headers produced no panic, hang or out-of-memory.

### `qk_verified_crate/`

- **All aarch64 and NEON code was cross-compiled and inspected but never executed.** That
  covers every NEON kernel, the `smmla` tile and the NEON vector ops. Miri cannot run it
  either. The first job on Apple silicon is to run the scalar-reference tests against
  `neon::*` (issue #30).
- There are no NEON Q4_K or Q6_K kernels and no complete i8mm dot kernel.
- Its manifest declares Rust 1.99 (`rust-version = "1.99"`, edition 2024), and the research
  ran its 12 tests on 1.99.0. They also passed on 1.97.0 with `--ignore-rust-version`, so
  1.99 is the declared floor rather than a proven hard requirement on x86; the aarch64
  `vdotq_s32` path does need 1.98 or newer.
- Block scales are stored as `u16` (alignment 2). `gguf_check.rs` stores them as `[u8; 2]`
  (alignment 1), which makes a `&[u8]` to `&[Block]` cast need only a length check. Unify
  on `[u8; 2]` when porting.
- The "C" baseline in `c_ref/` is a transcription of llama.cpp's AVX2 loops, not llama.cpp itself.
- All x86 timings came from a noisy 2-vCPU Xeon VM. Only same-run ratios mean anything;
  absolute numbers do not carry over to Apple silicon or bare-metal DDR5.
- The package has its own `Cargo.lock`. Build it from its own directory, for example
  `cd research/qk_verified_crate && cargo test --release` (12 tests pass on x86-64).
  `fuzz/` needs nightly and `cargo-fuzz`; `half_check/` is a standalone asm inspection.

## `tools/`: the tokenizer oracle

[`tools/tokenizer_oracle.py`](../tools/tokenizer_oracle.py) holds the byte-level BPE and
SentencePiece reference code from `tokenizer_and_models.md` (sections 2 and 3 of the research
notes, which are not in this repo). It is the executable spec for issues #14 to #18.

**It is not a verbatim copy.** The notes' `pre_split` snippet had no `return out` and so could
not have run as printed; it was added. The notes also describe the code as "exactly" what was
tested, but the snippet is a condensed rewrite of the scripts that produced the published
numbers. So the extracted file was re-checked when it was imported, with
[`tools/check_oracle.py`](../tools/check_oracle.py), against llama-cpp-python 0.3.36 (exact
llama.cpp commit unknown), vocabulary only, on 26 fixed strings plus 4,000 seeded random strings:

| Vocabulary | `--pre` | Result |
|---|---|---|
| Qwen3-0.6B | `qwen2` | 4026 / 4026 match |
| Llama-3.2-1B | `llama3` | 4026 / 4026 match |
| Qwen3.5-0.8B | `qwen35` | 4026 / 4026 match |
| Gemma-3-1B | `spm` | 4026 / 4026 match |

Limits of that evidence:

- The 8,028-string and exhaustive code-point-sweep figures in the report came from the original
  scripts, not from this file. This file's corpus is smaller and random.
- The corpus cannot tell the digit-grouping presets apart on Qwen3 (that vocabulary has no
  multi-digit tokens), nor `ignore_merges` on any vocabulary tested. The check does go red when
  the wrong setting is used on a vocabulary that has multi-digit tokens: Llama-3.2 with
  `--pre qwen2` fails 351 of 4,026.
- `Spm` has no constructor and no `encode()` (the notes give them as comments). `check_oracle.py`
  supplies the missing special-token partition: Gemma-3 stores runs of spaces and newlines as
  USER_DEFINED tokens that llama.cpp splits out before SentencePiece runs. Only Gemma-3 was tested
  (`add_space_prefix=False`); the Llama-2 dummy-prefix branch is untested.
- Hugging Face `tokenizers` differs from llama.cpp on Qwen vocabularies (NFC normalisation), so
  llama.cpp is the oracle, not HF.
- SmolLM2's `smollm` pre-tokenizer is not covered.

To rerun: `pip install llama-cpp-python`, then
`python3 tools/check_oracle.py --gguf models/Qwen3-0.6B-Q8_0.gguf --pre qwen2`.
