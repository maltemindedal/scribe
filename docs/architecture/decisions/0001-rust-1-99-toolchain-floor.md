# 0001: Rust 1.99 as the minimum toolchain

- **Status:** Accepted
- **Source:** [issue #2](https://github.com/maltemindedal/scribe/issues/2),
  implemented in [PR #46](https://github.com/maltemindedal/scribe/pull/46)

## Context

Scribe's planned kernels use SIMD intrinsics and floating-point operations
that only recent stable Rust provides.

## Decision

The workspace sets `rust-version = "1.99"` with edition 2024 and resolver 3 in
[`Cargo.toml`](../../../Cargo.toml), and
[`rust-toolchain.toml`](../../../rust-toolchain.toml) pins the `1.99` channel
with `rustfmt` and `clippy`.

Issue #2 gives the reason: 1.99 is needed for `vdotq_s32`, the `algebraic_*`
float operations, and AVX-512/VNNI intrinsics.

## Consequences

- Contributors and CI use exactly the pinned toolchain; rustup installs it
  automatically.
- Older toolchains cannot build the workspace.
