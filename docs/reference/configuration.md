# Configuration reference

Scribe has no runtime configuration yet: the `scribe` binary reads no
environment variables or config files. This page lists the settings that
affect building and testing.

## Environment variables

| Variable | Used by | Type | Default | Effect |
|---|---|---|---|---|
| `SCRIBE_MODELS` | Model tests ([`crates/gguf/tests/qwen3.rs`](../../crates/gguf/tests/qwen3.rs), [`crates/gguf/tests/resident.rs`](../../crates/gguf/tests/resident.rs), [`crates/gguf/tests/conformance.rs`](../../crates/gguf/tests/conformance.rs)), `cargo xtask models fetch`, `fuzz/run.sh` | Directory path | `models/` at the repository root | Directory the ignored model tests load model files from, `cargo xtask models fetch` downloads them into, and `fuzz/run.sh` reads CI headers from. |
| `SCRIBE_EAGER_READ` | [`crates/gguf/tests/resident.rs`](../../crates/gguf/tests/resident.rs) | Present or absent | Absent | When set to any value, the resident-set test opens the model with `MapOptions { populate: true }`. On Linux and Android that prefaults the file inside `open` and the test fails. `verify.sh model-test` runs the test once with the variable unset, then sets it for one extra run that must fail. |

### `SCRIBE_MODELS`

When set, model files are looked up as `$SCRIBE_MODELS/<file>`. When unset,
they are looked up in `crates/gguf/../../models/`, which is `models/` at the
repository root. A relative value is resolved against the test's working
directory, which Cargo sets to the crate directory (`crates/gguf/`), so use
an absolute path.

### `SCRIBE_EAGER_READ`

Used only by `open_qwen3_does_not_resident_the_file`. Any value, including an
empty one, makes that test call `GgufFile::open_with` with `populate: true`.
On Linux and Android, `MAP_POPULATE` prefaults the file during `open` and the resident-set
assertion fails. On other operating systems the flag is ignored, so the test
then reads the mapping `open` returned, which fails the same assertion. Leave
the variable unset to run the test for real. `verify.sh model-test` sets it
for one extra run and requires that run to fail.

### CI-only variables

The CI workflow ([`.github/workflows/ci.yml`](../../.github/workflows/ci.yml))
sets these for every job. Set them locally to reproduce CI exactly.

| Variable | CI value | Effect |
|---|---|---|
| `RUSTFLAGS` | `-D warnings` | Compiler warnings fail the build. |
| `CARGO_TERM_COLOR` | `always` | Coloured Cargo output in logs. |

## Toolchain

[`rust-toolchain.toml`](../../rust-toolchain.toml) pins the channel and
components rustup installs:

| Key | Value |
|---|---|
| `channel` | `1.99` |
| `components` | `rustfmt`, `clippy` |

## Workspace settings

Set in the root [`Cargo.toml`](../../Cargo.toml) and inherited by every crate.

| Setting | Value |
|---|---|
| `edition` | `2024` |
| `resolver` | `3` |
| `rust-version` | `1.99` |
| `license` | `MIT` |
| `publish` | `false` |
| `members` | `crates/*` |
| `exclude` | `research`, `fuzz` (`research/` holds imported artifacts, `fuzz/` is the cargo-fuzz harness; CI builds neither) |
| Lint `unsafe_op_in_unsafe_fn` | `deny` |
| Lint group `clippy::all` | `warn` |
| `[profile.release]` `lto` | `"fat"` |
| `[profile.release]` `codegen-units` | `1` |
| `[profile.mutation]` | inherits `release`, with `overflow-checks = true` and `debug-assertions = true`. CI and `verify.sh model-test` build `seeded_header_mutations_do_not_panic` with `--profile mutation`. |
| `[profile.dev.package.xtask]` `opt-level` | `3` (the xtask hashes multi-gigabyte files) |

[`.cargo/config.toml`](../../.cargo/config.toml) defines the alias
`cargo xtask` = `cargo run --package xtask --`.
