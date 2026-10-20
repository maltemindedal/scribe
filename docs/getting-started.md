# Getting started

This tutorial takes you from a fresh machine to a built Scribe workspace with
passing tests and a running `scribe` binary. It assumes you have used a
terminal before but not necessarily Rust.

Scribe is early in development. Today the workspace builds, the GGUF parser
is tested, and the `scribe` binary prints its help and version. It does not
load or run a model yet; the [roadmap](https://github.com/maltemindedal/scribe/issues/1)
lists what comes next.

## 1. Install the prerequisites

You need:

- **Git**, to clone the repository.
- **rustup**, the Rust toolchain installer, from <https://rustup.rs>.

You do not need to install a specific Rust version yourself. The repository
pins Rust 1.99 with `rustfmt` and `clippy` in
[`rust-toolchain.toml`](../rust-toolchain.toml), and rustup installs it in the
next step.

## 2. Clone the repository

```sh
git clone https://github.com/maltemindedal/scribe
cd scribe
```

## 3. Install the pinned toolchain

```sh
rustup toolchain install
rustup show active-toolchain
```

`rustup toolchain install` with no arguments reads `rust-toolchain.toml`.
The second command confirms which toolchain is active. On Linux x86-64 it
prints:

```text
1.99-x86_64-unknown-linux-gnu (overridden by '/path/to/scribe/rust-toolchain.toml')
```

The target triple differs on other platforms (for example
`aarch64-apple-darwin` on Apple silicon).

## 4. Build the workspace

```sh
cargo build --workspace
```

This compiles all ten crates under `crates/`. The first build downloads
`memmap2` and `libc`. `gguf` depends on `memmap2`, and `memmap2` depends
on `libc`.

## 5. Run the tests

```sh
cargo test --workspace
```

Every `test result:` line reports `ok`. Most crates have no tests yet; the
GGUF parser's tests do the work. Seven tests are reported as ignored. Cargo
may print them in a different order:

```text
test parses_qwen3_0_6b_q8_0 ... ignored, needs models/Qwen3-0.6B-Q8_0.gguf (see #5)
test tensor_bytes_over_every_qwen3_tensor_is_a_lookup ... ignored, needs models/Qwen3-0.6B-Q8_0.gguf (see #5)
test open_qwen3_does_not_resident_the_file ... ignored, needs models/Qwen3-0.6B-Q8_0.gguf (see #5)
test card_matches_committed_goldens ... ignored, needs the CI models under models/ or $SCRIBE_MODELS
test seeded_header_mutations_do_not_panic ... ignored, needs the CI models under models/ or $SCRIBE_MODELS
test stories260k_every_header_prefix_is_an_error ... ignored, needs the CI models under models/ or $SCRIBE_MODELS
test llama_cpp_logits_have_zero_kld_with_themselves_and_not_with_a_perturbed_copy ... ignored, needs models/reference/llama.cpp dumps (tools/llama.cpp/run.sh)
```

They need real model files. [Running the model tests](guides/running-the-model-tests.md)
shows how to run them.

## 6. Run the `scribe` binary

```sh
cargo run -p cli -- --help
```

Expected output:

```text
scribe: a from-scratch LLM inference engine

Usage: scribe <COMMAND>

No commands are implemented yet; see the roadmap at
https://github.com/maltemindedal/scribe/issues/1

Options:
  -h, --help     Print help
  -V, --version  Print version
```

The package is called `cli`; the binary it builds is called `scribe`. The
[CLI reference](reference/cli.md) lists every option.

## Next steps

- To change code, read [CONTRIBUTING.md](../CONTRIBUTING.md).
- To understand how the crates fit together, read the
  [architecture overview](architecture/overview.md).
