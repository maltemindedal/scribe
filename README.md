# Scribe

Scribe is a from-scratch LLM inference engine in Rust, for people who want to
read, verify and contribute to every layer of one.

## Overview

The goal is to run Qwen3-0.6B Q8_0 on CPU (Apple silicon first, x86-64 too),
check every layer against reference implementations, and serve an
OpenAI-compatible API. Scribe is in early development. Today it has the Cargo
workspace, CI, a GGUF parser that maps model files, and align-1 block layouts.
The rest of the engine is on the
[roadmap](https://github.com/maltemindedal/scribe/issues/1).

## Quick start

Prerequisites: Git and [rustup](https://rustup.rs). The repository pins Rust
1.99, and rustup installs it for you.

```sh
git clone https://github.com/maltemindedal/scribe
cd scribe
rustup toolchain install
cargo build --workspace
cargo test --workspace
```

Every `test result:` line reports `ok`, with seven tests ignored because
they need a model file or reference dumps. Then run the binary:

```sh
cargo run -p cli -- --version
```

```text
scribe 0.1.0
```

[Getting started](docs/getting-started.md) walks through each step.

## Usage example

The `gguf` crate maps a GGUF file and parses its header, metadata, and tensor table:

```rust
use gguf::GgufFile;
use kernels::BlockQ8_0;

// SAFETY: nothing truncates or writes the file while the mapping is alive.
// Truncation delivers SIGBUS on the next fault and kills the process. A write
// is undefined behaviour for the slices this returns.
let file = unsafe { GgufFile::open("models/Qwen3-0.6B-Q8_0.gguf")? };
let q = file.tensor_blocks::<BlockQ8_0>("blk.0.attn_q.weight")?;
println!("{} blocks", q.len());
```

The `scribe` binary has no commands yet; see the [CLI reference](docs/reference/cli.md).

## Documentation

| | |
|---|---|
| [Getting started](docs/getting-started.md) | Build, test and run Scribe locally. |
| [Running the model tests](docs/guides/running-the-model-tests.md) | Fetch a real model and run the ignored tests. |
| [Reference](docs/README.md#reference) | CLI, configuration and the GGUF parser API. |
| [Architecture](docs/architecture/overview.md) | The crates, their boundaries and design decisions. |
| [All docs](docs/README.md) | The full index. |

## Project structure

```text
.github/             CI workflows, issue and pull request templates
crates/              The ten workspace crates; see the architecture overview
docs/                Documentation
Cargo.toml           Workspace manifest
rust-toolchain.toml  Pinned Rust toolchain
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the checks to run and how work is organised, and the [Code of Conduct](CODE_OF_CONDUCT.md). Report vulnerabilities as described in [SECURITY.md](SECURITY.md).

## Licence

MIT, see [LICENSE](LICENSE).
