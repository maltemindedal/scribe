# Scribe

Scribe is a from-scratch LLM inference engine in Rust, for people who want to
read, verify and contribute to every layer of one.

## Overview

The goal is to run Qwen3-0.6B Q8_0 on CPU (Apple silicon first, x86-64 too),
check every layer against reference implementations, and serve an
OpenAI-compatible API. Scribe is in early development: today it has the Cargo
workspace, CI, and a hardened parser for GGUF model files. Everything else is
planned on the [roadmap](https://github.com/maltemindedal/scribe/issues/1).

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

Every `test result:` line reports `ok`, with two tests ignored because
they need model files or reference dumps. Then run the binary:

```sh
cargo run -p cli -- --version
```

```text
scribe 0.1.0
```

[Getting started](docs/getting-started.md) walks through each step.

## Usage example

The `gguf` crate parses a GGUF file's header, metadata and tensor table:

```rust
use gguf::Gguf;

let bytes = std::fs::read("models/Qwen3-0.6B-Q8_0.gguf")?;
let g = Gguf::parse(&bytes)?;
let q = g.tensor("blk.0.attn_q.weight").unwrap();
println!("{:?} {:?}", q.ggml_type, q.dims()); // Q8_0 [1024, 2048]
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
