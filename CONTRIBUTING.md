# Contributing to Scribe

Documentation lives in [`docs/`](docs/README.md); the
[architecture overview](docs/architecture/overview.md) and the
[decision records](docs/architecture/decisions/) explain why the code is
shaped the way it is.

Everyone taking part in Scribe is expected to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

Install Git and [rustup](https://rustup.rs), then clone and build:

```sh
git clone https://github.com/maltemindedal/scribe
cd scribe
rustup toolchain install
cargo build --workspace
```

`rustup toolchain install` with no arguments installs the toolchain pinned in
[`rust-toolchain.toml`](rust-toolchain.toml) (Rust 1.99 with `rustfmt` and
`clippy`). `gguf` depends on `kernels` and on `memmap2`. The first build
downloads `memmap2` and `libc`.
[Getting started](docs/getting-started.md) walks through each step with the
expected output.

## Verification commands

CI runs these on `ubuntu-latest` (x86-64) and `macos-latest` (arm64) for every
pull request and every push to `main`
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)). Run them before
opening a pull request:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

CI also sets `RUSTFLAGS=-D warnings`, so any compiler warning fails the
build. To reproduce CI exactly:

```sh
RUSTFLAGS="-D warnings" cargo test --workspace
```

Tests that need model files are `#[ignore]`, so `cargo test` skips them. See
[Running the model tests](docs/guides/running-the-model-tests.md).

CI never measures performance
([ADR 0002](docs/architecture/decisions/0002-correctness-only-ci.md)).

## Test layout

| Location | Scope |
| --- | --- |
| `crates/*/src/**` `#[cfg(test)]` modules | Unit tests next to the code under test |
| `crates/*/tests/` | Integration tests against the crate's public API |
| `#[ignore]` tests | Tests that need real model files from `models/` |

The GGUF parser's strictness tests in
[`crates/gguf/tests/strictness.rs`](crates/gguf/tests/strictness.rs) build a
valid file and change exactly one thing, so each test proves which rule
rejects it. Follow the same pattern for new validation rules.

## Conventions

- **`unsafe` stays confined.** Do not add `unsafe` outside `kernels` and the
  mmap module in `gguf`. The other crates forbid it; see
  the [architecture overview](docs/architecture/overview.md#boundaries-enforced-in-code).
- **Keep crate boundaries.** Each crate's doc comment states what it owns;
  keep code inside those boundaries.
- **No model files in git.** `/models/` and `*.gguf` are gitignored.
- **The code is the source of truth for docs.** When documentation and
  behaviour disagree, fix the documentation. Document only what is built;
  link planned work to the roadmap.

## Issues and pull requests

The [roadmap issue](https://github.com/maltemindedal/scribe/issues/1) lists
every planned issue, grouped by milestone, with a suggested order. Issue
labels:

| Label | Meaning |
| --- | --- |
| `ready-for-agent` | Fully specified and ready to pick up. |
| `needs-human` | Needs the maintainer: hardware, files or a decision. |
| `M0` to `M9` | Milestone. |
| `stretch` | Planned for after M9. |
| `tracking` | Tracking issue, such as the roadmap. |

_As of 2026-10; source of truth: the label list in
[issue #1](https://github.com/maltemindedal/scribe/issues/1)._

Pull requests follow the
[pull request template](.github/pull_request_template.md): say what changed
and why, link the issue with `Closes #N`, and tick the verification commands.
There is no fixed merge method. Each pull request chooses a merge commit,
a squash, or a rebase. Squash a single-purpose pull request whose commits
are noisy. Use a merge commit when its separate commits are worth keeping.
The merge-ready handoff states which method it recommends and why.

Report security issues privately, as described in [SECURITY.md](SECURITY.md).

## Licence

Scribe is licensed under the [MIT licence](LICENSE).
