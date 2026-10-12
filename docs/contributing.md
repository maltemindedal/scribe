# Contributing

This page covers the development setup, the checks a change must pass, and
how work is organised. If you have not built Scribe yet, start with
[Getting started](getting-started.md).

## Development setup

Install rustup and run `rustup toolchain install` in the repository; it
installs the pinned toolchain with `rustfmt` and `clippy`. See
[Configuration](reference/configuration.md#toolchain) for the pinned values.

## Checks

CI runs these three commands on Linux x86-64 and macOS arm64 for every pull
request ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)). Run them
before opening a pull request:

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

Tests that need model files are ignored by default; see
[Running the model tests](guides/running-the-model-tests.md).

## Rules in the code

- Do not add `unsafe` outside `kernels` (and, once it exists, the mmap module
  in `gguf`). The other crates forbid it; see the
  [architecture overview](architecture/overview.md#boundaries-enforced-in-code).
- Never commit model files. `/models/` and `*.gguf` are gitignored.
- The workspace has no third-party dependencies yet. Each crate's doc comment
  states what it owns; keep code inside those boundaries.

## Finding work

The [roadmap issue](https://github.com/maltemindedal/scribe/issues/1) lists
every planned issue, grouped by milestone, with a suggested order. Issue
labels:

| Label | Meaning |
|---|---|
| `ready-for-agent` | Fully specified and ready to pick up. |
| `needs-human` | Needs the maintainer: hardware, files or a decision. |
| `M0` to `M9` | Milestone. |
| `stretch` | Planned for after M9. |
| `tracking` | Tracking issue, such as the roadmap. |

_As of 2026-10; source of truth: the label list in
[issue #1](https://github.com/maltemindedal/scribe/issues/1)._

## Pull requests

- Reference the issue the change closes, for example `Closes #7`.
- Pull requests are squash-merged into `main`, so each one lands as a single
  commit.

## Licence

Scribe is licensed under the [MIT licence](../LICENSE).
