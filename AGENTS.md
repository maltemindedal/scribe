# AGENTS.md

Scribe is a from-scratch LLM inference engine in Rust, built in milestones from
the [roadmap issue #1](https://github.com/maltemindedal/scribe/issues/1). Today
it has a ten-crate workspace, CI, and a hardened GGUF parser in `crates/gguf`;
the other crates are stubs. A change has to keep every check green on both
x86-64 Linux and arm64 macOS, keep `unsafe` where it is allowed, and leave the
docs matching the code.

## Commands

Run these from the repository root before every PR. They mirror
`.github/workflows/ci.yml`, which also sets `RUSTFLAGS=-D warnings`:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTFLAGS="-D warnings" cargo test --workspace
```

- The toolchain is pinned to Rust 1.99 in `rust-toolchain.toml`. If `cargo`
  reports an older version, run `rustup toolchain install` in the repo.
- Single test: `cargo test -p gguf --test strictness <name>`.
- Model tests are `#[ignore]` and need `models/Qwen3-0.6B-Q8_0.gguf` (or
  `SCRIBE_MODELS` pointing at a directory holding it):
  `cargo test -p gguf -- --ignored`. `docs/guides/running-the-model-tests.md`
  has the pinned download URL.

## Conventions

- `unsafe` only in `kernels`, and later in the planned mmap module of `gguf`.
  Every other crate has `#![forbid(unsafe_code)]`; `gguf` has
  `#![deny(unsafe_code)]` so that module can opt back in. The workspace denies
  `unsafe_op_in_unsafe_fn`.
- No third-party dependencies yet. `engine` has no HTTP and no JSON; `server`
  is the only crate allowed `serde_json`.
- Never commit model files (`/models/`, `*.gguf` are gitignored).
- Every check must be shown to be able to fail: for a new validation rule, add
  a test in the style of `crates/gguf/tests/strictness.rs` (a valid file
  changed in exactly one place) and confirm it goes red with the rule disabled.
- Pull requests use `.github/pull_request_template.md`, say `Closes #N`, and
  are squash-merged.

## Docs

The code is the source of truth: when docs and behaviour disagree, fix the
docs, in the same PR as the code. Document only what is built; link planned
work to the roadmap.

- Adding a `scribe` command or option: `docs/reference/cli.md`.
- Adding an environment variable or changing toolchain or workspace settings:
  `docs/reference/configuration.md`.
- Changing the `gguf` API, a validation rule or an `ErrorKind`:
  `docs/reference/gguf-parser.md`.
- Filling a stub crate or adding a dependency between crates:
  `docs/architecture/overview.md`.
- A decision with a written rationale gets a record in
  `docs/architecture/decisions/`, in the shape of the existing ones.
- Adding a doc: give it a row in `docs/README.md`.

## Agent skills

Project skills live in `.agents/skills/`.

### Verification

`.agents/skills/verify-scribe/` drives the build, tests, CLI and model fetch and
saves evidence under `/tmp/scribe-verify/`. Run
`.agents/skills/verify-scribe/scripts/verify.sh all` before claiming a change
works.

### Issue tracker

Issues are tracked in GitHub Issues for `maltemindedal/scribe`. The roadmap,
#1, groups every issue by milestone and gives a suggested order.

### Labels

`ready-for-agent` (fully specified), `needs-human` (needs the maintainer's
hardware, files or a decision), `M0` to `M9` (milestones), `stretch`,
`tracking`, plus the triage labels `needs-triage`, `needs-info`,
`ready-for-human` and `wontfix`.
