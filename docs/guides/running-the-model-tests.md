# Running the model tests

Some tests check Scribe against real model files. They are marked
`#[ignore]`, so `cargo test --workspace` skips them. This guide shows how to
fetch the models they need and run them.

Today there is one such test, `parses_qwen3_0_6b_q8_0` in
[`crates/gguf/tests/qwen3.rs`](../../crates/gguf/tests/qwen3.rs). It parses
Qwen3-0.6B-Q8_0 and checks its tensor count, metadata count, vocabulary size
and a set of tensor shapes.

## 1. Download the models

Model files never go in the repository: `/models/` and `*.gguf` are listed in
[`.gitignore`](../../.gitignore). The `xtask` crate downloads the files pinned
in [`crates/xtask/models.toml`](../../crates/xtask/models.toml) into `models/`
and verifies each file's size and SHA-256:

```sh
cargo xtask models fetch --set ci
```

`--set ci` (the default) fetches the four GGUF files CI uses, about 810 MB in
total. `--set all` adds the large reference models; see
[MODELS.md](../../MODELS.md). Re-running is a no-op for files that are present
and valid. A file that fails verification is reported with the expected and
actual hash and is left in place; delete it and re-run. Downloads use `curl`
7.71 or later, which must be on `PATH`; in a terminal it shows a progress
bar. A failed download is retried three times; if it still fails, no partial
file is left in `models/`.

## 2. Run the ignored tests

```sh
cargo test -p gguf -- --ignored
```

`--ignored` runs only the ignored tests. Expected result:

```text
test parses_qwen3_0_6b_q8_0 ... ok
```

## Keep models outside the repository

To keep model files somewhere else, for example a shared download directory,
point `SCRIBE_MODELS` at that directory:

```sh
SCRIBE_MODELS=/path/to/models cargo test -p gguf -- --ignored
```

See [Configuration](../reference/configuration.md#scribe_models) for how the
path is resolved.
