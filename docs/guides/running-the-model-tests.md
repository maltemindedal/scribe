# Running the model tests

Some tests check Scribe against real model files. They are marked
`#[ignore]`, so `cargo test --workspace` skips them. This guide shows how to
fetch the model they need and run them.

Today there is one such test, `parses_qwen3_0_6b_q8_0` in
[`crates/gguf/tests/qwen3.rs`](../../crates/gguf/tests/qwen3.rs). It parses
Qwen3-0.6B-Q8_0 and checks its tensor count, metadata count, vocabulary size
and a set of tensor shapes.

## 1. Download the model

Model files never go in the repository: `/models/` and `*.gguf` are listed in
[`.gitignore`](../../.gitignore). Download the file into `models/` at the
repository root:

```sh
mkdir -p models
curl -fL -o models/Qwen3-0.6B-Q8_0.gguf \
  https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf
```

The URL pins the Hugging Face revision the test was written against. The
file is 639,446,688 bytes.

_As of 2026-10; source of truth: the Hugging Face revision recorded in
[PR #48](https://github.com/maltemindedal/scribe/pull/48) and the model table
in [issue #5](https://github.com/maltemindedal/scribe/issues/5)._

A model downloader with checksums (`cargo xtask models fetch`) is planned in
[issue #5](https://github.com/maltemindedal/scribe/issues/5) and will replace
this manual step.

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
