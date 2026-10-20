# Running the model tests

Some tests check Scribe against real model files. They are marked
`#[ignore]`, so `cargo test --workspace` skips them. This guide shows how to
fetch the models they need and run them.

Today there are four such tests:

- `parses_qwen3_0_6b_q8_0` in
  [`crates/gguf/tests/qwen3.rs`](../../crates/gguf/tests/qwen3.rs) parses
  Qwen3-0.6B-Q8_0 and checks its tensor count, metadata count, vocabulary
  size and a set of tensor shapes.
- `tensor_bytes_over_every_qwen3_tensor_is_a_lookup` in the same file opens
  the model and checks that a byte lookup of every tensor matches the lengths
  in the tensor table.
- `open_qwen3_does_not_resident_the_file` in
  [`crates/gguf/tests/resident.rs`](../../crates/gguf/tests/resident.rs) maps
  the same file and checks that the process resident set stays far below the
  file size until a tensor is touched. It also checks that `output.weight`
  is absent and that `output_weight` returns the `token_embd.weight` bytes.
- `llama_cpp_logits_have_zero_kld_with_themselves_and_not_with_a_perturbed_copy`
  in [`crates/bench/tests/kld.rs`](../../crates/bench/tests/kld.rs) needs the
  llama.cpp reference dumps for stories15M-q8_0 and Qwen3-0.6B-Q8_0. Write
  them first with `tools/llama.cpp/run.sh` (see
  [its README](../../tools/llama.cpp/README.md)), then run
  `cargo test -p bench -- --ignored`. CI does not build llama.cpp, so it
  skips this test with `--skip llama_cpp_`.

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
test tensor_bytes_over_every_qwen3_tensor_is_a_lookup ... ok
test open_qwen3_does_not_resident_the_file ... ok
```

## Keep models outside the repository

To keep model files somewhere else, for example a shared download directory,
point `SCRIBE_MODELS` at that directory:

```sh
SCRIBE_MODELS=/path/to/models cargo test -p gguf -- --ignored
```

See [Configuration](../reference/configuration.md#scribe_models) for how the
path is resolved.
