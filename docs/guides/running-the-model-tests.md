# Running the model tests

Some tests check Scribe against real model files. They are marked
`#[ignore]`, so `cargo test --workspace` skips them. This guide shows how to
fetch the models they need and run them.

Today there are seven such tests:

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
- `card_matches_committed_goldens` in
  [`crates/gguf/tests/conformance.rs`](../../crates/gguf/tests/conformance.rs)
  compares `card::to_json` with the four files in
  `crates/gguf/tests/golden/`.
- `seeded_header_mutations_do_not_panic` in the same file copies each CI
  model's header and runs 3000 seeded mutations. About half of the flipped
  bytes land in the tensor-info span, from the first tensor info through
  `tensor_data_offset`. The rest stay inside the first 4096 bytes. It checks
  that `Gguf::parse` returns. It does not check how many of those buffers parse.
- `stories260k_every_header_prefix_is_an_error` in the same file checks that
  `stories260K.gguf` parses, and that every prefix of its header, including
  the header itself, does not. The test stops before that loop when the
  header is longer than 64 KiB.
- `llama_cpp_logits_have_zero_kld_with_themselves_and_not_with_a_perturbed_copy`
  in [`crates/bench/tests/kld.rs`](../../crates/bench/tests/kld.rs) needs the
  llama.cpp reference dumps for stories15M-q8_0 and Qwen3-0.6B-Q8_0. Write
  them first with `tools/llama.cpp/run.sh` (see
  [its README](../../tools/llama.cpp/README.md)), then run
  `cargo test -p bench -- --ignored`. CI does not build llama.cpp, so that
  step skips `llama_cpp_logits_have_zero_kld_with_themselves_and_not_with_a_perturbed_copy`
  and passes `--exact`.

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
test card_matches_committed_goldens ... ok
test seeded_header_mutations_do_not_panic ... ok
test stories260k_every_header_prefix_is_an_error ... ok
test parses_qwen3_0_6b_q8_0 ... ok
test tensor_bytes_over_every_qwen3_tensor_is_a_lookup ... ok
test open_qwen3_does_not_resident_the_file ... ok
```

The lines above are the ignored `gguf` tests. Cargo may print them in a
different order.

## 3. Regenerate the header-card goldens

Install `gguf==0.19.0` from
[`tools/requirements-gguf.txt`](../../tools/requirements-gguf.txt). From the
repository root, rewrite one golden:

```sh
python3 tools/gguf_dump.py models/stories260K.gguf \
  --out crates/gguf/tests/golden/stories260K.json
```

Repeat for `stories15M-q8_0`, `SmolLM2-135M-Instruct-Q8_0`, and
`Qwen3-0.6B-Q8_0`. The script checks `importlib.metadata.version("gguf")`
and exits 2 when the version is not 0.19.0. It writes `PATH.tmp` beside the
golden and replaces the golden only after `gguf-card accept` exits 0.
Running it again on the same file writes the same bytes. The Rust tests do
not run Python.

`verify.sh conformance` runs that script for each of the four models and
compares the output to the committed golden. It is not part of
`verify.sh all`. Without `gguf==0.19.0` importable, that step prints `SKIP`.
A skip does not verify the goldens.

## Keep models outside the repository

To keep model files somewhere else, for example a shared download directory,
point `SCRIBE_MODELS` at that directory:

```sh
SCRIBE_MODELS=/path/to/models cargo test -p gguf -- --ignored
```

See [Configuration](../reference/configuration.md#scribe_models) for how the
path is resolved.

## 4. Fuzz the parser for ten minutes

`fuzz/` is not a workspace member. It needs a nightly toolchain and
`cargo-fuzz`, and it does not change the stable pin in `rust-toolchain.toml`.
From the repository root, after the CI models are on disk:

```sh
cargo +nightly install cargo-fuzz
bash fuzz/run.sh
```

The script copies each CI header into `fuzz/corpus/parse/` and runs

```sh
cargo +nightly fuzz run parse -- -rss_limit_mb=512 -timeout=10 -max_total_time=600
```

`-rss_limit_mb=512` stops the process if it passes 512 MiB. `-timeout=10`
stops an input that runs longer than 10 seconds. `-max_total_time=600` stops
the run after 600 seconds. A panic fails the run. CI does not run this
script. The deterministic mutation test in section 2 is the CI check.
