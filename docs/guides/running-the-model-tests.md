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
  CI runs this test with
  `cargo test -p gguf --test conformance --profile mutation -- --ignored --exact seeded_header_mutations_do_not_panic`.
  `verify.sh model-test` adds `--nocapture` to that command.
  `[profile.mutation]` inherits release and turns overflow checks and debug
  assertions back on. The other ignored tests stay on the debug profile.
  On the macOS runner, a debug run of the conformance test binary took 45 to
  47 seconds.
- `stories260k_every_header_prefix_is_an_error` in the same file checks that
  `stories260K.gguf` parses, and that every prefix of its header, including
  the header itself, does not. The test fails before that loop when the
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
and valid. An existing file that fails the size check or the SHA-256 check
is left in place. The message includes the expected value and the actual
value. Delete that file and re-run. Downloads use `curl` 7.71 or later, which
must be on `PATH`. When stderr is a terminal, curl shows a progress bar.
Curl retries a failed download three times. If the download still fails, or
the new bytes fail verification, `models/` has no partial file.

## 2. Run the ignored tests

```sh
cargo test -p gguf -- --ignored --skip seeded_header_mutations_do_not_panic --exact
cargo test -p gguf --test conformance --profile mutation -- --ignored --exact seeded_header_mutations_do_not_panic
```

`--ignored` runs only the ignored tests. The second command uses the mutation
profile named above. Expected result, across both commands:

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
SCRIBE_MODELS=/path/to/models cargo test -p gguf -- --ignored --skip seeded_header_mutations_do_not_panic --exact
SCRIBE_MODELS=/path/to/models cargo test -p gguf --test conformance --profile mutation -- --ignored --exact seeded_header_mutations_do_not_panic
```

See [Configuration](../reference/configuration.md#scribe_models) for how the
path is resolved.

## 4. Fuzz the parser for ten minutes

`fuzz/` is not a workspace member. It needs a nightly toolchain,
`cargo-fuzz`, and `g++`. It does not change the stable pin in
`rust-toolchain.toml`. `fuzz/run.sh` exits if `g++` is not on `PATH`.
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
