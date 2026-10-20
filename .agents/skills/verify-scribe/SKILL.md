---
name: verify-scribe
description: Drive Scribe (Rust GGUF inference engine) the way a contributor or API client does and capture proof. Covers the workspace build and tests, the `scribe` CLI, `cargo xtask models fetch`, and the ignored real-model tests, including header-card goldens. `verify.sh conformance` compares those goldens to `tools/gguf_dump.py` when gguf 0.19.0 is installed, and is not part of `all`. The OpenAI-compatible HTTP server (#36) is a stub that reports SKIP until it lands. Use before claiming a Scribe change works.
---

# Verify Scribe

Scribe is a Cargo workspace (`crates/*`, Rust 1.99 pinned in `rust-toolchain.toml`). Today a user touches three surfaces: `cargo build`/`cargo test`, the `scribe` binary (`crates/cli`), and `cargo xtask models fetch`. The HTTP server in `crates/server` is an empty crate until issue #36 lands. `research/` and `fuzz/` are excluded from the workspace. This skill does not run the fuzzer.

Everything goes through one helper, run from anywhere in the checkout:

```sh
.agents/skills/verify-scribe/scripts/verify.sh <doctor|build|test|cli|models|model-test|conformance|reference|server|cleanup|all>
```

Each step prints `PASS`/`FAIL`/`SKIP` lines and ends with `RESULT: PASS`, `RESULT: PASS with SKIP` or `RESULT: FAIL`. The exit code is 0 unless something failed, and a SKIP still means that feature is unverified.

## Launch

Nothing stays running. Scribe is a short-lived CLI today, so launch means building once: `verify.sh build` runs `cargo build --workspace` and is ready when it exits 0. The binary is `target/debug/scribe`. Each CLI drive is a fresh process, so there is no shared state to isolate. Teardown is `verify.sh cleanup`.

Models are the one piece of state. They live in `models/` (gitignored) or in `$SCRIBE_MODELS`. Point `SCRIBE_MODELS` at a scratch directory when a run must not reuse or touch the shared cache.

## Doctor

`verify.sh doctor` is read-only. It prints the artifacts directory, the commit and branch, the arch, and the rustc version. It requires rustc 1.99 or newer and checks for `curl` (`models fetch` needs 7.71 or newer). It also counts the files in the models directory. Run it first, and again whenever a result looks wrong.

## Drive

- `verify.sh test` runs `cargo test --workspace --no-fail-fast` and requires a zero exit and every `test result:` line to be `ok`. Seven tests are ignored by design: `parses_qwen3_0_6b_q8_0`, `tensor_bytes_over_every_qwen3_tensor_is_a_lookup`, `open_qwen3_does_not_resident_the_file`, `card_matches_committed_goldens`, `seeded_header_mutations_do_not_panic`, `stories260k_every_header_prefix_is_an_error`, and the llama.cpp KLD test in `crates/bench/tests/kld.rs`.
- `verify.sh cli` builds, then drives `scribe --version`, `-V`, `--help`, `-h`, no arguments, and `scribe foo`. It asserts the stdout, stderr and exit codes documented in `docs/reference/cli.md`.
- `verify.sh models` runs `cargo xtask models fetch --set ${SCRIBE_VERIFY_SET:-ci}`. This downloads about 810 MB the first time. Every `checking` line must have a matching `ok` or `fetched` line, and no `FAILED` line may appear.
- `verify.sh model-test` runs `cargo test -p gguf -- --ignored --nocapture`, skipping `seeded_header_mutations_do_not_panic`, then runs that test again with `cargo test -p gguf --test conformance --profile mutation`. It requires `test parses_qwen3_0_6b_q8_0 ... ok`, `test open_qwen3_does_not_resident_the_file ... ok`, `test card_matches_committed_goldens ... ok`, `test seeded_header_mutations_do_not_panic ... ok`, and `test stories260k_every_header_prefix_is_an_error ... ok`. The resident-set test prints a `rss ` line with VmRSS and RssFile before open, after open, and after touching one tensor. The step then reruns that test with `SCRIBE_EAGER_READ=1` and requires a non-zero exit. That run opens with `populate: true`, which on Linux prefaults the file inside `open` and fails the resident-set check. Run `models` first. The four CI GGUF files must be present.
- `verify.sh conformance` runs `tools/gguf_dump.py --out` for each of the four CI models and `cmp`s the result to `crates/gguf/tests/golden/<stem>.json`. It is not part of `all`. It prints `SKIP` when `python3` cannot import `gguf` at version `0.19.0`. That `SKIP` does not verify the goldens. Run `models` first.
- `verify.sh reference` runs `tools/llama.cpp/run.sh`, which builds llama.cpp at `tools/llama.cpp/PIN` and dumps stories15M-q8_0 and Qwen3-0.6B-Q8_0. It checks that llama-perplexity's self KLD is within 1e-5 of 0 for both models (it is float noise around 0, measured 0.000001 ± 0.000001), and that each 17-token dump records `-nr`, `system_info`, the pinned commit and the cmake flags in `BUILD_INFO`. It then runs `cargo run -p bench --bin compare`. The logits compared with themselves must give `check mean_kld <= 0e0: PASS`, and with a 1e-3 perturbed copy they must give `FAIL` and exit 1. Last, it reruns the Qwen3 17-token dump and requires the same `SHA256SUMS`. It SKIPs without `git`, `cmake` and `c++`, or without both models. Run `models` first. The first run fetches and builds llama.cpp, which took 105 s in total on an 8-core box. Later runs take about 20 s.
- `verify.sh server` is a stub. It prints `SKIP` while `crates/server` is empty and `scribe` has no `serve` command. Once either exists, the stub turns into `FAIL` so the next agent replaces it with a real launch and drive (see `features/openai-server.md`).
- `verify.sh all` runs doctor, test, cli, server and cleanup. It stops after doctor if doctor fails. It leaves out the network-heavy `models` and `model-test` steps, and it does not run `conformance`.

Feature recipes are in [`features/README.md`](features/README.md).

## Evidence

Artifacts go to `$SCRIBE_VERIFY_ARTIFACTS`, default `/tmp/scribe-verify/<run-id>`. Set `SCRIBE_VERIFY_RUN=<id>` to keep several steps in one directory. Each drive writes `<name>.cmd`, `<name>.stdout`, `<name>.stderr` and `<name>.exit`. `test.summary` collects the `test result:` lines.

Proof standards:

- Drive the real binary and the real `cargo xtask` entry point, never internal functions.
- Capture the command and its exit code, not just the output.
- For `models`, check the side effect too: `models-ls.txt` lists the files and sizes on disk.
- A `SKIP` is never a pass for that feature. Report it as unverified.

## Cleanup

`verify.sh cleanup` kills only the PIDs this run wrote to `$ART/*.pid` (none today), never anything by name. It never deletes the artifacts directory. Downloaded models are kept on purpose because they are a pinned, verified cache. Delete a scratch `SCRIBE_MODELS` directory yourself if you created one.

## Helpers

- `scripts/verify.sh` is the only helper. Its usage is shown above.
