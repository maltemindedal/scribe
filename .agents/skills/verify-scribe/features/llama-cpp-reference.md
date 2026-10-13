# llama.cpp reference dumps

A contributor builds llama.cpp at the pinned commit and writes activation and logit dumps for fixed token ids. They then compare a dump against another one by NMSE and KL divergence, which is rung 5 of the verification ladder (#21).

## Sub-features

- `ref-build` builds llama.cpp at `tools/llama.cpp/PIN`, CPU only, with the x86-64 ISA pinned to AVX2 (no AVX-512, no AMX). `BUILD_INFO` records the commit, the cmake flags, the llama.cpp args (`-nr`) and llama.cpp's `system_info` line.
- `ref-dump` writes `index.tsv`, `*.f32` and `logits.f32` for stories15M-q8_0 and Qwen3-0.6B-Q8_0 at 1, 17 and 128 tokens.
- `ref-kld-llama` records llama-perplexity's own `--kl-divergence` result in `kld/self.log`; it must be within 1e-6 of 0.
- `ref-determinism` reruns the Qwen3 17-token dump with all threads and gets the same `SHA256SUMS`.
- `ref-kld-self` gives mean KLD 0 and 100% same top for `compare` of a dump against itself.
- `ref-kld-perturbed` gives KLD above 0 for `compare --perturb 1e-3`, so `--max-kld 0` fails with exit 1.

## How to get to it (user POV)

- Run `tools/llama.cpp/run.sh`, then `cargo run --release -p bench --bin compare -- REF_DIR TEST_DIR`.

## Driving it with verify.sh

Preconditions:

- `verify.sh models` passed, and `git`, `cmake` and `c++` are installed.

- **Run.** Run `verify.sh reference`. The output shows `PASS run.sh dumped ...` for both models, `PASS ... dump records the llama.cpp commit`, `PASS ... logits vs themselves: mean KLD 0` and `PASS ... perturbed copy: KLD above 0, so the check fails`, then `RESULT: PASS (reference)`.
- **Proof.** `reference-run.stdout` and the `reference-self-*` and `reference-perturb-*` files in the artifacts directory. The dumps themselves are under `models/reference/llama.cpp/`.

## Gotchas

- Without `-nr` on a `GGML_NATIVE` build, an AMX CPU repacks Q8_0 weights and two 8-thread runs differed (Qwen3 128 tokens, mean KLD 6.1e-3). `run.sh` passes `-nr` and `build.sh` turns AMX off, so dumps are bit-identical at any thread count.
- llama-perplexity prints its KLD to 6 decimals; Qwen3 shows 0.000001, so the bound is 1e-6. The exact zero check is `compare` on the f32 dumps.
- `all` does not run this step, because it needs the models and a C++ toolchain.
