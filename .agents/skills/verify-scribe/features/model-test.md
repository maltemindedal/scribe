# Real-model parser test

The ignored test `parses_qwen3_0_6b_q8_0` parses Qwen3-0.6B-Q8_0 from a fully read buffer and checks its tensor count, metadata count, vocabulary size, and tensor shapes.

The ignored test `open_qwen3_does_not_resident_the_file` maps the same file. It checks that VmRSS stays below the header size plus 32 MiB, that the mapping is read-only and private, and that `output_weight` is the `token_embd.weight` bytes. It prints one `rss ` line with the measured KiB counts.

## Sub-features

- `parse-qwen3` runs `parses_qwen3_0_6b_q8_0`, which reports `ok`.
- `resident-qwen3` runs `open_qwen3_does_not_resident_the_file`, which reports `ok` and prints the `rss ` line.
- `resident-qwen3-eager` runs that same test with `SCRIBE_EAGER_READ=1`. The test opens with `MapOptions { populate: true }`. On Linux that prefaults the file inside `open`, the resident-set assertion fails, and the process exits non-zero. On other operating systems `MAP_POPULATE` is ignored, so the test reads the mapping `open` returned and the same assertion fails.

## How to get to it (user POV)

- Run `cargo test -p gguf -- --ignored` (see `docs/guides/running-the-model-tests.md`).
- To see the `rss ` line, add `--nocapture`.

## Driving it with verify.sh

Preconditions:

- `verify.sh models` passed, so `Qwen3-0.6B-Q8_0.gguf` is in the models directory.

- **Run.** Run `verify.sh model-test`. The output shows `PASS parses_qwen3_0_6b_q8_0 passes`, `PASS open_qwen3_does_not_resident_the_file passes`, and `PASS eager populate of the model file fails the resident-set check`.
- **Proof.** `model-test.stdout`, `model-test.rss`, and `model-test-eager.exit` in the artifacts directory. `model-test.rss` is the `rss ` line. `model-test-eager.exit` is not 0.

## Gotchas

- Without the model the tests fail rather than skip. Run `models` first.
- `SCRIBE_MODELS` must be the same for both `models` and `model-test`.
- `SCRIBE_EAGER_READ` is the negative run only. A normal `model-test` sets it for the eager invocation and leaves the passing invocation unset.
- The resident-set bound is the header size plus 32 MiB. The file is 624460 KiB and the header is 5812 KiB. Five runs on the machine that added the test grew RssFile by 5312 KiB and VmRSS by 5356 KiB, with no spread. Touching the 512-byte norm tensor added 64 KiB each run. Opening with `populate: true` on Linux prefaults the file and is far above the bound.
