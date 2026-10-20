# Real-model parser test

The ignored test `parses_qwen3_0_6b_q8_0` parses Qwen3-0.6B-Q8_0 from a fully read buffer and checks its tensor count, metadata count, vocabulary size, and tensor shapes.

The ignored test `tensor_bytes_over_every_qwen3_tensor_is_a_lookup` opens the same file and checks that a byte lookup of every tensor matches the lengths in the tensor table.

The ignored test `open_qwen3_does_not_resident_the_file` maps the same file. It checks that VmRSS stays below the header size plus 32 MiB, that the mapping is read-only and private, and that `output_weight` is the `token_embd.weight` bytes. It prints one `rss ` line with the measured KiB counts.

The ignored test `card_matches_committed_goldens` compares `card::to_json` with `crates/gguf/tests/golden/<stem>.json` for the four CI models.

The ignored test `seeded_header_mutations_do_not_panic` copies each of those headers and runs 3000 seeded mutations through `Gguf::parse`. It passes when every call returns. It does not count how many buffers parse. `verify.sh model-test` and CI run this test with `--profile mutation`. That profile inherits release and keeps overflow checks and debug assertions. The other ignored tests stay on the debug profile.

The ignored test `stories260k_every_header_prefix_is_an_error` checks that `stories260K.gguf` parses, and that every prefix of its header, including the exact header, is an error. If that header is longer than 64 KiB, the test fails before the loop.

## Sub-features

- `parse-qwen3` runs `parses_qwen3_0_6b_q8_0`, which reports `ok`.
- `resident-qwen3` runs `open_qwen3_does_not_resident_the_file`, which reports `ok` and prints the `rss ` line.
- `resident-qwen3-eager` runs that same test with `SCRIBE_EAGER_READ=1`. The test opens with `MapOptions { populate: true }`. On Linux that prefaults the file inside `open`, the resident-set assertion fails, and the process exits non-zero. On other operating systems `MAP_POPULATE` is ignored, so the test reads the mapping `open` returned and the same assertion fails.
- `card-goldens` runs `card_matches_committed_goldens`, which reports `ok`.
- `header-mutations` runs `seeded_header_mutations_do_not_panic`, which reports `ok`.
- `stories260k-prefixes` runs `stories260k_every_header_prefix_is_an_error`, which reports `ok`.

## How to get to it (user POV)

- Run the two `cargo test` commands in section 2 of `docs/guides/running-the-model-tests.md`.
- To see the `rss ` line, add `--nocapture` to the debug command.

## Driving it with verify.sh

Preconditions:

- `verify.sh models` passed, so the four CI GGUF files are in the models directory.

- **Run.** Run `verify.sh model-test`. The output shows `PASS parses_qwen3_0_6b_q8_0 passes`, `PASS open_qwen3_does_not_resident_the_file passes`, `PASS card_matches_committed_goldens passes`, `PASS seeded_header_mutations_do_not_panic passes`, `PASS stories260k_every_header_prefix_is_an_error passes`, and `PASS eager populate of the model file fails the resident-set check`.
- **Proof.** `model-test.stdout`, `model-test.rss`, and `model-test-eager.exit` in the artifacts directory. `model-test.rss` is the `rss ` line. `model-test-eager.exit` is not 0.

## Gotchas

- Without the four CI models the tests fail rather than skip. Run `models` first.
- `SCRIBE_MODELS` must be the same for both `models` and `model-test`.
- `SCRIBE_EAGER_READ` is the negative run only. A normal `model-test` sets it for the eager invocation and leaves the passing invocation unset.
- The resident-set bound is the header size plus 32 MiB. The file is 624460 KiB and the header is 5812 KiB. Five runs on the machine that added the test grew RssFile by 5312 KiB and VmRSS by 5356 KiB, with no spread. Touching the 512-byte norm tensor added 64 KiB each run. Opening with `populate: true` on Linux prefaults the file and is far above the bound.
