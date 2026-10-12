# scribe CLI

The `scribe` binary has no commands yet. It prints its version and help, and it rejects anything else with exit code 1.

## Sub-features

- `cli-version` prints `scribe <workspace version>` to stdout and exits 0.
- `cli-help` prints help starting with `Usage: scribe <COMMAND>` for `-h`/`--help`.
- `cli-noargs` prints the same help with no arguments.
- `cli-unknown` prints `error: unknown command 'foo'` plus the help to stderr and exits 1.

## How to get to it (user POV)

- Run `cargo run -p cli -- <ARGS>`, or `target/debug/scribe <ARGS>` after `cargo build`.

## Driving it with verify.sh

Preconditions:

- `verify.sh doctor` passes.

- **Drive all four.** Run `verify.sh cli`. Each sub-feature prints a `PASS` line, then `RESULT: PASS (cli)`.
- **Proof.** `cli-*.cmd`, `.stdout`, `.stderr` and `.exit` in the artifacts directory.

## Gotchas

- Only the first argument is read. `scribe --version extra` still prints the version.
- The expected version is read from the root `Cargo.toml`, so a version bump needs no change here.
- When real commands land (for example `serve` for #36), add them here and to `verify.sh`.
