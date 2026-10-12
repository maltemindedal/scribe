# CLI reference

The `scribe` binary is built from the `cli` package
([`crates/cli/src/main.rs`](../../crates/cli/src/main.rs)). Run it through
Cargo with `cargo run -p cli -- <ARGS>`, or directly as `target/debug/scribe`
after `cargo build`.

```text
Usage: scribe <COMMAND>
```

No commands are implemented yet. Planned commands are tracked on the
[roadmap](https://github.com/maltemindedal/scribe/issues/1).

## Options

| Option | Effect | Exit code |
|---|---|---|
| `-h`, `--help` | Prints the help text to stdout. | 0 |
| `-V`, `--version` | Prints `scribe <version>`, for example `scribe 0.1.0`. The version comes from the workspace `Cargo.toml`. | 0 |
| _(no arguments)_ | Same as `--help`. | 0 |

## Errors

Any other first argument is rejected:

```text
$ scribe foo
error: unknown command 'foo'
```

The error and the help text go to stderr, and the exit code is 1.

Only the first argument is read; anything after it is ignored.
