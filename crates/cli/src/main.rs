//! The `scribe` command-line tool.
#![forbid(unsafe_code)]

use std::process::ExitCode;

const HELP: &str = "\
scribe: a from-scratch LLM inference engine

Usage: scribe <COMMAND>

No commands are implemented yet; see the roadmap at
https://github.com/maltemindedal/scribe/issues/1

Options:
  -h, --help     Print help
  -V, --version  Print version
";

fn main() -> ExitCode {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        None | Some("-h" | "--help") => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        Some("-V" | "--version") => {
            println!("scribe {}", env!("CARGO_PKG_VERSION"));
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("error: unknown command '{other}'\n\n{HELP}");
            ExitCode::FAILURE
        }
    }
}
