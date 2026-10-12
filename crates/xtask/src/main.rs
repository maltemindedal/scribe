//! Developer tasks, run as `cargo xtask <task>`.
//!
//! Today: `cargo xtask models fetch [--set ci|all]`, which downloads the
//! pinned test models listed in `models.toml` into `models/` and verifies
//! their SHA-256.
#![forbid(unsafe_code)]

mod fetch;
mod manifest;
mod sha256;

use std::path::PathBuf;
use std::process::ExitCode;

use manifest::Set;

const HELP: &str = "\
Usage: cargo xtask models fetch [--set ci|all]

Downloads the pinned test models into models/ (or $SCRIBE_MODELS) and
verifies size and SHA-256. Files that are already present and valid are
left alone.

Options:
  --set ci|all  ci: the four GGUF files CI uses (default)
                all: ci plus the large reference models
  -h, --help    Print help
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args.as_slice() {
        [] | ["-h" | "--help"] => {
            print!("{HELP}");
            ExitCode::SUCCESS
        }
        ["models", "fetch", rest @ ..] => match parse_set(rest).and_then(run_fetch) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("error: unrecognised arguments\n\n{HELP}");
            ExitCode::from(2)
        }
    }
}

fn parse_set(args: &[&str]) -> Result<Set, String> {
    match args {
        [] => Ok(Set::Ci),
        ["--set", "ci"] => Ok(Set::Ci),
        ["--set", "all"] => Ok(Set::All),
        ["--set", other] => Err(format!("unknown set `{other}` (expected ci or all)")),
        _ => Err(format!("unrecognised arguments\n\n{HELP}")),
    }
}

fn models_dir() -> PathBuf {
    match std::env::var_os("SCRIBE_MODELS") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"),
    }
}

fn run_fetch(set: Set) -> Result<(), String> {
    let models = manifest::parse(manifest::MANIFEST)?;
    let dir = models_dir();
    let mut failures = 0;
    for model in manifest::select(&models, set) {
        println!("checking {} ({} bytes)", model.dest, model.size);
        match fetch::ensure(&dir, model, &fetch::curl) {
            Ok(fetch::Outcome::Present) => println!("ok       {} (already present)", model.dest),
            Ok(fetch::Outcome::Downloaded) => println!("fetched  {}", model.dest),
            Err(e) => {
                eprintln!("FAILED   {}: {e}", model.dest);
                failures += 1;
            }
        }
    }
    if failures > 0 {
        return Err(format!("{failures} model(s) failed"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_defaults_to_ci_and_rejects_unknown() {
        assert_eq!(parse_set(&[]), Ok(Set::Ci));
        assert_eq!(parse_set(&["--set", "all"]), Ok(Set::All));
        assert!(parse_set(&["--set", "huge"]).is_err());
        assert!(parse_set(&["--bogus"]).is_err());
    }
}
