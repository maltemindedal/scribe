//! Developer tasks, run as `cargo xtask <task>`.
//!
//! Today: `cargo xtask models fetch [--set ci|all]`, which downloads the
//! pinned test models listed in `models.toml` into `models/` and verifies
//! their SHA-256; `cargo xtask docs-check [root]`, which checks the prose
//! against the code (see `docs.rs`); and `cargo xtask pr-check`, which checks
//! a pull request's merge-method section.
#![forbid(unsafe_code)]

mod docs;
mod fetch;
mod manifest;
mod pr_check;
mod sha256;

use std::path::PathBuf;
use std::process::ExitCode;

use manifest::Set;

const HELP: &str = "\
Usage: cargo xtask models fetch [--set ci|all]
       cargo xtask docs-check [root]
       cargo xtask pr-check --base SHA --head SHA [--body-file PATH]

Downloads the pinned test models into models/ (or $SCRIBE_MODELS) and
verifies size and SHA-256. Files that are already present and valid are
left alone.

docs-check fails when prose restates a fact outside its owner, names
dependencies Cargo does not have, or miscounts the ignored tests. root
defaults to the repository root.

Options:
  --set ci|all  ci: the four GGUF files CI uses (default)
                all: ci plus the large reference models
  -h, --help    Print help

pr-check fails (exit 1, one line per error) unless the pull request body has
a `## Merge method` section that names merge, squash or rebase, gives a
reason, and fits the commits in base..head. The body is read from
--body-file, else from $PR_BODY.
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
        ["docs-check", rest @ ..] if rest.len() <= 1 => {
            let root = rest.first().map_or_else(repo_root, PathBuf::from);
            match docs::run(&root) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        ["pr-check", rest @ ..] => run_pr_check(rest),
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

fn run_pr_check(args: &[&str]) -> ExitCode {
    let result = parse_pr_check(args).and_then(|(base, head, body_file)| {
        let body = match body_file {
            Some(path) => {
                std::fs::read_to_string(&path).map_err(|e| format!("cannot read {path}: {e}"))?
            }
            None => std::env::var("PR_BODY").unwrap_or_default(),
        };
        let (commits, added_md_lines) = pr_check::gather(&base, &head)?;
        Ok(pr_check::check(&body, &commits, &added_md_lines))
    });
    match result {
        Ok(errors) if errors.is_empty() => {
            println!("pr-check: ok");
            ExitCode::SUCCESS
        }
        Ok(errors) => {
            for error in errors {
                eprintln!("pr-check: {error}");
            }
            ExitCode::FAILURE
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

/// `--base SHA --head SHA [--body-file PATH]`, in any order.
fn parse_pr_check(args: &[&str]) -> Result<(String, String, Option<String>), String> {
    let (mut base, mut head, mut body_file) = (None, None, None);
    let mut it = args.iter();
    while let Some(&flag) = it.next() {
        let slot = match flag {
            "--base" => &mut base,
            "--head" => &mut head,
            "--body-file" => &mut body_file,
            other => return Err(format!("unrecognised argument `{other}`\n\n{HELP}")),
        };
        *slot = Some(
            it.next()
                .ok_or(format!("{flag} needs a value"))?
                .to_string(),
        );
    }
    match (base, head) {
        (Some(base), Some(head)) => Ok((base, head, body_file)),
        _ => Err(format!("pr-check needs --base and --head\n\n{HELP}")),
    }
}

/// `$SCRIBE_MODELS`, else `models/` at the repository root.
///
/// The default is fixed at compile time on purpose: the xtask only runs from
/// a checkout through `cargo xtask`, and the model tests in
/// `crates/gguf/tests` resolve the same directory the same way, so the two
/// always agree.
fn models_dir() -> PathBuf {
    match std::env::var_os("SCRIBE_MODELS") {
        Some(dir) => PathBuf::from(dir),
        None => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"),
    }
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
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

    #[test]
    fn pr_check_takes_base_and_head_in_any_order() {
        let parsed = parse_pr_check(&["--head", "h", "--base", "b", "--body-file", "f"]);
        assert_eq!(parsed, Ok(("b".into(), "h".into(), Some("f".into()))));
        assert!(parse_pr_check(&["--base", "b"]).is_err());
        assert!(parse_pr_check(&["--base", "b", "--head"]).is_err());
        assert!(parse_pr_check(&["--base", "b", "--head", "h", "--x"]).is_err());
    }
}
