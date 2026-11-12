//! The workspace forbids `unsafe_code` through `[workspace.lints]`, and a
//! member only gets that table when its manifest says
//! `[lints] workspace = true`. A member without the line compiles `unsafe`
//! with no diagnostic, so this test reads every manifest.
//!
//! Cargo cannot override one key of an inherited table. The crates in
//! `UNSAFE_CRATES` therefore carry their own copy, which must equal the
//! workspace tables with `unsafe_code` left out.

#![allow(
    clippy::expect_used,
    reason = "test code: a failed expect is the test failing"
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const UNSAFE_CRATES: &[&str] = &["kernels", "mapfile"];

const MEMBERS: &str = r#"members = ["crates/*"]"#;

/// Section header to its sorted `key = value` lines. Comments and blank
/// lines are dropped, so two tables compare equal when they set the same
/// lints, in any order.
type Sections = BTreeMap<String, Vec<String>>;

fn sections(manifest: &str) -> Sections {
    let mut out = Sections::new();
    let mut current = String::new();
    for line in manifest.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            current = line.to_owned();
            out.entry(current.clone()).or_default();
        } else {
            out.entry(current.clone())
                .or_default()
                .push(line.to_owned());
        }
    }
    for lines in out.values_mut() {
        lines.sort();
    }
    out
}

/// The `[<prefix>.<tool>]` tables of a manifest, keyed by `<tool>`.
fn lint_tables(sections: &Sections, prefix: &str) -> BTreeMap<String, Vec<String>> {
    sections
        .iter()
        .filter_map(|(header, lines)| {
            let tool = header.strip_prefix(prefix)?.strip_suffix(']')?;
            Some((tool.to_owned(), lines.clone()))
        })
        .collect()
}

/// One line per manifest that breaks the policy. Each line names the fix.
fn problems(workspace: &str, members: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    let workspace = sections(workspace);
    let mut expected = lint_tables(&workspace, "[workspace.lints.");
    let forbid = r#"unsafe_code = "forbid""#;
    let forbids = expected
        .get("rust")
        .is_some_and(|lines| lines.iter().any(|line| line == forbid));
    if !forbids {
        out.push(format!(
            "Cargo.toml: [workspace.lints.rust] must set `{forbid}`"
        ));
    }
    let globs = workspace
        .get("[workspace]")
        .is_some_and(|lines| lines.iter().any(|line| line == MEMBERS));
    if !globs {
        out.push(format!(
            "Cargo.toml: [workspace] must say `{MEMBERS}`. This test reads only crates/*, so a member anywhere else is not checked."
        ));
    }
    if let Some(rust) = expected.get_mut("rust") {
        rust.retain(|line| !line.starts_with("unsafe_code"));
    }

    for (name, manifest) in members {
        let manifest = sections(manifest);
        let inherits = manifest
            .get("[lints]")
            .is_some_and(|lines| lines.as_slice() == ["workspace = true"]);
        let own = lint_tables(&manifest, "[lints.");
        if UNSAFE_CRATES.contains(&name.as_str()) {
            if inherits || own != expected {
                out.push(format!(
                    "crates/{name}/Cargo.toml: this crate is allowed `unsafe`, so it cannot inherit the workspace lints. Its [lints.*] tables must equal the [workspace.lints.*] tables with `unsafe_code` left out. Copy them again from the root Cargo.toml."
                ));
            }
        } else if !inherits || !own.is_empty() {
            out.push(format!(
                "crates/{name}/Cargo.toml: must contain exactly `[lints]` with `workspace = true`. Without it `{forbid}` does not apply to this crate. Only {UNSAFE_CRATES:?} may carry their own lint tables."
            ));
        }
    }
    out
}

#[test]
fn every_member_takes_the_workspace_lints_or_is_an_unsafe_crate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workspace = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let mut members = Vec::new();
    for entry in fs::read_dir(root.join("crates")).unwrap() {
        let dir = entry.unwrap().path();
        if let Ok(manifest) = fs::read_to_string(dir.join("Cargo.toml")) {
            let name = dir.file_name().unwrap().to_str().unwrap().to_owned();
            members.push((name, manifest));
        }
    }
    for name in UNSAFE_CRATES {
        assert!(members.iter().any(|(member, _)| member == name), "{name}");
    }
    let found = problems(&workspace, &members);
    assert!(found.is_empty(), "\n{}", found.join("\n"));
}

const WORKSPACE: &str = r#"
[workspace]
members = ["crates/*"]

# A comment.
[workspace.lints.rust]
unsafe_code = "forbid"
unsafe_op_in_unsafe_fn = "deny"

[workspace.lints.clippy]
all = { level = "warn", priority = -1 }
panic = "deny"
"#;

const INHERITS: &str = "[package]\nname = \"m\"\n\n[lints]\nworkspace = true\n";

const OWN_TABLES: &str = r#"
[package]
name = "m"

[lints.rust]
unsafe_op_in_unsafe_fn = "deny"

[lints.clippy]
panic = "deny"
all = { level = "warn", priority = -1 }
"#;

/// `text` with its one `from` replaced by `to`.
fn swap(text: &str, from: &str, to: &str) -> String {
    let (before, after) = text.split_once(from).expect("the fixture contains `from`");
    format!("{before}{to}{after}")
}

fn check(workspace: &str, name: &str, manifest: &str) -> Vec<String> {
    problems(workspace, &[(name.to_owned(), manifest.to_owned())])
}

#[test]
fn the_policy_accepts_an_inheriting_member_and_a_matching_unsafe_crate() {
    assert_eq!(check(WORKSPACE, "sampler", INHERITS), [""; 0]);
    assert_eq!(check(WORKSPACE, "kernels", OWN_TABLES), [""; 0]);
}

#[test]
fn a_member_without_the_lints_table_is_reported() {
    let found = check(WORKSPACE, "sampler", "[package]\nname = \"m\"\n");
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].starts_with("crates/sampler/Cargo.toml"),
        "{found:?}"
    );
}

#[test]
fn a_member_outside_the_unsafe_crates_with_its_own_tables_is_reported() {
    let found = check(WORKSPACE, "sampler", OWN_TABLES);
    assert_eq!(found.len(), 1, "{found:?}");
}

#[test]
fn an_unsafe_crate_whose_tables_drift_is_reported() {
    for drifted in [
        swap(OWN_TABLES, "panic = \"deny\"\n", ""),
        swap(OWN_TABLES, "\"deny\"\n\n", "\"allow\"\n\n"),
        format!("{OWN_TABLES}todo = \"allow\"\n"),
        format!("{OWN_TABLES}\n[lints.rustdoc]\nall = \"allow\"\n"),
        swap(OWN_TABLES, "[lints.clippy]", "[lints.clippy2]"),
    ] {
        let found = check(WORKSPACE, "mapfile", &drifted);
        assert_eq!(found.len(), 1, "{drifted}\n{found:?}");
    }
}

#[test]
fn a_workspace_that_stops_forbidding_unsafe_is_reported() {
    let relaxed = swap(WORKSPACE, "\"forbid\"", "\"deny\"");
    let found = check(&relaxed, "sampler", INHERITS);
    assert_eq!(found.len(), 1, "{found:?}");
}

#[test]
fn a_workspace_with_members_outside_crates_is_reported() {
    let wider = swap(WORKSPACE, "[\"crates/*\"]", "[\"crates/*\", \"tools/x\"]");
    let found = check(&wider, "sampler", INHERITS);
    assert_eq!(found.len(), 1, "{found:?}");
}
